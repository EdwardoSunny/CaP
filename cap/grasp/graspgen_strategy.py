"""GraspGen-based grasp generation strategy.

Uses the NVIDIA GraspGen diffusion model to generate grasp poses from a
segmented object point cloud, then filters and selects the best grasp.
"""

import logging
import traceback
from typing import Optional

import numpy as np
import scipy.spatial.transform as st
import torch
import trimesh.transformations as tra

from grasp_gen.grasp_server import GraspGenSampler, load_grasp_cfg
from grasp_gen.utils.point_cloud_utils import point_cloud_outlier_removal

from cap.grasp.base import GraspResult, GraspStrategy

logger = logging.getLogger(__name__)


class GraspGenStrategy(GraspStrategy):
    """Grasp strategy using the NVIDIA GraspGen diffusion model.

    Loads a pretrained GraspGen checkpoint and runs inference on a segmented
    object point cloud to produce candidate 6-DOF grasp poses.  The best
    grasp is selected by preferring more vertical (top-down) orientations
    and proximity to the current end-effector pose.
    """

    def __init__(self, gripper_config_path: str):
        """
        Initialize GraspGen model and parameters.

        Args:
            gripper_config_path: Path to the GraspGen gripper YAML config file
                (e.g. ``GraspGen/GraspGenModels/checkpoints/graspgen_robotiq_2f_140.yml``).
        """
        from pathlib import Path

        print("Loading GraspGen model...")
        gripper_config = str(gripper_config_path)
        if not Path(gripper_config).exists():
            logger.warning(
                f"GraspGen config not found at {gripper_config}. "
                "Grasp generation will be disabled."
            )
            self.grasp_sampler = None
            self.gripper_name = None
        else:
            grasp_cfg = load_grasp_cfg(gripper_config)
            self.gripper_name = grasp_cfg.data.gripper_name
            self.grasp_sampler = GraspGenSampler(grasp_cfg)
            print(f"GraspGen initialized successfully for gripper: {self.gripper_name}")

        # GraspGen parameters
        self.grasp_threshold = 0.8
        self.num_grasps = 200
        self.topk_num_grasps = -1

        # Gripper depth offset: GraspGen was trained on a Robotiq 2F-140 which has
        # longer fingers than the actual xArm gripper. The grasp center (midpoint
        # between fingertips) sits ~160mm higher on the Robotiq. We compensate by
        # shifting each grasp pose forward along its approach axis (Z) by this amount,
        # moving the grasp center down to where our shorter gripper's fingertips are.
        self.grasp_z_offset_m = 0.160  # meters, positive = closer to object

        # CRITICAL: Transform between robot TCP frame and GraspGen convention
        # GraspGen assumes: X = finger closing, Y = width, Z = approach
        # Real robot has:   Y = finger closing, X = width, Z = approach
        # Solution: Rotate -90° around Z to go from GraspGen -> Robot TCP
        # This is applied AFTER grasps are generated to convert them to robot frame
        self.T_graspgen_to_tcp = tra.rotation_matrix(-np.pi / 2, [0, 0, 1])
        logger.info("Gripper frame correction: GraspGen -> Robot TCP = -90° around Z")
        logger.info("  GraspGen: fingers along X, approach along Z")
        logger.info("  Robot:    fingers along Y, approach along Z")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_grasps(
        self,
        object_points: np.ndarray,
        object_colors: Optional[np.ndarray] = None,
        scene_points: Optional[np.ndarray] = None,
        scene_colors: Optional[np.ndarray] = None,
        current_eef_pose: Optional[np.ndarray] = None,
    ) -> GraspResult:
        """
        Generate and select the best grasp for the given object point cloud.

        Combines point-cloud pre-processing, GraspGen inference, and
        filter-and-select into a single call.

        Args:
            object_points: Nx3 segmented object points in robot frame (meters).
            object_colors: Nx3 RGB colours [0-1] (optional).
            scene_points: Mx3 full scene points (optional, kept for viz).
            scene_colors: Mx3 full scene colours (optional).
            current_eef_pose: 4x4 current EEF transform (optional,
                used for proximity-based selection and Z-symmetry
                normalisation).

        Returns:
            :class:`GraspResult` with the selected grasp and metadata.
        """
        grasps, scores = self._generate_grasps_on_pointcloud(object_points)

        if grasps is None or scores is None:
            return GraspResult(
                object_points=object_points,
                object_colors=object_colors,
                scene_points=scene_points,
                scene_colors=scene_colors,
            )

        best_idx, best_grasp, best_score, grasps, scores = (
            self._filter_and_select_best_grasp(grasps, scores, current_eef_pose)
        )

        if best_grasp is None:
            return GraspResult(
                object_points=object_points,
                object_colors=object_colors,
                scene_points=scene_points,
                scene_colors=scene_colors,
            )

        return GraspResult(
            best_grasp=best_grasp,
            best_score=best_score,
            best_idx=best_idx,
            all_grasps=grasps,
            all_scores=scores,
            object_points=object_points,
            object_colors=object_colors,
            scene_points=scene_points,
            scene_colors=scene_colors,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _generate_grasps_on_pointcloud(self, points):
        """
        Generate grasps on a point cloud using GraspGen.

        Pipeline:
        1. Statistical outlier removal (remove scattered noise from depth/segmentation)
        2. DBSCAN clustering (keep only largest cluster — removes mask bleeding)
        3. Downsample if too many points (outlier removal is O(N^2) memory)
        4. Center point cloud (GraspGen expects centered input)
        5. GraspGen's own outlier removal (KNN-based)
        6. Run GraspGen inference
        7. Un-center grasps back to robot frame
        8. Apply gripper frame correction (GraspGen -> robot TCP convention)

        Args:
            points: Nx3 numpy array of 3D points in robot frame (meters)

        Returns:
            tuple: (grasps, scores) - Nx4x4 poses in robot frame, N confidence scores
                   Returns (None, None) if generation fails
        """
        if self.grasp_sampler is None:
            logger.warning("GraspGen not initialized, cannot generate grasps")
            return None, None

        try:
            import open3d as o3d

            logger.info(f"Pre-processing segmented point cloud: {len(points)} raw points")

            # --- Step 1: Statistical outlier removal ---
            # Removes scattered noise from depth sensor / stereo matching edges
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd_clean, stat_inlier_idx = pcd.remove_statistical_outlier(
                nb_neighbors=20, std_ratio=2.0
            )
            n_stat_removed = len(points) - len(pcd_clean.points)
            logger.info(f"  Statistical outlier removal: {len(points)} -> {len(pcd_clean.points)} ({n_stat_removed} removed)")

            # --- Step 2: DBSCAN clustering — keep only largest cluster ---
            # Removes disconnected blobs from SAM mask bleeding into background
            if len(pcd_clean.points) > 10:
                labels = np.array(pcd_clean.cluster_dbscan(
                    eps=0.015,  # 15mm neighborhood radius
                    min_points=10,
                    print_progress=False,
                ))
                if len(labels) > 0 and labels.max() >= 0:
                    # Find largest cluster
                    unique_labels, counts = np.unique(labels[labels >= 0], return_counts=True)
                    largest_label = unique_labels[counts.argmax()]
                    cluster_mask = labels == largest_label
                    n_before_cluster = len(pcd_clean.points)
                    pcd_clean = pcd_clean.select_by_index(np.where(cluster_mask)[0])
                    n_cluster_removed = n_before_cluster - len(pcd_clean.points)
                    logger.info(f"  DBSCAN clustering: kept largest cluster ({len(pcd_clean.points)} points, {n_cluster_removed} from smaller clusters removed)")
                else:
                    logger.info(f"  DBSCAN clustering: no valid clusters found, keeping all points")

            points = np.asarray(pcd_clean.points)
            logger.info(f"  After pre-processing: {len(points)} clean points")

            if len(points) == 0:
                logger.warning("No points remaining after pre-processing")
                return None, None

            # Downsample if too many points (GraspGen outlier removal uses O(N^2) memory)
            max_points = 5000
            if len(points) > max_points:
                logger.info(f"Downsampling {len(points)} -> ~{max_points} points (voxel grid)")
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points)
                bbox = pcd.get_axis_aligned_bounding_box()
                volume = np.prod(bbox.get_extent())
                voxel_size = (volume / max_points) ** (1.0 / 3.0)
                pcd_down = pcd.voxel_down_sample(voxel_size)
                points = np.asarray(pcd_down.points)
                logger.info(f"After downsampling: {len(points)} points (voxel_size={voxel_size:.4f} m)")

            # Center point cloud (GraspGen expects centered input)
            pc_mean = points.mean(axis=0)
            pc_centered = points - pc_mean

            # GraspGen's own KNN-based outlier removal
            pc_filtered, pc_removed = point_cloud_outlier_removal(
                torch.from_numpy(pc_centered)
            )
            pc_filtered = pc_filtered.numpy()
            logger.info(f"  GraspGen outlier removal: {len(pc_filtered)} kept, {len(pc_removed.numpy())} removed")

            # Run GraspGen inference
            grasps_inferred, grasp_conf_inferred = GraspGenSampler.run_inference(
                pc_filtered,
                self.grasp_sampler,
                grasp_threshold=self.grasp_threshold,
                num_grasps=self.num_grasps,
                topk_num_grasps=self.topk_num_grasps,
            )

            if len(grasps_inferred) == 0:
                logger.warning("GraspGen returned no valid grasps")
                return None, None

            # Convert to numpy
            grasp_conf_inferred = grasp_conf_inferred.cpu().numpy()
            grasps_inferred = grasps_inferred.cpu().numpy()
            grasps_inferred[:, 3, 3] = 1  # Ensure homogeneous coordinate

            # Un-center: shift grasps back to robot frame
            T_add_mean = tra.translation_matrix(pc_mean)
            grasps_robot = np.array([T_add_mean @ g for g in grasps_inferred])

            # Apply gripper frame correction: GraspGen fingers-along-X -> robot fingers-along-Y
            # Rotate -90 deg around Z
            grasps_robot = np.array([g @ self.T_graspgen_to_tcp for g in grasps_robot])

            # Apply gripper depth offset: GraspGen was trained on Robotiq 2F-140
            # (longer fingers) but we use a shorter xArm gripper. Shift each grasp
            # along its own approach axis (Z column of rotation) to compensate.
            if self.grasp_z_offset_m != 0.0:
                for i in range(len(grasps_robot)):
                    approach_axis = grasps_robot[i, :3, 2]  # Z column = approach direction
                    grasps_robot[i, :3, 3] += approach_axis * self.grasp_z_offset_m
                logger.info(f"Applied gripper depth offset: {self.grasp_z_offset_m*1000:.0f} mm along approach axis")

            logger.info(f"Generated {len(grasps_robot)} raw grasps (scores: {grasp_conf_inferred.min():.3f} - {grasp_conf_inferred.max():.3f})")
            return grasps_robot, grasp_conf_inferred

        except Exception as e:
            logger.error(f"Error generating grasps: {e}")
            logger.error(traceback.format_exc())
            return None, None

    def _filter_and_select_best_grasp(self, grasps, grasp_scores, current_eef_pose=None):
        """
        Filter grasps and select the best one (closest to current EEF).

        Pipeline:
        1. Normalize: gripper Z-symmetry (pick 180-flip closer to current EEF)
        2. Select: top-50% most vertical, then closest to EEF position

        Args:
            grasps: Nx4x4 numpy array of grasp poses in robot frame (meters)
            grasp_scores: N numpy array of GraspGen confidence scores [0-1]
            current_eef_pose: 4x4 current EEF transform (optional).
                If provided, used for Z-symmetry normalisation and
                proximity-based selection.

        Returns:
            tuple: (best_idx, best_grasp, best_score, grasps, scores)
                   Returns (None, None, None, None, None) if no grasps survive filtering
        """
        n_initial = len(grasps)
        logger.info(f"  {n_initial} grasps to select from (scores: {grasp_scores.min():.3f} - {grasp_scores.max():.3f})")

        # --- Get current EEF pose ---
        eef_transform = current_eef_pose  # Already a 4x4 matrix (or None)

        # --- Normalize: Gripper Z-symmetry ---
        # Parallel gripper is symmetric under 180 deg rotation around Z (approach).
        # Pick whichever orientation (original or +180 around Z) is closer to EEF.
        if eef_transform is not None:
            Rz180 = np.eye(4)
            Rz180[0, 0] = -1
            Rz180[1, 1] = -1  # 180 deg around Z

            eef_rot_scipy = st.Rotation.from_matrix(eef_transform[:3, :3])
            n_flipped = 0
            for i in range(len(grasps)):
                rot_orig = st.Rotation.from_matrix(grasps[i, :3, :3])
                rot_flip = rot_orig * st.Rotation.from_matrix(Rz180[:3, :3])
                delta_orig = (rot_orig * eef_rot_scipy.inv()).magnitude()
                delta_flip = (rot_flip * eef_rot_scipy.inv()).magnitude()
                if delta_flip < delta_orig:
                    grasps[i] = grasps[i] @ Rz180
                    n_flipped += 1
            logger.info(f"  Z-symmetry: flipped {n_flipped}/{len(grasps)} grasps to reduce rotation from EEF")

        # --- Select best: prefer more vertical grasps, then closest to EEF ---
        # "Top-down" = approach axis (Z column of grasp) aligned with world -Z.
        # Compute dot product of grasp Z-axis with [0, 0, -1]. Perfect top-down = 1.0.
        down = np.array([0.0, 0.0, -1.0])
        topdown_scores = np.array([np.dot(g[:3, 2], down) for g in grasps])

        # Keep the top 50% most vertical grasps (relative filter — always keeps half,
        # works whether grasps are top-down, angled, or horizontal)
        topdown_threshold = np.median(topdown_scores)
        topdown_mask = topdown_scores >= topdown_threshold
        n_topdown = topdown_mask.sum()
        logger.info(f"  Vertical preference: keeping {n_topdown}/{len(grasps)} grasps above median alignment ({topdown_threshold:.3f})")

        if eef_transform is not None:
            eef_pos = eef_transform[:3, 3]
            distances = np.linalg.norm(grasps[:, :3, 3] - eef_pos, axis=1)

            # From the more vertical half, pick closest to EEF
            masked_distances = np.where(topdown_mask, distances, np.inf)
            best_idx = int(np.argmin(masked_distances))
            logger.info(f"  Best grasp: #{best_idx} (closest to EEF from vertical set, distance={distances[best_idx]:.3f} m, alignment={topdown_scores[best_idx]:.3f})")
        else:
            # No EEF available — from the more vertical half, pick highest confidence
            masked_scores = np.where(topdown_mask, grasp_scores, -1)
            best_idx = int(np.argmax(masked_scores))
            logger.info(f"  Best grasp: #{best_idx} (highest score from vertical set, alignment={topdown_scores[best_idx]:.3f})")

        best_grasp = grasps[best_idx]
        best_score = grasp_scores[best_idx]

        best_pos = best_grasp[:3, 3]
        best_euler = st.Rotation.from_matrix(best_grasp[:3, :3]).as_euler("xyz", degrees=True)
        logger.info(f"  Position: [{best_pos[0]:.3f}, {best_pos[1]:.3f}, {best_pos[2]:.3f}] m")
        logger.info(f"  Orientation: [{best_euler[0]:.1f}, {best_euler[1]:.1f}, {best_euler[2]:.1f}] deg")
        logger.info(f"  Score: {best_score:.3f}")

        return best_idx, best_grasp, best_score, grasps, grasp_scores
