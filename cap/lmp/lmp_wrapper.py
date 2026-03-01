import time
import numpy as np
import scipy.spatial.transform as st
import logging
import traceback
from typing import List, Optional, Union, Tuple
from pathlib import Path
import torch
import shapely
import trimesh.transformations as tra
from cap.lmp.lmp import LMP, LMPFGen
from ril_env.precise_sleep import precise_wait
from ril_env.xarm_controller import XArmConfig, XArm
from ril_env.real_env import RealEnv

# Import the segmentation model loading function
from cap.segment_pc import load_segmentation_models

# Import GraspGen
from grasp_gen.grasp_server import GraspGenSampler, load_grasp_cfg
from grasp_gen.utils.point_cloud_utils import point_cloud_outlier_removal

logger = logging.getLogger(__name__)


class LMPWrapper:

    def __init__(self, env, xarm_config, frequency=30, command_latency=0.01, camera_serials=["327122079374", "317222072157"]):
        """
        Initialize robot primitives using teleop script's exact components.

        Args:
            env: RealEnv instance from teleop script
            xarm_config: XArmConfig instance from teleop script
            frequency: Control frequency in Hz
            command_latency: Command latency in seconds
        """
        self.env = env
        self._xarm_config = xarm_config
        self._frequency = frequency
        self._command_latency = command_latency
        self._dt = 1.0 / frequency
        self._current_grasp = 0.0

        # Get initial robot state - same as teleop script
        state = self.env.get_robot_state()
        self._current_pose = np.array(state["TCPPose"], dtype=np.float32)

        # LMP-specific additions
        self._setup_lmp_environment()

        logger.info(f"Robot Primitives initialized. Current pose: {self._current_pose}")

        # Store camera serials for point cloud capture
        self.camera_serials = camera_serials

        # Cache for detected objects (to avoid re-capturing for same command)
        self._detection_cache = {}  # {obj_name: {'full_pose': [x,y,z,r,p,y] in mm and deg}}

        # INITIALIZE SEGMENTATION MODELS GLOBALLY using new API
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading segmentation models on {self.device}...")

        # Get project root (assuming lmp is in cap/lmp/)
        project_root = Path(__file__).parent.parent.parent

        # Load models using the new API
        # Use SAM2's internal config (no path, just name) to avoid Hydra path issues
        self.sam_generator, self.clip_model, self.clip_processor = load_segmentation_models(
            sam2_checkpoint=str(project_root / "ckpt" / "sam2.1_hiera_large.pt"),
            sam2_config="sam2.1/sam2.1_hiera_l",  # Use SAM2 package's internal config
            clip_model_name="laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
            device=self.device
        )
        print("All segmentation models initialized successfully.")

        # INITIALIZE GRASPGEN MODEL
        print("Loading GraspGen model...")
        gripper_config = str(project_root / "GraspGen" / "GraspGenModels" / "checkpoints" / "graspgen_robotiq_2f_140.yml")
        if not Path(gripper_config).exists():
            logger.warning(f"GraspGen config not found at {gripper_config}. Grasp generation will be disabled.")
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
        self.visualize_top_k = 10  # Only visualize top K grasps

        # Gripper depth offset: GraspGen was trained on a Robotiq 2F-140 which has
        # longer fingers than the actual xArm gripper. The grasp center (midpoint
        # between fingertips) sits ~160mm higher on the Robotiq. We compensate by
        # shifting each grasp pose forward along its approach axis (Z) by this amount,
        # moving the grasp center down to where our shorter gripper's fingertips are.
        self.grasp_z_offset_m = 0.160  # meters, positive = closer to object (opposite to approach dir)

        # CRITICAL: Transform between robot TCP frame and GraspGen convention
        # GraspGen assumes: X = finger closing, Y = width, Z = approach
        # Real robot has:   Y = finger closing, X = width, Z = approach
        # Solution: Rotate -90° around Z to go from GraspGen -> Robot TCP
        # This is applied AFTER grasps are generated to convert them to robot frame
        self.T_graspgen_to_tcp = tra.rotation_matrix(-np.pi/2, [0, 0, 1])
        logger.info("Gripper frame correction: GraspGen -> Robot TCP = -90° around Z")
        logger.info("  GraspGen: fingers along X, approach along Z")
        logger.info("  Robot:    fingers along Y, approach along Z")

    def _setup_lmp_environment(self):
        """Setup LMP-specific environment variables and object tracking."""
        # Mock object tracking (replace with actual computer vision)
        self.known_objects = ["bread", "red ball", "white bottle", "red cup", "tissue box"]
        # Stored visualization data for before/after comparison
        self._last_viz_data = None

        # Workspace bounds for your robot (adjust these to match your setup)
        self.workspace_bounds = {
            "x_min": -0.3,
            "x_max": 0.3,
            "y_min": -0.8,
            "y_max": -0.2,
            "z_table": 0.0,
        }

        # Predefined positions for LMP commands
        self.corner_positions = {
            "top left corner": (-0.25, -0.25, 0),
            "top right corner": (0.25, -0.25, 0),
            "bottom left corner": (-0.25, -0.75, 0),
            "bottom right corner": (0.25, -0.75, 0),
        }

        self.side_positions = {
            "top side": (0, -0.25, 0),
            "bottom side": (0, -0.75, 0),
            "left side": (-0.25, -0.5, 0),
            "right side": (0.25, -0.5, 0),
        }

        # Color definitions for object recognition
        self.colors = {
            "red": (1.0, 0.0, 0.0, 1.0),
            "green": (0.0, 1.0, 0.0, 1.0),
            "blue": (0.0, 0.0, 1.0, 1.0),
            "yellow": (1.0, 1.0, 0.0, 1.0),
            "orange": (1.0, 0.5, 0.0, 1.0),
            "purple": (0.5, 0.0, 0.5, 1.0),
            "pink": (1.0, 0.75, 0.8, 1.0),
            "cyan": (0.0, 1.0, 1.0, 1.0),
            "brown": (0.6, 0.3, 0.1, 1.0),
            "gray": (0.5, 0.5, 0.5, 1.0),
        }

    def detect_object_location(self, prompt):
        """
        Detect and segment an object in the scene using text prompt.
        Then generate grasps on the segmented point cloud using GraspGen.
        Returns the best grasp pose for the LMP to use.

        Complete pipeline:
        1. Segment object with SAM+CLIP
        2. Generate grasps with GraspGen
        3. Visualize grasps (meshcat) - waits for user to close
        4. Return best grasp pose in robot frame (millimeters + degrees)

        Args:
            prompt: Text description of object to find (e.g., "cup", "bottle")

        Returns:
            np.array: [x, y, z, roll, pitch, yaw] best grasp pose in MILLIMETERS and DEGREES
                      Returns None if no grasps found
        """
        from cap.segment_pc import RobotFrameMerger

        # Get project root for paths
        project_root = Path(__file__).parent.parent.parent

        # Create RobotFrameMerger with the EXACT same initialization as segment_pc.py
        merger = RobotFrameMerger(
            camera_serials=self.camera_serials,
            calib_file=project_root / "transforms" / "transforms.npy",
            max_depth=2.0,
            min_depth=0.1,
            sam_generator=self.sam_generator,
            clip_model=self.clip_model,
            clip_processor=self.clip_processor,
            device=self.device,
        )

        try:
            # Capture full scene first (no segmentation) for visualization context
            logger.info("Capturing full scene point cloud...")
            full_points, full_colors = merger.capture_merged_pointcloud(text_prompt=None)

            # Capture segmented object
            logger.info(f"Segmenting '{prompt}'...")
            merged_points, merged_colors = merger.capture_merged_pointcloud(
                text_prompt=prompt
            )

            logger.info(f"Merged point cloud: {len(merged_points) if merged_points is not None else 0} total points")

            if merged_points is None or len(merged_points) == 0:
                logger.warning(f"No points found for object: {prompt}")
                return None

            # Generate grasps if GraspGen is available
            if self.grasp_sampler is not None:
                logger.info("Generating grasps on segmented point cloud...")
                grasps, grasp_scores = self._generate_grasps_on_pointcloud(merged_points)

                if grasps is not None and len(grasps) > 0:
                    # Filter grasps and select best (closest to EEF)
                    logger.info(f"Filtering and selecting best grasp...")
                    result = self._filter_and_select_best_grasp(grasps, grasp_scores)
                    best_idx, best_grasp, best_score, grasps_filtered, scores_filtered = result

                    if best_grasp is None:
                        logger.warning("No grasps survived filtering")
                        return None

                    # Convert 4x4 grasp matrix to [x, y, z, roll, pitch, yaw]
                    import scipy.spatial.transform as st
                    position_m = best_grasp[:3, 3]
                    position_mm = position_m * 1000.0
                    rotation_matrix = best_grasp[:3, :3]
                    rotation = st.Rotation.from_matrix(rotation_matrix)
                    orientation_deg = rotation.as_euler('xyz', degrees=True)

                    grasp_pose = np.concatenate([position_mm, orientation_deg])

                    logger.info(f"\nBest grasp pose for robot:")
                    logger.info(f"  Position (mm): [{position_mm[0]:.1f}, {position_mm[1]:.1f}, {position_mm[2]:.1f}]")
                    logger.info(f"  Orientation (deg): [{orientation_deg[0]:.1f}, {orientation_deg[1]:.1f}, {orientation_deg[2]:.1f}]")
                    logger.info(f"  Score: {best_score:.3f}")

                    # --- Interactive visualization before execution ---
                    self._visualize_before_execution(
                        full_points, full_colors,
                        merged_points, merged_colors,
                        grasps_filtered, scores_filtered,
                        best_idx, best_grasp, prompt,
                    )

                    # Store data for "after" visualization
                    self._last_viz_data = {
                        "full_points": full_points,
                        "full_colors": full_colors,
                        "seg_points": merged_points,
                        "seg_colors": merged_colors,
                        "grasps": grasps_filtered,
                        "scores": scores_filtered,
                        "best_idx": best_idx,
                        "best_grasp": best_grasp,
                        "prompt": prompt,
                    }

                    return grasp_pose

                else:
                    logger.warning("No grasps generated for this object")
                    return None
            else:
                logger.warning("GraspGen not initialized, cannot generate grasps")
                return None

        finally:
            # Always cleanup cameras
            merger.cleanup()

    def _visualize_before_execution(
        self, full_points, full_colors,
        seg_points, seg_colors,
        grasps, scores, best_idx, best_grasp, prompt,
    ):
        """
        Show an interactive Open3D visualization of the full pipeline output.
        Blocks until the user closes the window, then execution continues.

        Shows:
        - Full scene point cloud (original colors)
        - Segmented object (red-tinted)
        - All grasp arrows (green=high score, red=low)
        - Best grasp in purple with RGB axes + gripper fingers
        - Current EEF in cyan with RGB axes + gripper fingers
        - Robot base coordinate frame
        """
        import open3d as o3d
        import scipy.spatial.transform as spt

        logger.info("Opening visualization (close window to continue with execution)...")

        geometries = []

        # --- Full scene ---
        if full_points is not None and len(full_points) > 0:
            scene_pcd = o3d.geometry.PointCloud()
            scene_pcd.points = o3d.utility.Vector3dVector(full_points)
            scene_pcd.colors = o3d.utility.Vector3dVector(full_colors)
            geometries.append(scene_pcd)

        # --- Segmented object (red-tinted) ---
        if seg_points is not None and len(seg_points) > 0:
            seg_pcd = o3d.geometry.PointCloud()
            seg_pcd.points = o3d.utility.Vector3dVector(seg_points)
            red = np.array([1.0, 0.0, 0.0])
            tinted = seg_colors * 0.6 + red * 0.4
            seg_pcd.colors = o3d.utility.Vector3dVector(np.clip(tinted, 0, 1))
            geometries.append(seg_pcd)

        # --- Robot base coordinate frame ---
        robot_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15)
        geometries.append(robot_frame)

        # --- Helper: make arrow mesh ---
        def make_arrow(origin, direction, length=0.06, color=[1, 0, 0],
                       cyl_r=0.003, cone_r=0.006, cone_frac=0.3):
            cyl_h = length * (1 - cone_frac)
            cone_h = length * cone_frac
            arrow = o3d.geometry.TriangleMesh.create_arrow(
                cylinder_radius=cyl_r, cone_radius=cone_r,
                cylinder_height=cyl_h, cone_height=cone_h,
                resolution=8, cylinder_split=1, cone_split=1,
            )
            arrow.paint_uniform_color(color)
            arrow.compute_vertex_normals()

            direction = direction / np.linalg.norm(direction)
            z = np.array([0.0, 0.0, 1.0])
            v = np.cross(z, direction)
            s = np.linalg.norm(v)
            c = np.dot(z, direction)
            if s < 1e-8:
                R = np.eye(3) if c > 0 else np.diag([-1.0, -1.0, 1.0])
            else:
                vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
                R = np.eye(3) + vx + vx @ vx * (1 - c) / (s * s)
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = origin
            arrow.transform(T)
            return arrow

        # --- Helper: make RGB pose axes ---
        def make_pose_axes(transform_4x4, axis_len=0.08, cyl_r=0.004,
                          cone_r=0.008, label_color=None, label_r=0.012):
            axis_colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
            origin = transform_4x4[:3, 3]
            rot = transform_4x4[:3, :3]
            meshes = []
            for i, col in enumerate(axis_colors):
                meshes.append(make_arrow(origin, rot[:, i], length=axis_len,
                                         color=col, cyl_r=cyl_r, cone_r=cone_r))
            if label_color is not None:
                sp = o3d.geometry.TriangleMesh.create_sphere(radius=label_r)
                sp.paint_uniform_color(label_color)
                sp.compute_vertex_normals()
                sp.translate(origin)
                meshes.append(sp)
            return meshes

        # --- Helper: gripper fingers ---
        def make_fingers(grasp_4x4, color, width=0.08, depth=0.05, thick=0.012):
            pos = grasp_4x4[:3, 3]
            rot = grasp_4x4[:3, :3]
            meshes = []
            for sign in [1, -1]:
                box = o3d.geometry.TriangleMesh.create_box(width=thick, height=thick, depth=depth)
                box.paint_uniform_color(color)
                box.compute_vertex_normals()
                box_center = np.array([thick / 2, thick / 2, depth / 2])
                T = np.eye(4)
                T[:3, :3] = rot
                T[:3, 3] = pos + rot[:, 1] * sign * (width / 2) - rot @ box_center
                box.transform(T)
                meshes.append(box)
            palm = o3d.geometry.TriangleMesh.create_box(width=thick, height=width, depth=thick)
            palm.paint_uniform_color(color)
            palm.compute_vertex_normals()
            palm_center = np.array([thick / 2, width / 2, thick / 2])
            T_palm = np.eye(4)
            T_palm[:3, :3] = rot
            T_palm[:3, 3] = pos - rot @ palm_center
            palm.transform(T_palm)
            meshes.append(palm)
            return meshes

        # --- Score-to-color ---
        def score_to_color(score, s_min, s_max):
            if s_max == s_min:
                return np.array([0.0, 1.0, 0.0])
            t = (score - s_min) / (s_max - s_min)
            return np.array([1.0 - t, t, 0.0])

        # --- All grasp arrows ---
        purple = [0.6, 0.0, 1.0]
        if grasps is not None and len(grasps) > 0:
            s_min, s_max = scores.min(), scores.max()
            for i, g in enumerate(grasps):
                pos = g[:3, 3]
                z_axis = g[:3, :3][:, 2]
                is_best = (i == best_idx)
                color = purple if is_best else score_to_color(scores[i], s_min, s_max).tolist()

                arrow = make_arrow(
                    pos, z_axis,
                    length=0.08 if is_best else 0.05,
                    color=color,
                    cyl_r=0.004 if is_best else 0.002,
                    cone_r=0.008 if is_best else 0.005,
                )
                geometries.append(arrow)

                sp = o3d.geometry.TriangleMesh.create_sphere(
                    radius=0.01 if is_best else 0.005)
                sp.paint_uniform_color(color)
                sp.compute_vertex_normals()
                sp.translate(pos)
                geometries.append(sp)

        # --- Best grasp: RGB axes + purple sphere + fingers ---
        if best_grasp is not None:
            geometries.extend(make_pose_axes(
                best_grasp, axis_len=0.08, cyl_r=0.004, cone_r=0.008,
                label_color=purple, label_r=0.015,
            ))
            geometries.extend(make_fingers(best_grasp, color=[0.5, 0.2, 0.8]))

        # --- Current EEF: cyan sphere + RGB axes + fingers ---
        try:
            state = self.env.get_robot_state()
            eef_pose = np.array(state["TCPPose"])
            eef_pos_m = eef_pose[:3] / 1000.0
            eef_rot = spt.Rotation.from_euler("xyz", eef_pose[3:], degrees=True)
            eef_transform = np.eye(4)
            eef_transform[:3, :3] = eef_rot.as_matrix()
            eef_transform[:3, 3] = eef_pos_m

            geometries.extend(make_pose_axes(
                eef_transform, axis_len=0.08, cyl_r=0.004, cone_r=0.008,
                label_color=[0.0, 1.0, 1.0], label_r=0.015,
            ))
            geometries.extend(make_fingers(eef_transform, color=[0.0, 0.8, 0.8]))
        except Exception as e:
            logger.warning(f"Could not get EEF pose for visualization: {e}")

        # --- Show (blocks until window closed) ---
        logger.info("Close the visualization window to proceed with robot execution.")
        logger.info("Legend: Purple=best grasp, Cyan=current EEF, "
                     "Green->Red=all grasps, Red-tinted=segmented object")
        o3d.visualization.draw_geometries(
            geometries,
            window_name=f"Grasp Preview: '{prompt}' — Close to execute",
            width=1280, height=720,
        )
        logger.info("Visualization closed, continuing with execution...")

    def _visualize_after_execution(
        self, full_points, full_colors,
        seg_points, seg_colors,
        grasps, scores, best_idx, best_grasp, prompt,
    ):
        """
        Show an "after" visualization using the same scene data from before,
        but with the CURRENT EEF position (after the robot moved).

        This lets you see whether the robot actually reached the target grasp.
        Shows the target grasp (purple) and the actual EEF (cyan) side by side.
        """
        import open3d as o3d
        import scipy.spatial.transform as spt

        logger.info("Opening AFTER visualization (close window to continue)...")

        geometries = []

        # --- Full scene (same as before) ---
        if full_points is not None and len(full_points) > 0:
            scene_pcd = o3d.geometry.PointCloud()
            scene_pcd.points = o3d.utility.Vector3dVector(full_points)
            scene_pcd.colors = o3d.utility.Vector3dVector(full_colors)
            geometries.append(scene_pcd)

        # --- Segmented object (red-tinted, same as before) ---
        if seg_points is not None and len(seg_points) > 0:
            seg_pcd = o3d.geometry.PointCloud()
            seg_pcd.points = o3d.utility.Vector3dVector(seg_points)
            red = np.array([1.0, 0.0, 0.0])
            tinted = seg_colors * 0.6 + red * 0.4
            seg_pcd.colors = o3d.utility.Vector3dVector(np.clip(tinted, 0, 1))
            geometries.append(seg_pcd)

        # --- Robot base frame ---
        robot_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15)
        geometries.append(robot_frame)

        # Reuse the same helpers from _visualize_before_execution
        def make_arrow(origin, direction, length=0.06, color=[1, 0, 0],
                       cyl_r=0.003, cone_r=0.006, cone_frac=0.3):
            cyl_h = length * (1 - cone_frac)
            cone_h = length * cone_frac
            arrow = o3d.geometry.TriangleMesh.create_arrow(
                cylinder_radius=cyl_r, cone_radius=cone_r,
                cylinder_height=cyl_h, cone_height=cone_h,
                resolution=8, cylinder_split=1, cone_split=1,
            )
            arrow.paint_uniform_color(color)
            arrow.compute_vertex_normals()
            direction = direction / np.linalg.norm(direction)
            z = np.array([0.0, 0.0, 1.0])
            v = np.cross(z, direction)
            s = np.linalg.norm(v)
            c = np.dot(z, direction)
            if s < 1e-8:
                R = np.eye(3) if c > 0 else np.diag([-1.0, -1.0, 1.0])
            else:
                vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
                R = np.eye(3) + vx + vx @ vx * (1 - c) / (s * s)
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = origin
            arrow.transform(T)
            return arrow

        def make_pose_axes(transform_4x4, axis_len=0.08, cyl_r=0.004,
                          cone_r=0.008, label_color=None, label_r=0.012):
            axis_colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
            origin = transform_4x4[:3, 3]
            rot = transform_4x4[:3, :3]
            meshes = []
            for i, col in enumerate(axis_colors):
                meshes.append(make_arrow(origin, rot[:, i], length=axis_len,
                                         color=col, cyl_r=cyl_r, cone_r=cone_r))
            if label_color is not None:
                sp = o3d.geometry.TriangleMesh.create_sphere(radius=label_r)
                sp.paint_uniform_color(label_color)
                sp.compute_vertex_normals()
                sp.translate(origin)
                meshes.append(sp)
            return meshes

        def make_fingers(grasp_4x4, color, width=0.08, depth=0.05, thick=0.012):
            pos = grasp_4x4[:3, 3]
            rot = grasp_4x4[:3, :3]
            meshes = []
            for sign in [1, -1]:
                box = o3d.geometry.TriangleMesh.create_box(width=thick, height=thick, depth=depth)
                box.paint_uniform_color(color)
                box.compute_vertex_normals()
                box_center = np.array([thick / 2, thick / 2, depth / 2])
                T = np.eye(4)
                T[:3, :3] = rot
                T[:3, 3] = pos + rot[:, 1] * sign * (width / 2) - rot @ box_center
                box.transform(T)
                meshes.append(box)
            palm = o3d.geometry.TriangleMesh.create_box(width=thick, height=width, depth=thick)
            palm.paint_uniform_color(color)
            palm.compute_vertex_normals()
            palm_center = np.array([thick / 2, width / 2, thick / 2])
            T_palm = np.eye(4)
            T_palm[:3, :3] = rot
            T_palm[:3, 3] = pos - rot @ palm_center
            palm.transform(T_palm)
            meshes.append(palm)
            return meshes

        purple = [0.6, 0.0, 1.0]

        # --- Target grasp (purple) — where we wanted the robot to go ---
        if best_grasp is not None:
            geometries.extend(make_pose_axes(
                best_grasp, axis_len=0.08, cyl_r=0.004, cone_r=0.008,
                label_color=purple, label_r=0.015,
            ))
            geometries.extend(make_fingers(best_grasp, color=[0.5, 0.2, 0.8]))

            # Target approach arrow
            target_z = best_grasp[:3, :3][:, 2]
            geometries.append(make_arrow(
                best_grasp[:3, 3], target_z, length=0.08,
                color=purple, cyl_r=0.004, cone_r=0.008,
            ))

        # --- Actual EEF (cyan) — where the robot actually is now ---
        try:
            state = self.env.get_robot_state()
            eef_pose = np.array(state["TCPPose"])
            eef_pos_m = eef_pose[:3] / 1000.0
            eef_rot = spt.Rotation.from_euler("xyz", eef_pose[3:], degrees=True)
            eef_transform = np.eye(4)
            eef_transform[:3, :3] = eef_rot.as_matrix()
            eef_transform[:3, 3] = eef_pos_m

            geometries.extend(make_pose_axes(
                eef_transform, axis_len=0.08, cyl_r=0.004, cone_r=0.008,
                label_color=[0.0, 1.0, 1.0], label_r=0.015,
            ))
            geometries.extend(make_fingers(eef_transform, color=[0.0, 0.8, 0.8]))

            # Log the error between target and actual
            if best_grasp is not None:
                pos_error = np.linalg.norm(eef_pos_m - best_grasp[:3, 3]) * 1000  # mm
                target_rot = spt.Rotation.from_matrix(best_grasp[:3, :3])
                rot_error = (eef_rot * target_rot.inv()).magnitude() * 180 / np.pi  # deg
                logger.info(f"Execution error:")
                logger.info(f"  Position error: {pos_error:.1f} mm")
                logger.info(f"  Orientation error: {rot_error:.1f} deg")
        except Exception as e:
            logger.warning(f"Could not get EEF pose for after-visualization: {e}")

        # --- Show ---
        logger.info("AFTER visualization: Purple=target grasp, Cyan=actual EEF position")
        logger.info("Close the window to continue.")
        o3d.visualization.draw_geometries(
            geometries,
            window_name=f"After Execution: '{prompt}' — Purple=target, Cyan=actual",
            width=1280, height=720,
        )
        logger.info("After-visualization closed.")

    def _filter_and_select_best_grasp(self, grasps, grasp_scores):
        """
        Filter grasps and select the best one (closest to current EEF).

        Pipeline:
        1. Filter: gripper clearance buffer vs table (z=0)
        2. Filter: confidence threshold (>= 80%)
        3. Normalize: gripper Z-symmetry (pick 180-flip closer to current EEF)
        4. Select: closest to current EEF position (least movement)

        Args:
            grasps: Nx4x4 numpy array of grasp poses in robot frame (meters)
            grasp_scores: N numpy array of GraspGen confidence scores [0-1]

        Returns:
            tuple: (best_idx, best_grasp, best_score, grasps, scores)
                   Returns (None, None, None, None, None) if no grasps survive filtering
        """
        import scipy.spatial.transform as st

        n_initial = len(grasps)
        logger.info(f"  {n_initial} grasps to select from (scores: {grasp_scores.min():.3f} - {grasp_scores.max():.3f})")

        # --- Get current EEF pose ---
        eef_transform = None
        try:
            state = self.env.get_robot_state()
            eef_pose = np.array(state["TCPPose"], dtype=np.float32)
            eef_pos_m = eef_pose[:3] / 1000.0  # mm -> m
            eef_rot = st.Rotation.from_euler("xyz", eef_pose[3:], degrees=True)
            eef_transform = np.eye(4)
            eef_transform[:3, :3] = eef_rot.as_matrix()
            eef_transform[:3, 3] = eef_pos_m
            logger.info(f"  Current EEF: [{eef_pos_m[0]:.3f}, {eef_pos_m[1]:.3f}, {eef_pos_m[2]:.3f}] m")
        except Exception as e:
            logger.warning(f"  Could not get EEF pose: {e}")

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

    def get_object_center(self, prompt, top_percentile=10, use_cache=True):
        """
        Get the grasp pose for an object.
        This calls detect_object_location() which does the full pipeline:
        segmentation, grasp generation, and visualization.

        Args:
            prompt: Text description of object to find
            top_percentile: Unused, kept for compatibility
            use_cache: If True, use cached detection if available (default: True)

        Returns:
            np.array: [x, y, z, roll, pitch, yaw] FULL 6DOF grasp pose in MILLIMETERS and DEGREES
                      Or None if not found
        """
        # Check cache first
        if use_cache and prompt in self._detection_cache:
            logger.info(f"Using cached grasp pose for '{prompt}'")
            return self._detection_cache[prompt]['full_pose']

        # Detect object and get best grasp pose (this does segmentation + grasp generation + visualization)
        grasp_pose = self.detect_object_location(prompt)

        if grasp_pose is None:
            logger.warning(f"No grasp found for object: {prompt}")
            return None

        # grasp_pose is already [x, y, z, roll, pitch, yaw] in mm and degrees
        position_mm = grasp_pose[:3]
        orientation_deg = grasp_pose[3:]

        logger.info(f"Object '{prompt}' grasp pose:")
        logger.info(f"  Position (mm): [{position_mm[0]:.1f}, {position_mm[1]:.1f}, {position_mm[2]:.1f}]")
        logger.info(f"  Orientation (deg): [{orientation_deg[0]:.1f}, {orientation_deg[1]:.1f}, {orientation_deg[2]:.1f}]")

        # Cache the result
        self._detection_cache[prompt] = {
            'full_pose': grasp_pose
        }

        return grasp_pose

    def _visualize_grasp_in_robot_frame(self, points, colors, grasp_pose_4x4, prompt):
        """
        Visualize the best grasp pose in robot frame with Open3D.
        Shows:
        - Object point cloud
        - Robot base coordinate frame (large)
        - Current robot pose coordinate frame (medium - shows where robot IS NOW)
        - Grasp pose coordinate frame (LARGER - clearly shows rotation and target pose)
        - Approach arrow showing gripper direction

        This lets us verify the grasp is in the correct frame, position, AND orientation.
        Also shows current robot position for comparison.

        Args:
            points: Nx3 numpy array of object points in robot frame
            colors: Nx3 numpy array of RGB colors
            grasp_pose_4x4: 4x4 transformation matrix of grasp in robot frame
            prompt: Object description
        """
        try:
            import open3d as o3d
        except ImportError:
            logger.warning("Open3D not available. Skipping grasp visualization.")
            return

        logger.info("\n👁️  Opening Open3D visualization of GRASP POSE in ROBOT FRAME...")

        # Extract grasp position and rotation
        grasp_pos = grasp_pose_4x4[:3, 3]
        grasp_rot = grasp_pose_4x4[:3, :3]

        # Extract orientation as euler angles for logging
        import scipy.spatial.transform as st
        rotation = st.Rotation.from_matrix(grasp_rot)
        euler_deg = rotation.as_euler('xyz', degrees=True)

        # Get current robot pose to compare
        current_robot_pose = self.get_robot_pose()
        current_robot_pos = current_robot_pose[:3] / 1000.0  # Convert mm to meters
        current_robot_ori = current_robot_pose[3:]  # Already in degrees

        logger.info(f"📍 Grasp Pose Analysis:")
        logger.info(f"   Grasp position: [{grasp_pos[0]:.3f}, {grasp_pos[1]:.3f}, {grasp_pos[2]:.3f}] meters")
        logger.info(f"   Grasp orientation (RPY): [{euler_deg[0]:.1f}, {euler_deg[1]:.1f}, {euler_deg[2]:.1f}] degrees")
        logger.info(f"   Object center: [{points.mean(axis=0)[0]:.3f}, {points.mean(axis=0)[1]:.3f}, {points.mean(axis=0)[2]:.3f}] meters")
        logger.info(f"\n🤖 Current Robot Pose:")
        logger.info(f"   Robot position: [{current_robot_pos[0]:.3f}, {current_robot_pos[1]:.3f}, {current_robot_pos[2]:.3f}] meters")
        logger.info(f"   Robot orientation (RPY): [{current_robot_ori[0]:.1f}, {current_robot_ori[1]:.1f}, {current_robot_ori[2]:.1f}] degrees")

        # Calculate distance and rotation difference
        distance_to_grasp = np.linalg.norm(grasp_pos - current_robot_pos)
        logger.info(f"\n📏 Movement Required:")
        logger.info(f"   Distance to grasp: {distance_to_grasp:.3f} meters")
        logger.info(f"   Delta position: [{grasp_pos[0]-current_robot_pos[0]:.3f}, {grasp_pos[1]-current_robot_pos[1]:.3f}, {grasp_pos[2]-current_robot_pos[2]:.3f}] meters")
        logger.info(f"   Delta orientation: [{euler_deg[0]-current_robot_ori[0]:.1f}, {euler_deg[1]-current_robot_ori[1]:.1f}, {euler_deg[2]-current_robot_ori[2]:.1f}] degrees")

        # Create point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        # Create coordinate frame at robot origin (LARGE - this is the robot base)
        robot_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)

        # Create coordinate frame at grasp pose (LARGER - clearly shows orientation!)
        grasp_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15)
        grasp_frame.transform(grasp_pose_4x4)

        # Create coordinate frame at CURRENT ROBOT POSE (MEDIUM - shows where robot currently is)
        current_robot_rot = st.Rotation.from_euler('xyz', current_robot_ori, degrees=True)
        current_robot_transform = np.eye(4)
        current_robot_transform[:3, :3] = current_robot_rot.as_matrix()
        current_robot_transform[:3, 3] = current_robot_pos

        current_robot_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.12)
        current_robot_frame.transform(current_robot_transform)

        # Add a cyan sphere at current robot position for visibility
        current_robot_marker = o3d.geometry.TriangleMesh.create_sphere(radius=0.015)
        current_robot_marker.paint_uniform_color([0.0, 1.0, 1.0])  # Cyan
        current_robot_marker.translate(current_robot_pos)

        # Create approach arrow along Z-axis (gripper approach direction)
        # Arrow shows which direction the gripper will approach from
        arrow_length = 0.12
        arrow_start = grasp_pos
        arrow_end = grasp_pos + grasp_rot[:, 2] * arrow_length  # Z-axis direction

        # Create arrow geometry
        arrow = o3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius=0.005,
            cone_radius=0.01,
            cylinder_height=arrow_length * 0.7,
            cone_height=arrow_length * 0.3
        )
        arrow.paint_uniform_color([1.0, 0.0, 0.0])  # Red for approach direction

        # Transform arrow to align with grasp Z-axis
        # Default arrow points along +Z, we need to align it with grasp Z-axis
        arrow_transform = np.eye(4)
        arrow_transform[:3, :3] = grasp_rot
        arrow_transform[:3, 3] = grasp_pos
        arrow.transform(arrow_transform)

        # Create small sphere at grasp position for clarity
        grasp_marker = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
        grasp_marker.paint_uniform_color([1.0, 0.0, 1.0])  # Magenta
        grasp_marker.translate(grasp_pos)

        # Create gripper fingers visualization at GRASP pose (simplified)
        # REAL ROBOT: Fingers open/close along Y-axis (not X)
        finger_width = 0.08
        finger_thickness = 0.01
        finger_length = 0.04

        # Box is created from [0,0,0] to [width, height, depth], so center is at:
        box_local_center = np.array([finger_thickness/2, finger_width/2, finger_length/2])

        # Left finger (offset along +Y since real robot fingers are along Y)
        left_finger = o3d.geometry.TriangleMesh.create_box(
            width=finger_thickness, height=finger_width, depth=finger_length
        )
        left_finger.paint_uniform_color([0.3, 0.3, 0.8])  # Blue-ish
        left_finger_transform = np.eye(4)
        left_finger_transform[:3, :3] = grasp_rot
        # Position: grasp center + offset along Y - rotated box center offset
        left_finger_transform[:3, 3] = grasp_pos + grasp_rot[:, 1] * 0.04 - grasp_rot @ box_local_center
        left_finger.transform(left_finger_transform)

        # Right finger (offset along -Y since real robot fingers are along Y)
        right_finger = o3d.geometry.TriangleMesh.create_box(
            width=finger_thickness, height=finger_width, depth=finger_length
        )
        right_finger.paint_uniform_color([0.3, 0.3, 0.8])  # Blue-ish
        right_finger_transform = np.eye(4)
        right_finger_transform[:3, :3] = grasp_rot
        # Position: grasp center - offset along Y - rotated box center offset
        right_finger_transform[:3, 3] = grasp_pos - grasp_rot[:, 1] * 0.04 - grasp_rot @ box_local_center
        right_finger.transform(right_finger_transform)

        # Create current gripper fingers visualization at CURRENT robot pose
        # REAL ROBOT: Fingers open/close along Y-axis (not X)
        current_left_finger = o3d.geometry.TriangleMesh.create_box(
            width=finger_thickness, height=finger_width, depth=finger_length
        )
        current_left_finger.paint_uniform_color([0.8, 0.5, 0.2])  # Orange-ish
        current_left_finger_transform = np.eye(4)
        current_robot_rot_matrix = current_robot_rot.as_matrix()
        current_left_finger_transform[:3, :3] = current_robot_rot_matrix
        # Position: robot center + offset along Y - rotated box center offset
        current_left_finger_transform[:3, 3] = current_robot_pos + current_robot_rot_matrix[:, 1] * 0.04 - current_robot_rot_matrix @ box_local_center
        current_left_finger.transform(current_left_finger_transform)

        # Right finger (current)
        current_right_finger = o3d.geometry.TriangleMesh.create_box(
            width=finger_thickness, height=finger_width, depth=finger_length
        )
        current_right_finger.paint_uniform_color([0.8, 0.5, 0.2])  # Orange-ish
        current_right_finger_transform = np.eye(4)
        current_right_finger_transform[:3, :3] = current_robot_rot_matrix
        # Position: robot center - offset along Y - rotated box center offset
        current_right_finger_transform[:3, 3] = current_robot_pos - current_robot_rot_matrix[:, 1] * 0.04 - current_robot_rot_matrix @ box_local_center
        current_right_finger.transform(current_right_finger_transform)

        logger.info("\n🎨 Visualization Guide:")
        logger.info("   - WHITE/RGB points: Segmented object")
        logger.info("   - LARGE RGB axes at origin: Robot BASE frame")
        logger.info("   - MEDIUM RGB axes (CYAN marker): Current robot EE pose")
        logger.info("   - LARGER RGB axes (MAGENTA marker): TARGET grasp pose")
        logger.info("   ")
        logger.info("   Coordinate frames (REAL ROBOT):")
        logger.info("     - RED axis (X): Gripper width direction")
        logger.info("     - GREEN axis (Y): Gripper finger closing direction")
        logger.info("     - BLUE axis (Z): Gripper approach direction")
        logger.info("   ")
        logger.info("   - RED ARROW: Gripper approach vector at target (Z-axis)")
        logger.info("   - BLUE BOXES: Gripper fingers at TARGET grasp pose")
        logger.info("   - ORANGE BOXES: Gripper fingers at CURRENT robot pose")
        logger.info("   - CYAN sphere: Current robot EE position")
        logger.info("   - MAGENTA sphere: Target grasp center point")
        logger.info("\n⚠️  CHECK: ")
        logger.info("   1. Are the TARGET gripper (blue) and CURRENT gripper (orange) in different positions?")
        logger.info("   2. Is the target grasp frame ON the object (not at robot base)?")
        logger.info("   3. Does the RED ARROW point in a sensible approach direction?")
        logger.info("   4. Are the BLUE FINGERS (target) aligned to grasp the object?")
        logger.info("   5. Can the robot reach from ORANGE to BLUE position without collision?")
        logger.info("\nClose the window to continue...")

        # Visualize - include current robot pose visualization
        o3d.visualization.draw_geometries(
            [pcd, robot_frame, current_robot_frame, current_robot_marker,
             grasp_frame, grasp_marker, arrow,
             left_finger, right_finger, current_left_finger, current_right_finger],
            window_name=f"GRASP COMPARISON - '{prompt}' | Current:[{current_robot_pos[0]:.3f},{current_robot_pos[1]:.3f},{current_robot_pos[2]:.3f}] Target:[{grasp_pos[0]:.3f},{grasp_pos[1]:.3f},{grasp_pos[2]:.3f}]",
            width=1280,
            height=720,
        )

    def _visualize_detection(self, points, colors, center, prompt):
        """
        Visualize point cloud with center marker.
        Same visualization as test_lmp_detection.py for debugging.

        Args:
            points: Nx3 numpy array of 3D points
            colors: Nx3 numpy array of RGB colors
            center: [x, y, z] center position
            prompt: Object description for window title
        """
        try:
            import open3d as o3d
        except ImportError:
            logger.warning("Open3D not available. Skipping visualization.")
            return

        logger.info(f"\n👁️  Opening visualization for '{prompt}'...")
        logger.info("📍 Object Center Calculation:")
        logger.info(f"   Total points: {len(points)}")
        logger.info(f"   Center position: [{center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}]")

        # Create point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        # Create coordinate frame at robot origin
        robot_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

        # Create red sphere at center position
        center_marker = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
        center_marker.paint_uniform_color([1.0, 0.0, 0.0])  # Red
        center_marker.translate(center)

        # Create small coordinate frame at center
        center_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
        center_frame.translate(center)

        logger.info("🎨 Visualization Guide:")
        logger.info("   - WHITE/RGB points: Segmented object")
        logger.info("   - RED sphere: Calculated grasp center")
        logger.info("   - Large RGB axes: Robot origin")
        logger.info("   - Small RGB axes: Object center")
        logger.info("\nClose the window to continue...")

        # Visualize
        o3d.visualization.draw_geometries(
            [pcd, robot_frame, center_marker, center_frame],
            window_name=f"LMP Detection - {prompt} - Center at [{center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}]",
            width=1280,
            height=720,
        )

    def _visualize_grasps(self, points, colors, grasps, combined_scores, prompt, best_idx=None):
        """
        Visualize top K generated grasps with meshcat.
        Only shows the highest scoring grasps for clarity.
        Grasps are ranked by combined score (quality + height + distance).

        Args:
            points: Nx3 numpy array of 3D points
            colors: Nx3 numpy array of RGB colors
            grasps: Kx4x4 numpy array of grasp poses
            combined_scores: K numpy array of combined scores (quality + height + distance)
            prompt: Object description
            best_idx: Optional index of the selected best grasp to highlight
        """
        from grasp_gen.utils.meshcat_utils import (
            create_visualizer,
            get_color_from_score,
            visualize_grasp,
            visualize_pointcloud,
        )

        # Create visualizer
        vis = create_visualizer()

        # Center point cloud (same as gg_test.py)
        T_subtract_pc_mean = tra.translation_matrix(-points.mean(axis=0))
        pc_centered = tra.transform_points(points, T_subtract_pc_mean)

        # Normalize colors to [0, 1] range for visualization
        pc_color = colors / 255.0 if colors.max() > 1.0 else colors

        # Visualize point cloud
        visualize_pointcloud(vis, "pc", pc_centered, pc_color, size=0.0025)

        # Select top K grasps by COMBINED score (not just quality)
        top_k = min(self.visualize_top_k, len(grasps))
        top_k_indices = np.argsort(combined_scores)[-top_k:][::-1]  # Descending order

        top_grasps = grasps[top_k_indices]
        top_scores = combined_scores[top_k_indices]

        # Center grasps to match point cloud
        grasps_centered = np.array([T_subtract_pc_mean @ g for g in top_grasps])

        # Get colors for grasps based on scores
        scores_color = get_color_from_score(top_scores, use_255_scale=True)

        logger.info(f"Visualizing top {top_k} grasps (out of {len(grasps)} total) in meshcat...")
        logger.info(f"Top combined score: {top_scores[0]:.3f}, Lowest shown: {top_scores[-1]:.3f}")

        # Check if the selected best grasp is in the top K to highlight it
        best_idx_in_topk = None
        if best_idx is not None and best_idx in top_k_indices:
            best_idx_in_topk = list(top_k_indices).index(best_idx)
            logger.info(f"✨ Highlighting selected grasp (rank #{best_idx_in_topk + 1} in visualization)")

        # Visualize top K grasps
        for j, (grasp, score) in enumerate(zip(grasps_centered, top_scores)):
            # Highlight the selected grasp with gold color and thicker lines
            if j == best_idx_in_topk:
                visualize_grasp(
                    vis,
                    f"grasps/{j:03d}/grasp",
                    grasp,
                    color=[255, 215, 0],  # Gold color for selected grasp
                    gripper_name=self.gripper_name,
                    linewidth=1.2,  # Thicker line
                )
            else:
                visualize_grasp(
                    vis,
                    f"grasps/{j:03d}/grasp",
                    grasp,
                    color=scores_color[j],
                    gripper_name=self.gripper_name,
                    linewidth=0.6,
                )

        logger.info("\n✅ Grasp visualization ready in meshcat!")
        logger.info(f"   Showing top {top_k} grasps ranked by combined score")
        logger.info(f"   Combined score = 60% quality + 25% height + 15% proximity")
        if best_idx_in_topk is not None:
            logger.info(f"   🌟 GOLD grasp is the selected one (rank #{best_idx_in_topk + 1})")
        logger.info("   Press Enter to continue...")
        input()

    def clear_detection_cache(self):
        """Clear the detection cache. Call this when objects move or scene changes."""
        self._detection_cache.clear()
        logger.info("Detection cache cleared")

    def get_robot_pos(self):
        """Return robot end-effector xyz position in robot base frame."""
        state = self.env.get_robot_state()
        current_pose = np.array(state["TCPPose"], dtype=np.float32)
        return current_pose[:3]

    def get_robot_pose(self):
        """Return full robot pose [x, y, z, roll, pitch, yaw]."""
        state = self.env.get_robot_state()
        return np.array(state["TCPPose"], dtype=np.float32)

    def get_robot_xy(self):
        """Return robot end-effector xy position in robot base frame."""
        return self.get_robot_pos()[:2]

    def goto_pos(self, position_xyz_or_pose, duration=3.0, stage_val=0):
        """
        Move the robot end-effector to the desired position (and optionally orientation).

        Args:
            position_xyz_or_pose: Target [x, y, z] position OR [x, y, z, roll, pitch, yaw] pose
            duration: Time to complete movement
            stage_val: Stage value for action
        """
        current_pose = self.get_robot_pose()

        # Check if full 6DOF pose was provided
        if len(position_xyz_or_pose) == 6:
            # Full pose with orientation
            target_position = np.array(position_xyz_or_pose[:3], dtype=np.float32)
            target_orientation = np.array(position_xyz_or_pose[3:], dtype=np.float32).tolist()
            logger.info(f"goto_pos() called with full 6DOF pose:")
            logger.info(f"  Current pose: {current_pose.tolist()}")
            logger.info(f"  Target pose:  {position_xyz_or_pose}")
            logger.info(f"  Current orientation (deg): [{current_pose[3]:.1f}, {current_pose[4]:.1f}, {current_pose[5]:.1f}]")
            logger.info(f"  Target orientation (deg):  [{target_orientation[0]:.1f}, {target_orientation[1]:.1f}, {target_orientation[2]:.1f}]")
            logger.info(f"  ✅ Will execute rotation to target orientation")
        else:
            # Just position, keep current orientation
            target_position = np.array(position_xyz_or_pose, dtype=np.float32)
            target_orientation = current_pose[3:].tolist()
            logger.info(f"goto_pos() called with position only: {position_xyz_or_pose}")
            logger.info(f"  Keeping current orientation: [{target_orientation[0]:.1f}, {target_orientation[1]:.1f}, {target_orientation[2]:.1f}]")

        result = self._move_to_pose(
            target_position=target_position.tolist(),
            target_orientation=target_orientation,
            duration=duration,
            stage_val=stage_val,
        )

        # Show "after" visualization only when we've actually reached the grasp pose
        # (not the approach pose which is 100mm above). Compare target position to
        # the stored best_grasp position — only trigger if they match within 5mm.
        if self._last_viz_data is not None:
            d = self._last_viz_data
            grasp_pos_mm = d["best_grasp"][:3, 3] * 1000.0  # meters -> mm
            target_pos_mm = np.array(target_position[:3], dtype=np.float64)
            if np.linalg.norm(grasp_pos_mm - target_pos_mm) < 5.0:
                self._visualize_after_execution(
                    d["full_points"], d["full_colors"],
                    d["seg_points"], d["seg_colors"],
                    d["grasps"], d["scores"],
                    d["best_idx"], d["best_grasp"], d["prompt"],
                )
                self._last_viz_data = None  # Only show once

        return result

    def goto_xy(self, position_xy, duration=2.0, stage_val=0):
        """
        Move robot end-effector to desired xy position while maintaining same z.

        Args:
            position_xy: Target [x, y] position
            duration: Time to complete movement
            stage_val: Stage value for action
        """
        current_pos = self.get_robot_pos()
        target_xyz = np.concatenate([position_xy, [current_pos[2]]])
        return self.goto_pos(target_xyz, duration, stage_val)

    def goto_pose(self, position_xyz, orientation_rpy, duration=3.0, stage_val=0):
        """
        Move robot to specific pose (position + orientation).

        Args:
            position_xyz: Target [x, y, z] position
            orientation_rpy: Target [roll, pitch, yaw] orientation in degrees
            duration: Time to complete movement
            stage_val: Stage value for action
        """
        logger.info(f"goto_pose() called:")
        logger.info(f"  Target position: {position_xyz}")
        logger.info(f"  Target orientation: [{orientation_rpy[0]:.1f}, {orientation_rpy[1]:.1f}, {orientation_rpy[2]:.1f}]")
        logger.info(f"  ✅ Will execute rotation to target orientation")

        return self._move_to_pose(
            target_position=position_xyz,
            target_orientation=orientation_rpy,
            duration=duration,
            stage_val=stage_val,
        )

    def move_relative(self, delta_xyz, delta_rpy=None, duration=2.0, stage_val=0):
        """
        Move robot relative to current pose.

        Args:
            delta_xyz: Relative [dx, dy, dz] movement
            delta_rpy: Relative [droll, dpitch, dyaw] rotation (optional)
            duration: Time to complete movement
            stage_val: Stage value for action
        """
        if delta_rpy is None:
            delta_rpy = [0.0, 0.0, 0.0]

        return self._move_relative(
            delta_position=delta_xyz,
            delta_orientation=delta_rpy,
            duration=duration,
            stage_val=stage_val,
        )

    # def move_up(self, distance=0.05, duration=1.0, stage_val=0):
    def move_up(self, distance=2.00, duration=1.0, stage_val=0):
        """Move robot up by specified distance."""
        return self.move_relative(
            [0, 0, distance], duration=duration, stage_val=stage_val
        )

    def move_down(self, distance=2.00, duration=1.0, stage_val=0):
        """Move robot down by specified distance."""
        return self.move_relative(
            [0, 0, -distance], duration=duration, stage_val=stage_val
        )

    def open_gripper(self, stage_val=0):
        """Open the gripper."""
        return self._set_gripper(0.0, stage_val)

    def close_gripper(self, stage_val=0):
        """Close the gripper."""
        return self._set_gripper(1.0, stage_val)

    def set_gripper(self, grasp_value, stage_val=0):
        """Set gripper to specific value."""
        return self._set_gripper(grasp_value, stage_val)

    def pick_place(
        self,
        pick_pos,
        place_pos,
        pick_height=0.15,
        place_height=0.15,
        approach_height=0.25,
        stage_val=0,
    ):
        """
        Execute pick and place operation.

        Args:
            pick_pos: [x, y] or [x, y, z] pick position
            place_pos: [x, y] or [x, y, z] place position
            pick_height: Z height for picking (if pick_pos is 2D)
            place_height: Z height for placing (if place_pos is 2D)
            approach_height: Z height for approach movements
            stage_val: Stage value for actions
        """
        try:
            # Convert to 3D positions if needed
            if len(pick_pos) == 2:
                pick_pos_xyz = np.array([pick_pos[0], pick_pos[1], pick_height])
            else:
                pick_pos_xyz = np.array(pick_pos)

            if len(place_pos) == 2:
                place_pos_xyz = np.array([place_pos[0], place_pos[1], place_height])
            else:
                place_pos_xyz = np.array(place_pos)

            approach_pick = np.array(
                [pick_pos_xyz[0], pick_pos_xyz[1], approach_height]
            )
            approach_place = np.array(
                [place_pos_xyz[0], place_pos_xyz[1], approach_height]
            )

            logger.info(f"Pick and place: {pick_pos_xyz} -> {place_pos_xyz}")

            # Move to approach pick position
            self.goto_pos(approach_pick, duration=3.0, stage_val=stage_val)

            # Open gripper
            self.open_gripper(stage_val)
            time.sleep(0.5)

            # Move down to pick
            self.goto_pos(pick_pos_xyz, duration=2.0, stage_val=stage_val)

            # Close gripper
            self.close_gripper(stage_val)
            time.sleep(1.0)

            # Move up to approach height
            self.goto_pos(approach_pick, duration=2.0, stage_val=stage_val)

            # Move to approach place position
            self.goto_pos(approach_place, duration=3.0, stage_val=stage_val)

            # Move down to place
            self.goto_pos(place_pos_xyz, duration=2.0, stage_val=stage_val)

            # Open gripper
            self.open_gripper(stage_val)
            time.sleep(0.5)

            # Move up
            self.goto_pos(approach_place, duration=2.0, stage_val=stage_val)

            logger.info("Pick and place completed successfully")
            return True

        except Exception as e:
            logger.error(f"Pick and place failed: {e}")
            return False

    def follow_traj(self, trajectory, duration_per_point=1.0, stage_val=0):
        """
        Follow a trajectory of positions.

        Args:
            trajectory: List of [x, y] or [x, y, z] positions
            duration_per_point: Time to spend moving to each point
            stage_val: Stage value for actions
        """
        for i, pos in enumerate(trajectory):
            logger.info(f"Following trajectory point {i+1}/{len(trajectory)}: {pos}")
            if len(pos) == 2:
                self.goto_xy(pos, duration=duration_per_point, stage_val=stage_val)
            else:
                self.goto_pos(pos, duration=duration_per_point, stage_val=stage_val)

    def hold_position(self, duration, stage_val=0):
        """
        Hold current position for specified duration.

        Args:
            duration: Duration to hold position in seconds
            stage_val: Stage value for actions
        """
        steps = int(duration * self._frequency)
        t_start = time.monotonic()

        # Get current pose
        current_pose = self.get_robot_pose()

        for iter_idx in range(steps):
            # Same timing logic as teleop script
            t_cycle_end = t_start + (iter_idx + 1) * self._dt
            t_command_target = t_cycle_end + self._dt

            # Pump obs
            obs = self.env.get_obs()

            # Send current pose - same as teleop when no significant movement
            action = np.concatenate([current_pose, [self._current_grasp]])
            exec_timestamp = t_command_target - time.monotonic() + time.time()

            self.env.exec_actions(
                actions=[action],
                timestamps=[exec_timestamp],
                stages=[stage_val],
            )

            self._precise_wait(t_cycle_end)

    def wait(self, duration):
        """Simple wait function."""
        time.sleep(duration)

    def is_gripper_open(self):
        """Check if gripper is open."""
        return self._current_grasp < 0.5

    def is_gripper_closed(self):
        """Check if gripper is closed."""
        return self._current_grasp >= 0.5

    def get_gripper_state(self):
        """Get current gripper state."""
        return self._current_grasp

    # ========== LMP-Required Functions ==========

    def get_obj_names(self):
        """Return list of known object names."""
        return self.known_objects.copy()

    def is_obj_visible(self, obj_name):
        """Check if object is visible/known."""
        return obj_name in self.known_objects

    def get_obj_pos(self, obj_name, use_segmentation=True):
        """
        Get object position using segmentation or predefined positions.
        Returns full 6DOF pose [x, y, z, roll, pitch, yaw] for grasping.

        Args:
            obj_name: Name of object or position reference
            use_segmentation: If True, use AI segmentation to find object (default: True)

        Returns:
            np.array: [x, y, z, roll, pitch, yaw] - Full 6DOF grasp pose
                      For predefined positions, orientation is [179, 0, 0] (top-down)
        """
        obj_name_clean = obj_name.replace("the", "").replace("_", " ").strip()

        # Check if it's a predefined position first
        if obj_name_clean in self.corner_positions:
            pos = list(self.corner_positions[obj_name_clean])
            # Add default top-down orientation for predefined positions
            return pos + [179.0, 0.0, 0.0]  # Top-down grasp
        elif obj_name_clean in self.side_positions:
            pos = list(self.side_positions[obj_name_clean])
            # Add default top-down orientation for predefined positions
            return pos + [179.0, 0.0, 0.0]  # Top-down grasp

        # Use segmentation to find the object
        if use_segmentation:
            logger.info(f"Using segmentation to locate: {obj_name_clean}")
            grasp_pose = self.get_object_center(obj_name_clean)
            if grasp_pose is not None:
                return grasp_pose.tolist()
            else:
                logger.error(f"❌ Object detection failed for '{obj_name_clean}' - no grasp pose found")
                raise RuntimeError(f"Object '{obj_name_clean}' could not be detected. Segmentation returned no valid grasp pose.")

        # If segmentation is disabled, raise an error
        logger.error(f"❌ Cannot locate '{obj_name_clean}' - not a predefined position and segmentation is disabled")
        raise RuntimeError(f"Object '{obj_name_clean}' not found. It is not a predefined position (corners/sides) and segmentation is disabled.")

    def get_bbox(self, obj_name):
        """
        Get object bounding box.

        Args:
            obj_name: Name of object

        Returns:
            tuple: (min_x, min_y, max_x, max_y) bounding box
        """
        pos = self.get_obj_pos(obj_name)
        # Return simple bounding box around position (adjust size as needed)
        size = 0.02  # 2cm box
        return (pos[0] - size, pos[1] - size, pos[0] + size, pos[1] + size)

    def get_color(self, obj_name):
        """
        Extract color from object name.

        Args:
            obj_name: Name of object

        Returns:
            tuple: (r, g, b, a) color values
        """
        for color_name, rgb in self.colors.items():
            if color_name in obj_name.lower():
                return rgb
        return (0.5, 0.5, 0.5, 1.0)  # Default gray

    def denormalize_xy(self, pos_normalized):
        """
        Convert normalized coordinates [0,1] to workspace coordinates.

        Args:
            pos_normalized: [x, y] in range [0, 1]

        Returns:
            np.array: [x, y] in workspace coordinates
        """
        x_range = self.workspace_bounds["x_max"] - self.workspace_bounds["x_min"]
        y_range = self.workspace_bounds["y_max"] - self.workspace_bounds["y_min"]

        x = pos_normalized[0] * x_range + self.workspace_bounds["x_min"]
        y = pos_normalized[1] * y_range + self.workspace_bounds["y_min"]

        return np.array([x, y])

    def get_corner_name(self, pos):
        """
        Get the name of the closest corner to a position.

        Args:
            pos: [x, y] or [x, y, z] position

        Returns:
            str: Name of closest corner
        """
        pos_2d = np.array(pos[:2])
        corner_positions_2d = np.array(
            [[p[0], p[1]] for p in self.corner_positions.values()]
        )
        distances = np.linalg.norm(corner_positions_2d - pos_2d, axis=1)
        closest_idx = np.argmin(distances)
        corner_names = list(self.corner_positions.keys())
        return corner_names[closest_idx]

    def get_side_name(self, pos):
        """
        Get the name of the closest side to a position.

        Args:
            pos: [x, y] or [x, y, z] position

        Returns:
            str: Name of closest side
        """
        pos_2d = np.array(pos[:2])
        side_positions_2d = np.array(
            [[p[0], p[1]] for p in self.side_positions.values()]
        )
        distances = np.linalg.norm(side_positions_2d - pos_2d, axis=1)
        closest_idx = np.argmin(distances)
        side_names = list(self.side_positions.keys())
        return side_names[closest_idx]

    def put_first_on_second(self, obj1, obj2):
        """
        Put first object on second object or position.
        This is the core LMP function that maps to your pick_place.

        Args:
            obj1: Object name or position to pick
            obj2: Object name or position to place on

        Returns:
            bool: Success status
        """
        try:
            # Get pick position
            if isinstance(obj1, str):
                pick_pos = self.get_obj_pos(obj1)[:2]  # Get x,y only
            else:
                pick_pos = np.array(obj1)[:2]

            # Get place position
            if isinstance(obj2, str):
                place_pos = self.get_obj_pos(obj2)[:2]  # Get x,y only
            else:
                place_pos = np.array(obj2)[:2]

            logger.info(f"Executing put_first_on_second: {obj1} -> {obj2}")
            logger.info(f"Pick position: {pick_pos}, Place position: {place_pos}")

            # Execute pick and place
            return self.pick_place(pick_pos, place_pos)

        except Exception as e:
            logger.error(f"Error in put_first_on_second: {e}")
            return False

    # ========== Object Management Functions ==========

    def add_object(self, obj_name, position=None):
        """
        Add an object to the known objects list.

        Args:
            obj_name: Name of object to add
            position: Optional position (for future use with object tracking)
        """
        if obj_name not in self.known_objects:
            self.known_objects.append(obj_name)
            logger.info(f"Added object: {obj_name}")

    def remove_object(self, obj_name):
        """
        Remove an object from the known objects list.

        Args:
            obj_name: Name of object to remove
        """
        if obj_name in self.known_objects:
            self.known_objects.remove(obj_name)
            logger.info(f"Removed object: {obj_name}")

    def update_object_list(self, object_list):
        """
        Update the list of known objects.

        Args:
            object_list: List of object names
        """
        self.known_objects = object_list.copy()
        logger.info(f"Updated object list: {self.known_objects}")

    def clear_objects(self):
        """Clear all known objects."""
        self.known_objects.clear()
        logger.info("Cleared all objects")

    # ========== Utility Functions ==========

    def get_workspace_bounds(self):
        """Get workspace boundaries."""
        return self.workspace_bounds.copy()

    def set_workspace_bounds(self, bounds):
        """
        Set workspace boundaries.

        Args:
            bounds: Dictionary with keys 'x_min', 'x_max', 'y_min', 'y_max', 'z_table'
        """
        self.workspace_bounds.update(bounds)
        logger.info(f"Updated workspace bounds: {self.workspace_bounds}")

    def get_corner_positions(self):
        """Get all corner positions."""
        return self.corner_positions.copy()

    def get_side_positions(self):
        """Get all side positions."""
        return self.side_positions.copy()

    # ========== Private Methods (unchanged) ==========

    def _move_to_pose(
        self, target_position, target_orientation, duration=3.0, stage_val=0
    ):
        """Internal move to pose function using teleop script logic."""

        print("MOVING TO", target_position)
        try:
            target_pose = np.array(
                target_position + target_orientation, dtype=np.float64
            )

            # Get current pose from robot
            state = self.env.get_robot_state()
            start_pose = np.array(state["TCPPose"], dtype=np.float64)

            logger.info(f"_move_to_pose():")
            logger.info(f"   Start: {start_pose[:3].tolist()}")
            logger.info(f"   Target: {target_pose[:3].tolist()}")
            logger.info(f"   Start orientation: [{start_pose[3]:.1f}, {start_pose[4]:.1f}, {start_pose[5]:.1f}]")
            logger.info(f"   Target orientation: [{target_pose[3]:.1f}, {target_pose[4]:.1f}, {target_pose[5]:.1f}]")

            # Calculate interpolation steps
            interpolation_steps = int(duration * self._frequency)

            logger.debug(
                f"Moving from {start_pose} to {target_pose} over {duration}s ({interpolation_steps} steps)"
            )

            t_start = time.monotonic()

            for iter_idx in range(interpolation_steps):
                # Same timing logic as teleop script
                t_cycle_end = t_start + (iter_idx + 1) * self._dt
                t_command_target = t_cycle_end + self._dt

                # Pump obs - same as teleop script
                obs = self.env.get_obs()

                # Interpolation with proper rotation handling
                t = (iter_idx + 1) / interpolation_steps

                # Linear interpolation for position
                interpolated_position = start_pose[:3] + t * (target_pose[:3] - start_pose[:3])

                # SLERP for orientation to avoid gimbal lock and angle wrapping
                start_rot = st.Rotation.from_euler('xyz', start_pose[3:], degrees=True)
                target_rot = st.Rotation.from_euler('xyz', target_pose[3:], degrees=True)
                interpolated_rot = st.Slerp([0, 1], st.Rotation.concatenate([start_rot, target_rot]))
                interpolated_orientation = interpolated_rot(t).as_euler('xyz', degrees=True)

                # Combine position and orientation
                interpolated_pose = np.concatenate([interpolated_position, interpolated_orientation])

                # DEBUG: Print first and last commands
                if iter_idx == 0:
                    logger.info(f"\n🔍 DEBUG: First interpolated command (t={t:.3f}):")
                    logger.info(f"   Pose: {interpolated_pose.tolist()}")
                    logger.info(f"   Orientation: [{interpolated_pose[3]:.1f}, {interpolated_pose[4]:.1f}, {interpolated_pose[5]:.1f}]")
                elif iter_idx == interpolation_steps - 1:
                    logger.info(f"\n🔍 DEBUG: Last interpolated command (t={t:.3f}):")
                    logger.info(f"   Pose: {interpolated_pose.tolist()}")
                    logger.info(f"   Orientation: [{interpolated_pose[3]:.1f}, {interpolated_pose[4]:.1f}, {interpolated_pose[5]:.1f}]")
                    logger.info(f"   Should match target: [{target_pose[3]:.1f}, {target_pose[4]:.1f}, {target_pose[5]:.1f}]")

                # Create action with current grasp state - same format as teleop
                action = np.concatenate([interpolated_pose, [self._current_grasp]])

                # Execute with same timing logic as teleop
                exec_timestamp = t_command_target - time.monotonic() + time.time()
                self.env.exec_actions(
                    actions=[action],
                    timestamps=[exec_timestamp],
                    stages=[stage_val],
                )

                # Wait for cycle end - same as teleop
                self._precise_wait(t_cycle_end)

            # Update current pose
            self._current_pose = target_pose.copy()
            return True

        except Exception as e:
            logger.error(f"Error during movement: {e}")
            return False

    def _move_relative(
        self, delta_position, delta_orientation, duration=1.0, stage_val=0
    ):
        """Internal relative movement function using teleop script logic."""
        try:
            # Get current pose
            state = self.env.get_robot_state()
            current_pose = np.array(state["TCPPose"], dtype=np.float32)

            # Apply gains - same as teleop script
            dpos = (
                np.array(delta_position, dtype=np.float32)
                * self._xarm_config.position_gain
            )
            drot = (
                np.array(delta_orientation, dtype=np.float32)
                * self._xarm_config.orientation_gain
            )

            # Same rotation logic as teleop script
            curr_rot = st.Rotation.from_euler("xyz", current_pose[3:], degrees=True)
            delta_rot = st.Rotation.from_euler("xyz", drot, degrees=True)
            final_rot = delta_rot * curr_rot

            # Calculate target pose
            target_position = current_pose[:3] + dpos
            target_orientation = final_rot.as_euler("xyz", degrees=True)

            return self._move_to_pose(
                target_position=target_position.tolist(),
                target_orientation=target_orientation.tolist(),
                duration=duration,
                stage_val=stage_val,
            )

        except Exception as e:
            logger.error(f"Error during relative movement: {e}")
            return False

    def _set_gripper(self, grasp_value, stage_val=0, settle_time=1.5):
        """Internal gripper control function.
        
        Sends the gripper command and holds position while the gripper
        physically closes/opens. The Robotiq gripper takes ~1-2s to fully
        close, so we keep sending hold-position commands with the new grasp
        value for settle_time seconds.

        Args:
            grasp_value: 0.0 = open, 1.0 = closed
            stage_val: Stage value for actions
            settle_time: Time in seconds to hold position while gripper moves (default 1.5s)
        """
        try:
            # Update grasp state
            self._current_grasp = float(grasp_value)

            # Get current pose
            state = self.env.get_robot_state()
            current_pose = np.array(state["TCPPose"], dtype=np.float32)

            action = np.concatenate([current_pose, [self._current_grasp]])

            # Send hold-position + gripper command for the full settle duration.
            # This ensures the gripper has time to physically close/open while
            # the robot stays in place.
            steps = int(settle_time * self._frequency)
            t_start = time.monotonic()

            for iter_idx in range(steps):
                t_cycle_end = t_start + (iter_idx + 1) * self._dt
                t_command_target = t_cycle_end + self._dt

                exec_timestamp = t_command_target - time.monotonic() + time.time()
                self.env.exec_actions(
                    actions=[action],
                    timestamps=[exec_timestamp],
                    stages=[stage_val],
                )

                self._precise_wait(t_cycle_end)

            logger.info(f"Gripper {'closed' if grasp_value >= 0.5 else 'opened'} (value={self._current_grasp}, held {settle_time:.1f}s)")
            return True

        except Exception as e:
            logger.error(f"Error setting gripper: {e}")
            return False

    def _precise_wait(self, target_time):
        """Precise wait function - same as teleop script."""
        try:
            # Try to use the precise_wait from teleop script if available
            from ril_env.precise_sleep import precise_wait

            precise_wait(target_time)
        except ImportError:
            # Fallback to regular sleep
            wait_time = target_time - time.monotonic()
            if wait_time > 0:
                time.sleep(wait_time)


def setup_LMP(config, env, xarm_config):
    """
    Setup LMP system for real robot environment using enhanced LMPWrapper.

    Args:
        config: Configuration dictionary containing LMP configs

    Returns:
        tuple: (lmp_tabletop_ui, LMP_env) - The main LMP interface and environment wrapper
    """

    LMP_env = LMPWrapper(env, xarm_config)

    # Creating APIs that the LMPs can interact with
    fixed_vars = {
        "np": np,
        "time": __import__("time"),
    }

    # Add shapely geometry functions
    fixed_vars.update(
        {name: getattr(shapely.geometry, name) for name in shapely.geometry.__all__}
    )
    fixed_vars.update(
        {name: getattr(shapely.affinity, name) for name in shapely.affinity.__all__}
    )

    # Add LMP environment functions (all now available in LMPWrapper)
    variable_vars = {
        # Core robot functions
        "get_robot_pos": LMP_env.get_robot_pos,
        "get_robot_xy": LMP_env.get_robot_xy,
        "goto_pos": LMP_env.goto_pos,
        "goto_xy": LMP_env.goto_xy,
        "move_relative": LMP_env.move_relative,
        "move_up": LMP_env.move_up,
        "move_down": LMP_env.move_down,
        "pick_place": LMP_env.pick_place,
        "follow_traj": LMP_env.follow_traj,
        "wait": LMP_env.wait,
        # Gripper functions
        "open_gripper": LMP_env.open_gripper,
        "close_gripper": LMP_env.close_gripper,
        "set_gripper": LMP_env.set_gripper,
        "is_gripper_open": LMP_env.is_gripper_open,
        "is_gripper_closed": LMP_env.is_gripper_closed,
        # LMP-required functions (now implemented in LMPWrapper)
        "get_obj_pos": LMP_env.get_obj_pos,
        "get_obj_names": LMP_env.get_obj_names,
        "is_obj_visible": LMP_env.is_obj_visible,
        "put_first_on_second": LMP_env.put_first_on_second,
        "denormalize_xy": LMP_env.denormalize_xy,
        "get_corner_name": LMP_env.get_corner_name,
        "get_side_name": LMP_env.get_side_name,
        "get_bbox": LMP_env.get_bbox,
        "get_color": LMP_env.get_color,
        # Detection/grasp functions
        "get_object_center": LMP_env.get_object_center,
        "clear_detection_cache": LMP_env.clear_detection_cache,
        # Utility functions
        "say": lambda msg: print(f"robot says: {msg}"),
    }

    # Creating the function-generating LMP
    lmp_fgen = LMPFGen(config["lmp_config"]["lmps"]["fgen"], fixed_vars, variable_vars)

    # Creating other low-level LMPs
    variable_vars.update(
        {
            k: LMP(
                k, config["lmp_config"]["lmps"][k], lmp_fgen, fixed_vars, variable_vars
            )
            for k in [
                "parse_obj_name",
                "parse_position",
                "parse_question",
                "transform_shape_pts",
            ]
        }
    )

    # Creating the LMP that deals with high-level language commands

    print(fixed_vars)
    print("========================================")
    print(variable_vars)

    lmp_tabletop_ui = LMP(
        "tabletop_ui",
        config["lmp_config"]["lmps"]["tabletop_ui"],
        lmp_fgen,
        fixed_vars,
        variable_vars,
    )

    return lmp_tabletop_ui, LMP_env
