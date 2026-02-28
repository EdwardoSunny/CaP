#!/usr/bin/env python3
"""
Test the full segmentation + grasp generation pipeline.

Saves PLY files you can visualize with Open3D:
1. full_scene.ply          - Full merged point cloud (original colors)
2. segmented_<obj>.ply     - Segmented object (red-tinted)
3. grasps.ply              - Grasp poses as colored points (green=best, red=worst)
4. combined.ply            - Everything together for one-shot visualization

Usage:
    python test_segmentation_pipeline.py "orange cleaning wipes"
    python test_segmentation_pipeline.py "cup"
    python test_segmentation_pipeline.py "bottle"
"""

import sys
import numpy as np
from pathlib import Path
import torch
import open3d as o3d
import trimesh.transformations as tra
import scipy.spatial.transform as spt

from cap.segment_pc import RobotFrameMerger, load_segmentation_models
from grasp_gen.grasp_server import GraspGenSampler, load_grasp_cfg
from grasp_gen.utils.point_cloud_utils import point_cloud_outlier_removal
from xarm.wrapper import XArmAPI


def tint_red(colors, strength=0.4):
    """Apply a red tint to RGB colors."""
    red = np.array([1.0, 0.0, 0.0])
    tinted = colors * (1 - strength) + red * strength
    return np.clip(tinted, 0, 1)


def score_to_color(score, min_score, max_score):
    """Map a score to a green (best) -> yellow -> red (worst) color."""
    if max_score == min_score:
        return np.array([0.0, 1.0, 0.0])
    t = (score - min_score) / (max_score - min_score)
    r = 1.0 - t
    g = t
    return np.array([r, g, 0.0])


def generate_grasps(seg_points):
    """
    Run GraspGen on segmented points (already in robot frame, meters).

    Returns:
        grasps: Nx4x4 numpy array of grasp poses in robot frame
        scores: N numpy array of confidence scores
        or (None, None) if generation fails
    """
    project_root = Path(__file__).parent

    gripper_config = project_root / "GraspGen" / "GraspGenModels" / "checkpoints" / "graspgen_robotiq_2f_140.yml"
    if not gripper_config.exists():
        print(f"GraspGen config not found at {gripper_config}")
        print("Download models: git clone https://huggingface.co/adithyamurali/GraspGenModels")
        print(f"Expected at: {gripper_config.parent}")
        return None, None, None

    print("\nLoading GraspGen model...")
    grasp_cfg = load_grasp_cfg(str(gripper_config))
    grasp_sampler = GraspGenSampler(grasp_cfg)
    print(f"GraspGen loaded for gripper: {grasp_cfg.data.gripper_name}")

    # Downsample if too many points (GraspGen's outlier removal is O(N^2) memory)
    max_points = 5000
    if len(seg_points) > max_points:
        print(f"Downsampling {len(seg_points)} -> ~{max_points} points (voxel grid)")
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(seg_points)
        # Estimate voxel size to hit target count
        bbox = pcd.get_axis_aligned_bounding_box()
        volume = np.prod(bbox.get_extent())
        voxel_size = (volume / max_points) ** (1.0 / 3.0)
        pcd_down = pcd.voxel_down_sample(voxel_size)
        seg_points = np.asarray(pcd_down.points)
        print(f"After downsampling: {len(seg_points)} points (voxel_size={voxel_size:.4f} m)")

    # Center the point cloud (GraspGen expects centered input)
    pc_mean = seg_points.mean(axis=0)
    pc_centered = seg_points - pc_mean

    # Remove outliers
    pc_filtered, pc_removed = point_cloud_outlier_removal(
        torch.from_numpy(pc_centered)
    )
    pc_filtered = pc_filtered.numpy()
    print(f"Point cloud: {len(pc_filtered)} kept, {len(pc_removed.numpy())} outliers removed")

    # Run inference
    print("Running GraspGen inference...")
    grasps_raw, scores = GraspGenSampler.run_inference(
        pc_filtered,
        grasp_sampler,
        grasp_threshold=0.8,
        num_grasps=200,
        topk_num_grasps=-1,
    )

    if len(grasps_raw) == 0:
        print("No grasps generated.")
        return None, None, None

    grasps_raw = grasps_raw.cpu().numpy()
    scores = scores.cpu().numpy()
    grasps_raw[:, 3, 3] = 1  # ensure homogeneous

    # Un-center: shift grasps back to robot frame
    T_add_mean = tra.translation_matrix(pc_mean)
    grasps_robot = np.array([T_add_mean @ g for g in grasps_raw])

    # Apply gripper frame correction: GraspGen fingers-along-X -> real robot fingers-along-Y
    # Rotate -90 deg around Z
    T_correction = tra.rotation_matrix(-np.pi / 2, [0, 0, 1])
    grasps_robot = np.array([g @ T_correction for g in grasps_robot])

    # Assign colors BEFORE filtering so the full score range is reflected
    s_min, s_max = scores.min(), scores.max()
    grasp_colors = np.array([
        score_to_color(s, s_min, s_max) for s in scores
    ])
    print(f"Score range for coloring: {s_min:.3f} - {s_max:.3f}")

    # Filter out grasps where the gripper clearance buffer would go below z=0 (table).
    # The EEF position is the gripper center. The physical clearance needed is:
    #   X (red, thickness):    ±2.5 cm
    #   Y (green, finger span): ±8 cm
    # Z (approach) is ignored — the EEF position is already at the bottom of the gripper.
    gripper_half = np.array([0.025, 0.08, 0.0])  # half-extents in local frame (m)
    # 4 corners (only X and Y matter, Z=0 at grasp center)
    signs = np.array([[sx, sy, 0.0]
                      for sx in [-1, 1] for sy in [-1, 1]])
    local_corners = signs * gripper_half  # (4, 3)

    table_safe = np.ones(len(grasps_robot), dtype=bool)
    for i, g in enumerate(grasps_robot):
        pos = g[:3, 3]
        rot = g[:3, :3]
        # Transform corners to world frame
        world_corners = (rot @ local_corners.T).T + pos  # (4, 3)
        if world_corners[:, 2].min() < 0:
            table_safe[i] = False

    n_before = len(grasps_robot)
    grasps_robot = grasps_robot[table_safe]
    scores = scores[table_safe]
    grasp_colors = grasp_colors[table_safe]
    n_filtered = n_before - len(grasps_robot)
    if n_filtered > 0:
        print(f"Filtered out {n_filtered} grasps where gripper buffer intersects table (z=0)")

    # Filter out grasps with confidence below 80%
    high_conf = scores >= 0.80
    n_before2 = len(grasps_robot)
    grasps_robot = grasps_robot[high_conf]
    scores = scores[high_conf]
    grasp_colors = grasp_colors[high_conf]
    n_filtered2 = n_before2 - len(grasps_robot)
    if n_filtered2 > 0:
        print(f"Filtered out {n_filtered2} grasps below 80% confidence")

    if len(grasps_robot) == 0:
        print("No valid grasps remaining after filtering.")
        return None, None, None

    print(f"Generated {len(grasps_robot)} valid grasps (scores: {scores.min():.3f} - {scores.max():.3f})")

    return grasps_robot, scores, grasp_colors


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_segmentation_pipeline.py '<object name>'")
        print()
        print("Examples:")
        print("  python test_segmentation_pipeline.py 'orange cleaning wipes'")
        print("  python test_segmentation_pipeline.py 'cup'")
        print("  python test_segmentation_pipeline.py 'bottle'")
        sys.exit(1)

    text_prompt = sys.argv[1]

    # Configuration
    camera_serials = ["327122079374", "317222072157"]
    calib_file = Path("transforms/transforms.npy")

    print("=" * 60)
    print("Segmentation + GraspGen Pipeline Test")
    print("=" * 60)
    print(f"Object to find: '{text_prompt}'")
    print(f"Cameras: {camera_serials}")

    # Load segmentation models
    print("\nLoading SAM2 + CLIP models...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    sam_gen, clip_model, clip_proc = load_segmentation_models(
        sam2_checkpoint=Path("ckpt/sam2.1_hiera_large.pt"),
        sam2_config="sam2.1/sam2.1_hiera_l",
        clip_model_name="laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
        device=device,
    )
    print("Segmentation models loaded.")

    # Create merger
    print("\nInitializing cameras...")
    merger = RobotFrameMerger(
        camera_serials=camera_serials,
        calib_file=calib_file,
        max_depth=2.0,
        min_depth=0.1,
        sam_generator=sam_gen,
        clip_model=clip_model,
        clip_processor=clip_proc,
        device=device,
    )
    print("Cameras ready.")

    try:
        # --- Capture 1: Full scene (no segmentation) ---
        print("\n--- Capturing full scene (no segmentation) ---")
        full_points, full_colors = merger.capture_merged_pointcloud(text_prompt=None)

        if full_points is None or len(full_points) == 0:
            print("Failed to capture full scene point cloud.")
            return

        print(f"Full scene: {len(full_points)} points")
        print(f"  X: [{full_points[:,0].min():.3f}, {full_points[:,0].max():.3f}]")
        print(f"  Y: [{full_points[:,1].min():.3f}, {full_points[:,1].max():.3f}]")
        print(f"  Z: [{full_points[:,2].min():.3f}, {full_points[:,2].max():.3f}]")

        # --- Capture 2: Segmented object ---
        print(f"\n--- Segmenting '{text_prompt}' ---")
        seg_points, seg_colors = merger.capture_merged_pointcloud(text_prompt=text_prompt)

        if seg_points is None or len(seg_points) == 0:
            print(f"No points found for '{text_prompt}'. Saving full scene only.")
            seg_points, seg_colors = None, None
        else:
            print(f"Segmented object: {len(seg_points)} points")
            print(f"  X: [{seg_points[:,0].min():.3f}, {seg_points[:,0].max():.3f}]")
            print(f"  Y: [{seg_points[:,1].min():.3f}, {seg_points[:,1].max():.3f}]")
            print(f"  Z: [{seg_points[:,2].min():.3f}, {seg_points[:,2].max():.3f}]")

            centroid = seg_points.mean(axis=0)
            print(f"  Centroid: [{centroid[0]:.3f}, {centroid[1]:.3f}, {centroid[2]:.3f}] m")

        # --- Generate grasps on segmented object ---
        grasps, grasp_scores, grasp_colors = None, None, None
        if seg_points is not None:
            grasps, grasp_scores, grasp_colors = generate_grasps(seg_points)

        # --- Query robot end effector pose (full 4x4 transform) ---
        # TCP offset is configured in xArm Studio, so get_position() already
        # returns the grasp center (between fingertips), matching GraspGen poses.
        print("\n--- Querying robot end effector pose ---")
        robot_ip = "192.168.1.223"
        eef_transform = None  # 4x4 homogeneous transform in robot base frame (meters)
        try:
            arm = XArmAPI(robot_ip)
            arm.connect()
            code, pose = arm.get_position(is_radian=False)
            arm.disconnect()
            if code == 0:
                # pose is [x, y, z, roll, pitch, yaw] in mm and degrees
                eef_pos_m = np.array(pose[:3]) / 1000.0  # mm -> m
                eef_euler_deg = np.array(pose[3:6])
                eef_rot = spt.Rotation.from_euler("xyz", eef_euler_deg, degrees=True)

                eef_transform = np.eye(4)
                eef_transform[:3, :3] = eef_rot.as_matrix()
                eef_transform[:3, 3] = eef_pos_m

                print(f"  EEF position (grasp center): [{eef_pos_m[0]:.3f}, {eef_pos_m[1]:.3f}, {eef_pos_m[2]:.3f}] m")
                print(f"  EEF orientation: [{eef_euler_deg[0]:.1f}, {eef_euler_deg[1]:.1f}, {eef_euler_deg[2]:.1f}] deg (xyz euler)")
                print(f"  Frame convention: X=gripper width, Y=finger close, Z=approach")
            else:
                print(f"  Failed to get robot position (error code: {code})")
        except Exception as e:
            print(f"  Could not connect to robot at {robot_ip}: {e}")

        # --- Normalize grasp orientations for gripper symmetry ---
        # The parallel gripper is symmetric under 180-deg rotation around its
        # approach axis (Z).  GraspGen picks an arbitrary X/Y orientation, which
        # may require a needless 180-deg spin.  For each grasp, pick whichever
        # of the two Z-flips (original or +180 around Z) is closer to the
        # current EEF orientation.
        if grasps is not None and eef_transform is not None:
            print("\n--- Normalizing grasp orientations (gripper Z-symmetry) ---")
            Rz180 = np.eye(4)
            Rz180[0, 0] = -1
            Rz180[1, 1] = -1  # 180 deg around Z: flips X and Y, keeps Z

            eef_rot = spt.Rotation.from_matrix(eef_transform[:3, :3])
            n_flipped = 0
            for i in range(len(grasps)):
                rot_orig = spt.Rotation.from_matrix(grasps[i, :3, :3])
                rot_flip = rot_orig * spt.Rotation.from_matrix(Rz180[:3, :3])

                # Rotation needed to go from EEF to each candidate
                delta_orig = (rot_orig * eef_rot.inv()).magnitude()
                delta_flip = (rot_flip * eef_rot.inv()).magnitude()

                if delta_flip < delta_orig:
                    grasps[i] = grasps[i] @ Rz180
                    n_flipped += 1

            print(f"  Flipped {n_flipped}/{len(grasps)} grasps to reduce rotation from EEF")

        # --- Save point clouds ---
        print("\n--- Saving PLY files ---")

        # Full scene
        scene_pcd = o3d.geometry.PointCloud()
        scene_pcd.points = o3d.utility.Vector3dVector(full_points)
        scene_pcd.colors = o3d.utility.Vector3dVector(full_colors)
        o3d.io.write_point_cloud("full_scene.ply", scene_pcd)
        print(f"  full_scene.ply ({len(full_points)} points)")

        # Segmented object (red-tinted)
        if seg_points is not None:
            seg_pcd = o3d.geometry.PointCloud()
            seg_pcd.points = o3d.utility.Vector3dVector(seg_points)
            seg_pcd.colors = o3d.utility.Vector3dVector(tint_red(seg_colors))
            seg_filename = f"segmented_{text_prompt.replace(' ', '_')}.ply"
            o3d.io.write_point_cloud(seg_filename, seg_pcd)
            print(f"  {seg_filename} ({len(seg_points)} points)")

        # Grasps colored by confidence score (colors assigned before filtering)
        if grasps is not None and len(grasps) > 0:
            grasp_positions = grasps[:, :3, 3]

            grasp_pcd = o3d.geometry.PointCloud()
            grasp_pcd.points = o3d.utility.Vector3dVector(grasp_positions)
            grasp_pcd.colors = o3d.utility.Vector3dVector(grasp_colors)
            o3d.io.write_point_cloud("grasps.ply", grasp_pcd)
            print(f"  grasps.ply ({len(grasps)} grasp positions, green=best red=worst)")

            # Select best grasp: closest to EEF (least effort) if EEF available,
            # otherwise highest score
            grasp_positions = grasps[:, :3, 3]
            if eef_transform is not None:
                eef_pos = eef_transform[:3, 3]
                distances = np.linalg.norm(grasp_positions - eef_pos, axis=1)
                best_idx = int(np.argmin(distances))
                print(f"\n  Best grasp selection: closest to EEF (distance={distances[best_idx]:.3f} m)")
            else:
                best_idx = int(np.argmax(grasp_scores))
                print(f"\n  Best grasp selection: highest score (no EEF available)")

            best_pos = grasps[best_idx, :3, 3]
            best_rot = spt.Rotation.from_matrix(grasps[best_idx, :3, :3])
            best_euler = best_rot.as_euler("xyz", degrees=True)
            print(f"  Best grasp (#{best_idx}, score={grasp_scores[best_idx]:.3f}):")
            print(f"    Position: [{best_pos[0]:.3f}, {best_pos[1]:.3f}, {best_pos[2]:.3f}] m")
            print(f"    Orientation: [{best_euler[0]:.1f}, {best_euler[1]:.1f}, {best_euler[2]:.1f}] deg")

            # Save full grasp data as numpy for later use
            save_dict = {
                "grasps": grasps,           # Nx4x4 poses in robot frame (meters)
                "scores": grasp_scores,     # N confidence scores
                "colors": grasp_colors,     # Nx3 pre-computed colors
                "best_idx": best_idx,       # index of best grasp (closest to EEF)
            }
            if eef_transform is not None:
                save_dict["eef_transform"] = eef_transform  # 4x4 in robot frame (meters)
            np.savez("grasps.npz", **save_dict)
            print(f"  grasps.npz (full 4x4 poses + scores + colors + best_idx + eef)")

        # --- Build combined PLY with grasp arrows baked in ---
        print("\n--- Building combined.ply ---")

        all_points = [full_points]
        all_colors = [full_colors]

        # Add red-tinted segmented object
        if seg_points is not None:
            all_points.append(seg_points)
            all_colors.append(tint_red(seg_colors))

        # Add grasp arrows as point trails (using pre-computed colors)
        if grasps is not None and len(grasps) > 0:
            arrow_pts_list = []
            arrow_col_list = []
            arrow_length = 0.06  # 6cm arrows
            pts_per_arrow = 30

            purple = np.array([0.6, 0.0, 1.0])  # purple for best grasp

            for i, (g, color) in enumerate(zip(grasps, grasp_colors)):
                pos = g[:3, 3]
                z_axis = g[:3, :3][:, 2]  # approach direction
                c = purple if i == best_idx else color

                # Sample points along the approach arrow
                for t in np.linspace(0, 1, pts_per_arrow):
                    arrow_pts_list.append(pos + z_axis * arrow_length * t)
                    arrow_col_list.append(c)

                # Add a bright point at the grasp origin
                arrow_pts_list.append(pos)
                arrow_col_list.append(c)

            arrow_points = np.array(arrow_pts_list)
            arrow_colors = np.array(arrow_col_list)
            all_points.append(arrow_points)
            all_colors.append(arrow_colors)
            print(f"  Added {len(grasps)} grasp arrows ({len(arrow_points)} points)")

        combined_points = np.vstack(all_points)
        combined_colors = np.vstack(all_colors)

        combined_pcd = o3d.geometry.PointCloud()
        combined_pcd.points = o3d.utility.Vector3dVector(combined_points)
        combined_pcd.colors = o3d.utility.Vector3dVector(combined_colors)
        o3d.io.write_point_cloud("combined.ply", combined_pcd)
        print(f"  combined.ply ({len(combined_points)} points total)")

        # --- Print visualization command ---
        print("\n" + "=" * 60)
        print("Done. Visualize with:")
        print("=" * 60)
        print()
        print("  uv run python visualize.py")
        print()
        print("Legend:")
        print("  - RGB points: full scene")
        if seg_points is not None:
            print(f"  - Red-tinted points: segmented '{text_prompt}'")
        if grasps is not None:
            print(f"  - Green->Red arrows: {len(grasps)} grasps (green=high confidence, red=low)")
            print(f"  - Purple arrow = best grasp (closest to EEF / least effort)")
            print(f"  - Arrows point along approach direction (gripper Z-axis)")
        if eef_transform is not None:
            print(f"  - Current EEF: colored axes (R=X width, G=Y fingers, B=Z approach)")
            print(f"  - Best grasp: matching colored axes to compare orientation")
        print("  - Coordinate frame at origin: robot base (R=X, G=Y, B=Z)")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

    finally:
        merger.cleanup()


if __name__ == "__main__":
    main()
