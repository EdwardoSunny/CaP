#!/usr/bin/env python3
"""
Visualize saved output from test_segmentation_pipeline.py.

Loads the saved PLY and NPZ files and shows them in Open3D with proper
3D geometry: thick arrows, gripper fingers, best grasp highlight.

Usage:
    python visualize.py                        # show combined scene + grasps
    python visualize.py --scene-only           # full scene only, no grasps
    python visualize.py --top 10               # only show top 10 grasps
    python visualize.py --no-scene             # segmented object + grasps only
"""

import argparse
import sys
import numpy as np
import open3d as o3d
import scipy.spatial.transform as spt
from pathlib import Path


def score_to_color(score, min_score, max_score):
    """Map score to green (best) -> yellow -> red (worst)."""
    if max_score == min_score:
        return np.array([0.0, 1.0, 0.0])
    t = (score - min_score) / (max_score - min_score)
    return np.array([1.0 - t, t, 0.0])


def make_pose_axes(transform_4x4, axis_length=0.08, cylinder_radius=0.004,
                   cone_radius=0.008, label_color=None, label_radius=0.012):
    """
    Draw 3 colored axis arrows from a 4x4 homogeneous transform.

    Colors match the robot EEF convention:
      Red   = X axis (gripper width direction)
      Green = Y axis (finger closing direction)
      Blue  = Z axis (approach direction)

    Optionally adds a sphere at the origin with label_color.

    Returns a list of TriangleMesh geometries.
    """
    axis_colors = [
        [1.0, 0.0, 0.0],  # X = red
        [0.0, 1.0, 0.0],  # Y = green
        [0.0, 0.0, 1.0],  # Z = blue
    ]

    origin = transform_4x4[:3, 3]
    rot = transform_4x4[:3, :3]
    meshes = []

    for i, color in enumerate(axis_colors):
        direction = rot[:, i]  # i-th column = i-th axis
        arrow = make_arrow_mesh(
            origin, direction,
            length=axis_length,
            color=color,
            cylinder_radius=cylinder_radius,
            cone_radius=cone_radius,
        )
        meshes.append(arrow)

    if label_color is not None:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=label_radius)
        sphere.paint_uniform_color(label_color)
        sphere.compute_vertex_normals()
        sphere.translate(origin)
        meshes.append(sphere)

    return meshes


def make_arrow_mesh(origin, direction, length=0.06, color=[1, 0, 0],
                    cylinder_radius=0.003, cone_radius=0.006, cone_frac=0.3):
    """
    Create a solid 3D arrow (cylinder + cone) from origin along direction.
    Returns a TriangleMesh painted with the given color.
    """
    cyl_height = length * (1 - cone_frac)
    cone_height = length * cone_frac

    arrow = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=cylinder_radius,
        cone_radius=cone_radius,
        cylinder_height=cyl_height,
        cone_height=cone_height,
        resolution=8,
        cylinder_split=1,
        cone_split=1,
    )
    arrow.paint_uniform_color(color)
    arrow.compute_vertex_normals()

    direction = direction / np.linalg.norm(direction)
    z_axis = np.array([0.0, 0.0, 1.0])

    v = np.cross(z_axis, direction)
    s = np.linalg.norm(v)
    c = np.dot(z_axis, direction)

    if s < 1e-8:
        if c > 0:
            R = np.eye(3)
        else:
            R = np.diag([-1.0, -1.0, 1.0])
    else:
        vx = np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0],
        ])
        R = np.eye(3) + vx + vx @ vx * (1 - c) / (s * s)

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = origin
    arrow.transform(T)

    return arrow


def make_gripper_buffer(grasp_4x4, color,
                        size_x=0.05, size_y=0.16, size_z=0.005):
    """
    Create a thin slab representing the gripper clearance buffer,
    centered at the grasp origin, oriented by the grasp rotation.

    The buffer is the cross-section that must not intersect the table:
      X (red, thickness):    5 cm  (±2.5 cm)
      Y (green, finger span): 16 cm (±8 cm)
      Z (blue, approach):    thin slab (just for visibility)
    """
    box = o3d.geometry.TriangleMesh.create_box(
        width=size_x, height=size_y, depth=size_z
    )
    # Center the box at origin
    box.translate([-size_x / 2, -size_y / 2, -size_z / 2])
    box.paint_uniform_color(color)
    box.compute_vertex_normals()

    T = np.eye(4)
    T[:3, :3] = grasp_4x4[:3, :3]
    T[:3, 3] = grasp_4x4[:3, 3]
    box.transform(T)

    return box


def make_gripper_fingers(grasp_4x4, color, width=0.08, depth=0.05, thickness=0.012):
    """
    Create two boxes representing gripper fingers + a palm bar connecting them.
    """
    pos = grasp_4x4[:3, 3]
    rot = grasp_4x4[:3, :3]

    finger_half_gap = width / 2
    meshes = []

    for sign in [1, -1]:
        box = o3d.geometry.TriangleMesh.create_box(
            width=thickness, height=thickness, depth=depth
        )
        box.paint_uniform_color(color)
        box.compute_vertex_normals()

        box_center = np.array([thickness / 2, thickness / 2, depth / 2])
        T = np.eye(4)
        T[:3, :3] = rot
        T[:3, 3] = pos + rot[:, 1] * sign * finger_half_gap - rot @ box_center
        box.transform(T)
        meshes.append(box)

    palm = o3d.geometry.TriangleMesh.create_box(
        width=thickness, height=width, depth=thickness
    )
    palm.paint_uniform_color(color)
    palm.compute_vertex_normals()

    palm_center = np.array([thickness / 2, width / 2, thickness / 2])
    T_palm = np.eye(4)
    T_palm[:3, :3] = rot
    T_palm[:3, 3] = pos - rot @ palm_center
    palm.transform(T_palm)
    meshes.append(palm)

    return meshes


def main():
    parser = argparse.ArgumentParser(description="Visualize segmentation + grasp pipeline output")
    parser.add_argument("--scene-only", action="store_true", help="Show full scene only, no grasps")
    parser.add_argument("--no-scene", action="store_true", help="Show segmented object + grasps only (no full scene)")
    parser.add_argument("--top", type=int, default=0, help="Only show top N grasps by score (0 = all)")
    parser.add_argument("--seg", type=str, default=None, help="Path to segmented PLY (auto-detected if not given)")
    args = parser.parse_args()

    geometries = []

    # --- Load full scene ---
    scene_path = Path("full_scene.ply")
    if not args.no_scene and scene_path.exists():
        scene_pcd = o3d.io.read_point_cloud(str(scene_path))
        print(f"Loaded full scene: {len(scene_pcd.points)} points")
        geometries.append(scene_pcd)
    elif not args.no_scene:
        print("full_scene.ply not found, skipping.")

    # --- Load segmented object ---
    seg_path = None
    if args.seg:
        seg_path = Path(args.seg)
    else:
        candidates = sorted(Path(".").glob("segmented_*.ply"))
        if candidates:
            seg_path = candidates[0]

    if seg_path and seg_path.exists():
        seg_pcd = o3d.io.read_point_cloud(str(seg_path))
        print(f"Loaded segmented object: {seg_path.name} ({len(seg_pcd.points)} points)")
        geometries.append(seg_pcd)
    else:
        print("No segmented PLY found, skipping.")

    # --- Robot base coordinate frame ---
    robot_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15)
    geometries.append(robot_frame)

    # --- Load grasps ---
    grasps_path = Path("grasps.npz")
    if not args.scene_only and grasps_path.exists():
        data = np.load(str(grasps_path), allow_pickle=True)
        grasps = data["grasps"]       # Nx4x4
        scores = data["scores"]       # N
        # Use pre-computed colors if available (assigned before filtering)
        has_precomputed_colors = "colors" in data
        if has_precomputed_colors:
            colors = data["colors"]   # Nx3
        # Load EEF pose (full 4x4 transform, or legacy position-only)
        has_eef = "eef_transform" in data or "eef_position" in data
        eef_transform = None
        if "eef_transform" in data:
            eef_transform = data["eef_transform"]  # 4x4
        elif "eef_position" in data:
            # Legacy: position only, no rotation info
            eef_transform = np.eye(4)
            eef_transform[:3, 3] = data["eef_position"]

        # Determine best grasp: closest to EEF (least effort), or saved best_idx
        has_best_idx = "best_idx" in data
        if has_best_idx:
            best_idx = int(data["best_idx"])
        elif has_eef and eef_transform is not None:
            # Recompute: pick grasp closest to EEF
            eef_pos = eef_transform[:3, 3]
            grasp_positions = grasps[:, :3, 3]
            distances = np.linalg.norm(grasp_positions - eef_pos, axis=1)
            best_idx = int(np.argmin(distances))
        else:
            # Fallback: highest score
            best_idx = int(np.argmax(scores))

        # Limit to top N if requested (keep best_idx in range)
        if args.top > 0:
            # Always include the best grasp even if outside top N by score
            order = np.argsort(scores)[::-1]
            top_indices = set(order[:args.top].tolist())
            top_indices.add(best_idx)
            top_indices = sorted(top_indices)
            grasps = grasps[top_indices]
            scores = scores[top_indices]
            if has_precomputed_colors:
                colors = colors[top_indices]
            # Remap best_idx to new position
            best_idx = top_indices.index(best_idx) if best_idx in top_indices else 0

        print(f"Showing {len(grasps)} grasps (scores: {scores.min():.3f} - {scores.max():.3f})")

        best_grasp = grasps[best_idx]
        best_score = scores[best_idx]
        best_pos = best_grasp[:3, 3]
        best_euler = spt.Rotation.from_matrix(best_grasp[:3, :3]).as_euler("xyz", degrees=True)

        selection = "closest to EEF" if has_eef or has_best_idx else "highest score"
        print(f"Best grasp ({selection}): #{best_idx}, score={best_score:.3f}")
        print(f"  Position: [{best_pos[0]:.3f}, {best_pos[1]:.3f}, {best_pos[2]:.3f}] m")
        print(f"  Orientation: [{best_euler[0]:.1f}, {best_euler[1]:.1f}, {best_euler[2]:.1f}] deg")
        if has_eef and eef_transform is not None:
            eef_pos = eef_transform[:3, 3]
            dist = np.linalg.norm(best_pos - eef_pos)
            print(f"  Distance to EEF: {dist:.3f} m")
            eef_euler = spt.Rotation.from_matrix(eef_transform[:3, :3]).as_euler("xyz", degrees=True)
            print(f"  Current EEF position: [{eef_pos[0]:.3f}, {eef_pos[1]:.3f}, {eef_pos[2]:.3f}] m")
            print(f"  Current EEF orientation: [{eef_euler[0]:.1f}, {eef_euler[1]:.1f}, {eef_euler[2]:.1f}] deg")

        # Add solid 3D arrow for each grasp (approach direction = Z-axis)
        purple = [0.6, 0.0, 1.0]  # purple for best grasp (closest to EEF)

        for i, g in enumerate(grasps):
            pos = g[:3, 3]
            z_axis = g[:3, :3][:, 2]
            is_best = (i == best_idx)

            if is_best:
                color = purple
            elif has_precomputed_colors:
                color = colors[i]
            else:
                color = score_to_color(scores[i], scores.min(), scores.max())

            arrow = make_arrow_mesh(
                pos, z_axis,
                length=0.08 if is_best else 0.05,
                color=color,
                cylinder_radius=0.004 if is_best else 0.002,
                cone_radius=0.008 if is_best else 0.005,
            )
            geometries.append(arrow)

            # Small sphere at every grasp origin so positions are visible
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.005 if not is_best else 0.01)
            sphere.paint_uniform_color(color)
            sphere.compute_vertex_normals()
            sphere.translate(pos)
            geometries.append(sphere)

        # --- Best grasp: RGB axis arrows + purple sphere + gripper fingers ---
        # Axes: Red=X (width), Green=Y (fingers), Blue=Z (approach)
        grasp_axes = make_pose_axes(
            best_grasp,
            axis_length=0.08,
            cylinder_radius=0.004,
            cone_radius=0.008,
            label_color=purple,  # purple sphere at origin
            label_radius=0.015,
        )
        geometries.extend(grasp_axes)

        # Gripper clearance buffer + fingers at best grasp (purple-tinted)
        grasp_buf = make_gripper_buffer(best_grasp, color=[0.5, 0.2, 0.8])
        geometries.append(grasp_buf)
        fingers = make_gripper_fingers(best_grasp, color=[0.5, 0.2, 0.8])
        geometries.extend(fingers)

        # --- Current robot EEF: same RGB axes + cyan sphere + gripper box ---
        # Matching colors let you compare current vs desired orientation directly
        if has_eef and eef_transform is not None:
            eef_pos = eef_transform[:3, 3]
            print(f"EEF position: [{eef_pos[0]:.3f}, {eef_pos[1]:.3f}, {eef_pos[2]:.3f}] m")

            eef_axes = make_pose_axes(
                eef_transform,
                axis_length=0.08,
                cylinder_radius=0.004,
                cone_radius=0.008,
                label_color=[0.0, 1.0, 1.0],  # cyan sphere at origin
                label_radius=0.015,
            )
            geometries.extend(eef_axes)

            # Gripper clearance buffer + fingers at current EEF for comparison
            eef_buf = make_gripper_buffer(eef_transform, color=[0.0, 0.8, 0.8])
            geometries.append(eef_buf)
            eef_fingers = make_gripper_fingers(eef_transform, color=[0.0, 0.8, 0.8])
            geometries.extend(eef_fingers)

    elif not args.scene_only:
        print("grasps.npz not found, skipping grasp visualization.")

    # --- Show ---
    if len(geometries) <= 1:
        print("Nothing to visualize.")
        sys.exit(1)

    print()
    print("Legend:")
    print("  Coordinate frame at origin = robot base")
    if not args.scene_only and grasps_path.exists():
        print(f"  Green->Red 3D arrows = all grasps (green=high score, red=low)")
        print(f"  Purple sphere + slab + arrows = best grasp (closest to EEF)")
        print(f"  Axis colors (same for EEF and best grasp):")
        print(f"    Red   = X axis (gripper width direction)")
        print(f"    Green = Y axis (finger closing direction)")
        print(f"    Blue  = Z axis (approach direction)")
        if has_eef:
            print(f"  Cyan sphere + slab = current robot EEF (slab = gripper clearance buffer)")
    print()
    print("Close the window to exit.")

    o3d.visualization.draw_geometries(
        geometries,
        window_name="Pipeline Visualization",
        width=1280,
        height=720,
    )


if __name__ == "__main__":
    main()
