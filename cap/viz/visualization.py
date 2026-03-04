"""
Extracted and deduplicated visualization helpers for grasp planning.

Module-level helpers (make_arrow, make_pose_axes, make_fingers, score_to_color)
are shared by GraspVisualizer.visualize_before_execution and
GraspVisualizer.visualize_after_execution.

Open3D is imported lazily inside each function that needs it, so the module can
be imported even when Open3D is not installed.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level helper functions
# ---------------------------------------------------------------------------

def make_arrow(origin, direction, length=0.06, color=None,
               cyl_r=0.003, cone_r=0.006, cone_frac=0.3):
    """Create an Open3D arrow mesh pointing from *origin* along *direction*."""
    import open3d as o3d

    if color is None:
        color = [1, 0, 0]

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
    """Draw RGB pose axes at the given 4x4 transform, with an optional label sphere."""
    import open3d as o3d

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
    """Draw a simplified parallel-jaw gripper at the given grasp pose."""
    import open3d as o3d

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


def score_to_color(score, s_min, s_max):
    """Map a scalar score to a red (low) -> green (high) colour."""
    if s_max == s_min:
        return np.array([0.0, 1.0, 0.0])
    t = (score - s_min) / (s_max - s_min)
    return np.array([1.0 - t, t, 0.0])


# ---------------------------------------------------------------------------
# GraspVisualizer class
# ---------------------------------------------------------------------------

class GraspVisualizer:
    """Interactive Open3D visualizer for grasp pipeline results.

    Unlike the original methods on ``LMPWrapper``, the EEF transform is passed
    in explicitly so this class has no dependency on the robot environment.
    """

    # ----- before execution -------------------------------------------------

    def visualize_before_execution(
        self, full_points, full_colors,
        seg_points, seg_colors,
        grasps, scores, best_idx, best_grasp,
        eef_transform, prompt,
    ):
        """Show an interactive Open3D visualization of the full pipeline output.

        Blocks until the user closes the window, then execution continues.

        Shows:
        - Full scene point cloud (original colors)
        - Segmented object (red-tinted)
        - All grasp arrows (green=high score, red=low)
        - Best grasp in purple with RGB axes + gripper fingers
        - Current EEF in cyan with RGB axes + gripper fingers
        - Robot base coordinate frame

        Parameters
        ----------
        eef_transform : np.ndarray (4x4) or None
            Current end-effector transform.  ``None`` is acceptable —
            the EEF will simply be omitted from the visualisation.
        """
        import open3d as o3d

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
        if eef_transform is not None:
            geometries.extend(make_pose_axes(
                eef_transform, axis_len=0.08, cyl_r=0.004, cone_r=0.008,
                label_color=[0.0, 1.0, 1.0], label_r=0.015,
            ))
            geometries.extend(make_fingers(eef_transform, color=[0.0, 0.8, 0.8]))

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

    # ----- after execution --------------------------------------------------

    def visualize_after_execution(
        self, full_points, full_colors,
        seg_points, seg_colors,
        best_grasp, eef_transform, prompt,
    ):
        """Show an "after" visualization using the same scene data from before,
        but with the CURRENT EEF position (after the robot moved).

        This lets you see whether the robot actually reached the target grasp.
        Shows the target grasp (purple) and the actual EEF (cyan) side by side.

        Parameters
        ----------
        eef_transform : np.ndarray (4x4) or None
            Current end-effector transform after execution.  ``None`` is
            acceptable — the EEF will simply be omitted.
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
        if eef_transform is not None:
            geometries.extend(make_pose_axes(
                eef_transform, axis_len=0.08, cyl_r=0.004, cone_r=0.008,
                label_color=[0.0, 1.0, 1.0], label_r=0.015,
            ))
            geometries.extend(make_fingers(eef_transform, color=[0.0, 0.8, 0.8]))

            # Log the error between target and actual
            if best_grasp is not None:
                pos_error = np.linalg.norm(eef_transform[:3, 3] - best_grasp[:3, 3]) * 1000  # mm
                eef_rot = spt.Rotation.from_matrix(eef_transform[:3, :3])
                target_rot = spt.Rotation.from_matrix(best_grasp[:3, :3])
                rot_error = (eef_rot * target_rot.inv()).magnitude() * 180 / np.pi  # deg
                logger.info(f"Execution error:")
                logger.info(f"  Position error: {pos_error:.1f} mm")
                logger.info(f"  Orientation error: {rot_error:.1f} deg")

        # --- Show ---
        logger.info("AFTER visualization: Purple=target grasp, Cyan=actual EEF position")
        logger.info("Close the window to continue.")
        o3d.visualization.draw_geometries(
            geometries,
            window_name=f"After Execution: '{prompt}' — Purple=target, Cyan=actual",
            width=1280, height=720,
        )
        logger.info("After-visualization closed.")
