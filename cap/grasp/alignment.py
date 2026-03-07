"""Object-aligned yaw computation for grasp orientation.

Computes the optimal gripper yaw (rotation around Z) so that the gripper's
finger-gap axis (X in robot TCP frame) is aligned with the object's longest
bounding-box side in the XY plane. This ensures the fingers close across the
narrow dimension of elongated objects regardless of their orientation on the
table.

Robot TCP frame convention:
    Z = approach direction (points down for top-down grasps)
    Y = finger closing direction
    X = finger width / gap direction (parallel to the opening between fingers)
"""

import logging

import numpy as np
from scipy.spatial import ConvexHull

logger = logging.getLogger(__name__)


def compute_object_yaw(object_points: np.ndarray) -> float:
    """Compute the yaw angle (degrees) that aligns the gripper with an object.

    Projects the object point cloud onto the XY plane, computes the minimum-area
    oriented bounding box via the convex hull + rotating calipers, finds the
    longest side of that box, and returns its angle relative to the world X-axis.

    When used as the yaw in a top-down grasp orientation ``[180, 0, yaw]``,
    the gripper's X-axis (finger gap) will be parallel to the object's longest
    bounding-box side, so the fingers close on the short side.

    Args:
        object_points: Nx3 point cloud in robot frame (meters).

    Returns:
        Yaw angle in degrees, in the range (-90, 90].
    """
    if object_points is None or len(object_points) < 3:
        logger.warning("compute_object_yaw: not enough points, returning 0")
        return 0.0

    # Project onto XY
    xy = object_points[:, :2]  # (N, 2)

    # Convex hull of the 2D projection
    try:
        hull = ConvexHull(xy)
    except Exception:
        logger.warning("compute_object_yaw: convex hull failed, returning 0")
        return 0.0

    hull_pts = xy[hull.vertices]  # ordered hull vertices

    # Test each edge of the convex hull as a candidate rotation.
    # For each edge angle, rotate all hull points so the edge is axis-aligned,
    # compute the axis-aligned bounding box, and track the one with minimum area.
    best_angle = 0.0
    best_area = float("inf")
    best_width = 0.0
    best_height = 0.0

    n_hull = len(hull_pts)
    for i in range(n_hull):
        # Edge vector
        edge = hull_pts[(i + 1) % n_hull] - hull_pts[i]
        angle = np.arctan2(edge[1], edge[0])

        # Rotation matrix to align this edge with the X-axis
        cos_a, sin_a = np.cos(-angle), np.sin(-angle)
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])

        rotated = hull_pts @ rot.T
        min_xy = rotated.min(axis=0)
        max_xy = rotated.max(axis=0)
        extent = max_xy - min_xy  # [width along edge, height perpendicular]

        area = extent[0] * extent[1]
        if area < best_area:
            best_area = area
            best_angle = angle
            best_width = extent[0]   # length along this edge direction
            best_height = extent[1]  # length perpendicular to it

    # best_angle is the angle of the edge that produced the minimum bounding box.
    # best_width is the extent along that edge, best_height is perpendicular.
    # We want the gripper X-axis (finger gap) aligned with the LONGEST side.
    if best_height > best_width:
        # The longest side is perpendicular to the edge — rotate 90 deg
        yaw_rad = best_angle + np.pi / 2
    else:
        # The longest side is along the edge
        yaw_rad = best_angle

    # Normalize to (-90, 90] — the gripper is symmetric (finger gap has no
    # "forward" direction), so 0 deg and 180 deg are equivalent.
    while yaw_rad > np.pi / 2:
        yaw_rad -= np.pi
    while yaw_rad <= -np.pi / 2:
        yaw_rad += np.pi

    yaw_deg = np.degrees(yaw_rad)

    logger.info(
        f"compute_object_yaw: bbox longest side angle = {yaw_deg:.1f} deg, "
        f"bbox size = {best_width*1000:.1f} x {best_height*1000:.1f} mm "
        f"(area = {best_area*1e6:.1f} mm^2)"
    )

    return float(yaw_deg)
