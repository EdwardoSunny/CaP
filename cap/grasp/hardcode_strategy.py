"""Hardcode grasp strategy — use the segmented point cloud centroid directly.

No grasp generation model, no filtering, no scoring. Just computes the
centroid of the segmented object point cloud and returns a top-down grasp
at that location, using the current EEF orientation (or straight down if
EEF orientation is not available).
"""

import logging
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation

from cap.grasp.base import GraspResult, GraspStrategy

logger = logging.getLogger(__name__)


class HardcodeStrategy(GraspStrategy):
    """Return a grasp at the centroid of the segmented point cloud.

    The grasp pose is a 4x4 homogeneous transform in robot frame (meters):
    - Position = centroid of the object point cloud
    - Orientation = current EEF orientation (preserves whatever the arm is
      already doing), or straight-down if EEF is unavailable.
    """

    def generate_grasps(
        self,
        object_points: np.ndarray,
        object_colors: Optional[np.ndarray] = None,
        scene_points: Optional[np.ndarray] = None,
        scene_colors: Optional[np.ndarray] = None,
        current_eef_pose: Optional[np.ndarray] = None,
    ) -> GraspResult:
        if object_points is None or len(object_points) == 0:
            logger.warning("HardcodeStrategy: no object points, cannot compute centroid")
            return GraspResult(
                object_points=object_points,
                object_colors=object_colors,
            )

        centroid = object_points.mean(axis=0)  # (3,) in meters
        logger.info(f"HardcodeStrategy: centroid = [{centroid[0]:.4f}, {centroid[1]:.4f}, {centroid[2]:.4f}] m")

        # Build 4x4 grasp transform
        grasp = np.eye(4)
        grasp[:3, 3] = centroid

        if current_eef_pose is not None:
            # Preserve current EEF orientation
            grasp[:3, :3] = current_eef_pose[:3, :3]
            logger.info("HardcodeStrategy: using current EEF orientation")
        else:
            # Default: straight down (Z pointing down, X pointing forward)
            grasp[:3, :3] = Rotation.from_euler("xyz", [180, 0, 0], degrees=True).as_matrix()
            logger.info("HardcodeStrategy: using default top-down orientation")

        # Package as a single-grasp result
        all_grasps = grasp[np.newaxis, :, :]  # (1, 4, 4)
        all_scores = np.array([1.0])

        logger.info(f"HardcodeStrategy: returning centroid grasp at "
                     f"[{centroid[0]*1000:.1f}, {centroid[1]*1000:.1f}, {centroid[2]*1000:.1f}] mm")

        return GraspResult(
            best_grasp=grasp,
            best_score=1.0,
            best_idx=0,
            all_grasps=all_grasps,
            all_scores=all_scores,
            object_points=object_points,
            object_colors=object_colors,
        )
