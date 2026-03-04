"""Abstract base class for grasp strategies."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class GraspResult:
    """Result returned by a grasp strategy.

    All coordinates are in the robot base frame.
    Positions in **meters**, orientations as 4x4 homogeneous transforms.
    The consumer (LMPWrapper) is responsible for converting to mm/degrees
    before sending commands to the motion controller.
    """

    # Best grasp as a 4x4 homogeneous transform (meters, robot frame)
    best_grasp: Optional[np.ndarray] = None

    # Confidence score of the best grasp [0-1]
    best_score: float = 0.0

    # Index of the best grasp in the `all_grasps` array
    best_idx: Optional[int] = None

    # All candidate grasps (Nx4x4) after filtering
    all_grasps: Optional[np.ndarray] = None

    # Scores for all candidate grasps
    all_scores: Optional[np.ndarray] = None

    # The cleaned segmented object point cloud used for grasp generation (meters)
    object_points: Optional[np.ndarray] = None
    object_colors: Optional[np.ndarray] = None

    # Full scene point cloud for visualization context (meters)
    scene_points: Optional[np.ndarray] = None
    scene_colors: Optional[np.ndarray] = None

    @property
    def success(self) -> bool:
        return self.best_grasp is not None


class GraspStrategy(ABC):
    """
    Abstract interface for grasp pose generation.

    Implementations receive a segmented object point cloud (in robot frame, meters)
    and must return a GraspResult with the best grasp pose.

    Examples of concrete strategies:
    - GraspGenStrategy: uses NVIDIA GraspGen diffusion model
    - VLMGraspStrategy: uses a VLM to predict grasp locations
    - HeuristicGraspStrategy: rule-based top-down grasps at object centroid
    """

    @abstractmethod
    def generate_grasps(
        self,
        object_points: np.ndarray,
        object_colors: Optional[np.ndarray] = None,
        scene_points: Optional[np.ndarray] = None,
        scene_colors: Optional[np.ndarray] = None,
        current_eef_pose: Optional[np.ndarray] = None,
    ) -> GraspResult:
        """
        Generate and select a grasp pose for the given object point cloud.

        Args:
            object_points: Nx3 segmented object points in robot frame (meters)
            object_colors: Nx3 RGB colors [0-1] (optional, some strategies may use)
            scene_points: Mx3 full scene points (optional, for collision checking)
            scene_colors: Mx3 full scene colors (optional)
            current_eef_pose: 4x4 current EEF transform (optional, for proximity-based selection)

        Returns:
            GraspResult with the selected grasp and metadata.
        """
        ...
