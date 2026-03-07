"""Abstract base class for robot motion controllers."""

from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np


class MotionController(ABC):
    """
    Abstract interface for robot motion control.

    All positions are in millimeters, orientations in degrees (xyz intrinsic euler).
    Poses are [x, y, z, roll, pitch, yaw].
    """

    @abstractmethod
    def get_robot_pos(self) -> np.ndarray:
        """Return robot EEF [x, y, z] position in mm."""
        ...

    @abstractmethod
    def get_robot_pose(self) -> np.ndarray:
        """Return full robot pose [x, y, z, roll, pitch, yaw] in mm/degrees."""
        ...

    @abstractmethod
    def move_to_pose(
        self,
        target_position: List[float],
        target_orientation: List[float],
        duration: float = 3.0,
        stage_val: int = 0,
    ) -> bool:
        """
        Move EEF to an absolute pose.

        Args:
            target_position: [x, y, z] in mm
            target_orientation: [roll, pitch, yaw] in degrees
            duration: seconds to complete motion
            stage_val: stage value for action recording

        Returns:
            True if motion succeeded.
        """
        ...

    @abstractmethod
    def move_relative(
        self,
        delta_position: List[float],
        delta_orientation: Optional[List[float]] = None,
        duration: float = 1.0,
        stage_val: int = 0,
    ) -> bool:
        """
        Move EEF relative to current pose.

        Args:
            delta_position: [dx, dy, dz] in mm
            delta_orientation: [droll, dpitch, dyaw] in degrees (default [0,0,0])
            duration: seconds to complete motion
            stage_val: stage value for action recording

        Returns:
            True if motion succeeded.
        """
        ...

    @abstractmethod
    def set_gripper(self, grasp_value: float, stage_val: int = 0) -> bool:
        """
        Set gripper opening.

        Args:
            grasp_value: 0.0 = fully open, 1.0 = fully closed
            stage_val: stage value for action recording

        Returns:
            True if gripper command succeeded.
        """
        ...

    @abstractmethod
    def hold_position(self, duration: float, stage_val: int = 0):
        """Hold current position for *duration* seconds."""
        ...

    # --- convenience helpers (non-abstract, built on the primitives above) ---

    def get_robot_xy(self) -> np.ndarray:
        """Return robot EEF [x, y] position in mm."""
        return self.get_robot_pos()[:2]

    def open_gripper(self, stage_val: int = 0) -> bool:
        return self.set_gripper(0.0, stage_val)

    def close_gripper(self, stage_val: int = 0) -> bool:
        return self.set_gripper(1.0, stage_val)

    def align_gripper_yaw(self, target_yaw_deg: float, duration: float = 1.0, stage_val: int = 0) -> bool:
        """Rotate only the gripper yaw (wrist) to *target_yaw_deg*.

        Default implementation uses :meth:`move_to_pose` keeping position
        and roll/pitch unchanged.
        """
        pose = self.get_robot_pose()
        return self.move_to_pose(
            target_position=pose[:3].tolist(),
            target_orientation=[pose[3], pose[4], target_yaw_deg],
            duration=duration,
            stage_val=stage_val,
        )
