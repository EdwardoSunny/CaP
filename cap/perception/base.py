"""Abstract base class for perception modules."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List
import numpy as np


@dataclass
class PerceptionResult:
    """Result returned by a perception module."""

    # Segmented object point cloud in robot frame (meters)
    object_points: Optional[np.ndarray] = None
    object_colors: Optional[np.ndarray] = None

    # Full scene point cloud in robot frame (meters)
    scene_points: Optional[np.ndarray] = None
    scene_colors: Optional[np.ndarray] = None

    @property
    def success(self) -> bool:
        return self.object_points is not None and len(self.object_points) > 0


class PerceptionModule(ABC):
    """
    Abstract interface for perception (point cloud capture + object segmentation).

    Implementations handle camera management, depth processing, and object segmentation.
    """

    @abstractmethod
    def segment_object(self, text_prompt: str) -> PerceptionResult:
        """
        Capture point clouds from cameras, segment the target object, and return
        the segmented + full scene point clouds in robot frame.

        Args:
            text_prompt: Natural language description of the object to find
                         (e.g., "red cup", "tissue box")

        Returns:
            PerceptionResult with segmented object and full scene point clouds.
        """
        ...

    @abstractmethod
    def capture_scene(self) -> PerceptionResult:
        """
        Capture point clouds from cameras without segmentation.

        Returns:
            PerceptionResult with scene_points/scene_colors filled,
            object_points/object_colors will be None.
        """
        ...

    @abstractmethod
    def cleanup(self):
        """Release camera resources."""
        ...
