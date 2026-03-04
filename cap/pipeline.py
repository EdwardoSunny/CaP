"""
Pipeline orchestrator: perception -> grasp generation -> visualization.

Replaces the monolithic `detect_object_location` method from LMPWrapper,
decoupling perception, grasp strategy, and visualization into swappable components.
"""

import logging
from typing import Optional

import numpy as np

from cap.grasp.base import GraspResult, GraspStrategy
from cap.perception.base import PerceptionModule
from cap.viz import GraspVisualizer

logger = logging.getLogger(__name__)


class GraspPipeline:
    """
    Orchestrates: perception -> grasp generation -> visualization.

    Decoupled from both the robot motion layer and the LMP layer.
    You can swap perception or grasp strategy independently.
    """

    def __init__(
        self,
        perception: PerceptionModule,
        grasp_strategy: GraspStrategy,
        visualizer: Optional[GraspVisualizer] = None,
    ):
        self.perception = perception
        self.grasp_strategy = grasp_strategy
        self.visualizer = visualizer

    def detect_and_grasp(
        self,
        text_prompt: str,
        current_eef_pose: Optional[np.ndarray] = None,
    ) -> GraspResult:
        """
        Full pipeline: segment object -> generate grasps -> select best -> visualize.

        Args:
            text_prompt: Object description (e.g. "red cup")
            current_eef_pose: 4x4 EEF transform in robot frame (meters).
                             Used for proximity-based grasp selection.

        Returns:
            GraspResult with the best grasp and all metadata.
        """
        # 1. Perception: segment object
        logger.info(f"Segmenting '{text_prompt}'...")
        perception_result = self.perception.segment_object(text_prompt)

        if not perception_result.success:
            logger.warning(f"No points found for object: {text_prompt}")
            return GraspResult()

        logger.info(
            f"Segmented point cloud: {len(perception_result.object_points)} points"
        )

        # 2. Grasp generation + selection
        logger.info("Generating grasps on segmented point cloud...")
        grasp_result = self.grasp_strategy.generate_grasps(
            object_points=perception_result.object_points,
            object_colors=perception_result.object_colors,
            scene_points=perception_result.scene_points,
            scene_colors=perception_result.scene_colors,
            current_eef_pose=current_eef_pose,
        )

        if not grasp_result.success:
            logger.warning("No valid grasps generated")
            return grasp_result

        # Attach scene data to result for visualization
        grasp_result.scene_points = perception_result.scene_points
        grasp_result.scene_colors = perception_result.scene_colors

        # 3. Visualization (if available)
        if self.visualizer is not None and grasp_result.success:
            try:
                self.visualizer.visualize_before_execution(
                    full_points=perception_result.scene_points,
                    full_colors=perception_result.scene_colors,
                    seg_points=grasp_result.object_points,
                    seg_colors=grasp_result.object_colors,
                    grasps=grasp_result.all_grasps,
                    scores=grasp_result.all_scores,
                    best_idx=grasp_result.best_idx,
                    best_grasp=grasp_result.best_grasp,
                    eef_transform=current_eef_pose,
                    prompt=text_prompt,
                )
            except Exception as e:
                logger.warning(f"Visualization failed: {e}")

        return grasp_result

    @staticmethod
    def grasp_to_pose_6dof(grasp_result: GraspResult) -> Optional[np.ndarray]:
        """
        Convert the best grasp (4x4 matrix in meters) to [x, y, z, roll, pitch, yaw]
        in millimeters and degrees (robot command format).

        Returns None if no valid grasp.
        """
        if not grasp_result.success:
            return None

        import scipy.spatial.transform as st

        position_m = grasp_result.best_grasp[:3, 3]
        position_mm = position_m * 1000.0
        rotation = st.Rotation.from_matrix(grasp_result.best_grasp[:3, :3])
        orientation_deg = rotation.as_euler("xyz", degrees=True)

        return np.concatenate([position_mm, orientation_deg])
