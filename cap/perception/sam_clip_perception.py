"""Molmo+SAM perception module wrapping segment_pc.RobotFrameMerger.

SAM2AutomaticMaskGenerator produces all candidate masks unconditionally.
Molmo (a VLM hosted via OpenAI-compatible API) predicts point locations on the
target object.  Those points vote on which auto-generated mask corresponds to
the target, and that mask is applied to the depth-derived 3D point cloud.
"""

from typing import List

from cap.perception.base import PerceptionModule, PerceptionResult
from cap.segment_pc import RobotFrameMerger, load_sam_mask_generator


class SAMMolmoPerception(PerceptionModule):
    """
    Perception module using SAM2 automatic mask generation and
    Molmo (VLM) point voting for object selection.

    The merger is created fresh each call because it manages camera pipelines
    (start/stop). This matches the existing usage pattern.
    """

    def __init__(
        self,
        camera_serials: List[str],
        calib_file: str,
        sam2_checkpoint: str,
        sam2_config: str,
        device: str = "cuda",
    ):
        """
        Args:
            camera_serials: List of RealSense camera serial numbers.
            calib_file: Path to calibration transforms file (transforms.npy).
            sam2_checkpoint: Path to SAM2 checkpoint file.
            sam2_config: Path to SAM2 config YAML file.
            device: Device to load models on ('cuda' or 'cpu').
        """
        self.camera_serials = camera_serials
        self.calib_file = calib_file
        self.device = device

        # Pre-load SAM2 automatic mask generator once so it persists across calls
        self.sam_mask_generator = load_sam_mask_generator(
            sam2_checkpoint=sam2_checkpoint,
            sam2_config=sam2_config,
            device=device,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_merger(self) -> RobotFrameMerger:
        """Create a fresh RobotFrameMerger (opens camera pipelines)."""
        return RobotFrameMerger(
            camera_serials=self.camera_serials,
            calib_file=self.calib_file,
            sam_mask_generator=self.sam_mask_generator,
            device=self.device,
        )

    # ------------------------------------------------------------------
    # PerceptionModule interface
    # ------------------------------------------------------------------

    def segment_object(self, text_prompt: str) -> PerceptionResult:
        merger = self._make_merger()
        try:
            # Full scene (no segmentation)
            scene_points, scene_colors = merger.capture_merged_pointcloud(
                text_prompt=None,
            )

            # Segmented object via Molmo+SAM
            object_points, object_colors = merger.capture_merged_pointcloud(
                text_prompt=text_prompt,
            )

            return PerceptionResult(
                object_points=object_points,
                object_colors=object_colors,
                scene_points=scene_points,
                scene_colors=scene_colors,
            )
        finally:
            merger.cleanup()

    def capture_scene(self) -> PerceptionResult:
        merger = self._make_merger()
        try:
            scene_points, scene_colors = merger.capture_merged_pointcloud(
                text_prompt=None,
            )

            return PerceptionResult(
                scene_points=scene_points,
                scene_colors=scene_colors,
            )
        finally:
            merger.cleanup()

    def cleanup(self):
        # No-op: cameras are opened/closed per call by the merger.
        pass
