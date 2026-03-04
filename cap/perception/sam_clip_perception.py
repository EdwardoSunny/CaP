"""SAM+CLIP perception module wrapping segment_pc.RobotFrameMerger."""

from typing import List

from cap.perception.base import PerceptionModule, PerceptionResult
from cap.segment_pc import RobotFrameMerger, load_segmentation_models


class SAMCLIPPerception(PerceptionModule):
    """
    Perception module that uses SAM2 for segmentation and CLIP for
    text-guided object identification, built on top of RobotFrameMerger.

    The merger is created fresh each call because it manages camera pipelines
    (start/stop). This matches the existing usage pattern.
    """

    def __init__(
        self,
        camera_serials: List[str],
        calib_file: str,
        sam2_checkpoint: str,
        sam2_config: str,
        clip_model_name: str,
        device: str = "cuda",
    ):
        """
        Args:
            camera_serials: List of RealSense camera serial numbers.
            calib_file: Path to calibration transforms file (transforms.npy).
            sam2_checkpoint: Path to SAM2 checkpoint file.
            sam2_config: Path to SAM2 config YAML file.
            clip_model_name: HuggingFace model name for CLIP.
            device: Device to load models on ('cuda' or 'cpu').
        """
        self.camera_serials = camera_serials
        self.calib_file = calib_file
        self.device = device

        # Pre-load heavy models once so they persist across calls
        self.sam_generator, self.clip_model, self.clip_processor = (
            load_segmentation_models(
                sam2_checkpoint=sam2_checkpoint,
                sam2_config=sam2_config,
                clip_model_name=clip_model_name,
                device=device,
            )
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_merger(self) -> RobotFrameMerger:
        """Create a fresh RobotFrameMerger (opens camera pipelines)."""
        return RobotFrameMerger(
            camera_serials=self.camera_serials,
            calib_file=self.calib_file,
            sam_generator=self.sam_generator,
            clip_model=self.clip_model,
            clip_processor=self.clip_processor,
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

            # Segmented object
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
