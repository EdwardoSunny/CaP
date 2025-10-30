#!/usr/bin/env python3
"""
Simple test script for point cloud capture with segmentation.

Usage:
    python test_simple.py "cup"
    python test_simple.py "orange disinfecting wipes"
    python test_simple.py  # No segmentation, capture everything
"""

import sys
from pathlib import Path
from cap.segment_pc import RobotFrameMerger, load_segmentation_models
import torch


def main():
    # Get text prompt from command line (optional)
    text_prompt = sys.argv[1] if len(sys.argv) > 1 else None

    # Configuration
    camera_serials = ["327122079374", "317422074281"]
    calib_file = Path("transforms/transforms.npy")

    print(f"Camera serials: {camera_serials}")
    print(f"Text prompt: {text_prompt if text_prompt else 'None (capturing everything)'}")

    # Load segmentation models if we have a prompt
    sam_gen, clip_model, clip_proc = None, None, None
    device = "cpu"

    if text_prompt:
        print("\nLoading AI models...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        sam_gen, clip_model, clip_proc = load_segmentation_models(
            sam2_checkpoint=Path("ckpt/sam2.1_hiera_large.pt"),
            sam2_config=Path("configs/sam2.1/sam2.1_hiera_l.yaml"),
            clip_model_name="laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
            device=device
        )
        print("Models loaded!")

    # Create merger
    print("\nInitializing cameras...")
    merger = RobotFrameMerger(
        camera_serials=camera_serials,
        calib_file=calib_file,
        sam_generator=sam_gen,
        clip_model=clip_model,
        clip_processor=clip_proc,
        device=device,
    )

    # Capture
    print("\nCapturing point cloud...")
    points, colors = merger.capture_merged_pointcloud(text_prompt=text_prompt)

    if points is not None:
        # Save
        filename = "output_segmented" if text_prompt else "output_full"
        merger.save_pointcloud(points, colors, filename=filename)

        # Visualize
        print(f"\n✅ Captured {len(points)} points!")
        print(f"Saved to: {filename}.ply")
        print("\nOpening visualizer...")
        merger.visualize_pointcloud(points, colors)
    else:
        print("\n❌ Failed to capture point cloud")

    # Cleanup
    merger.cleanup()


if __name__ == "__main__":
    main()
