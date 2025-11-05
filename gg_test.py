# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import argparse
import sys
import os
from pathlib import Path

import numpy as np
import torch
import trimesh.transformations as tra

from grasp_gen.grasp_server import GraspGenSampler, load_grasp_cfg
from grasp_gen.utils.meshcat_utils import (
    create_visualizer,
    get_color_from_score,
    get_normals_from_mesh,
    make_frame,
    visualize_grasp,
    visualize_mesh,
    visualize_pointcloud,
)
from grasp_gen.utils.point_cloud_utils import point_cloud_outlier_removal

# Import the segmentation API
from cap.segment_pc import RobotFrameMerger, load_segmentation_models


def parse_args():
    parser = argparse.ArgumentParser(
        description="Capture point cloud and visualize grasps after GraspGen inference"
    )
    parser.add_argument(
        "object_name",
        type=str,
        help="Name of the object to detect and grasp (e.g., 'orange cleaning wipes', 'cup')",
    )
    parser.add_argument(
        "--grasp_threshold",
        type=float,
        default=0.8,
        help="Threshold for valid grasps. If -1.0, then the top 100 grasps will be ranked and returned",
    )
    parser.add_argument(
        "--num_grasps",
        type=int,
        default=200,
        help="Number of grasps to generate",
    )
    parser.add_argument(
        "--return_topk",
        action="store_true",
        help="Whether to return only the top k grasps",
    )
    parser.add_argument(
        "--topk_num_grasps",
        type=int,
        default=-1,
        help="Number of top grasps to return when return_topk is True",
    )

    return parser.parse_args()


def process_point_cloud(pc, grasps, grasp_conf):
    """Process point cloud and grasps by centering them."""
    scores = get_color_from_score(grasp_conf, use_255_scale=True)
    print(f"Scores with min {grasp_conf.min():.3f} and max {grasp_conf.max():.3f}")

    # Ensure grasps have correct homogeneous coordinate
    grasps[:, 3, 3] = 1

    # Center point cloud and grasps
    T_subtract_pc_mean = tra.translation_matrix(-pc.mean(axis=0))
    pc_centered = tra.transform_points(pc, T_subtract_pc_mean)
    grasps_centered = np.array(
        [T_subtract_pc_mean @ np.array(g) for g in grasps.tolist()]
    )

    return pc_centered, grasps_centered, scores


if __name__ == "__main__":
    args = parse_args()

    # Hardcoded gripper config
    gripper_config = "GraspGen/GraspGenModels/checkpoints/graspgen_robotiq_2f_140.yml"

    if not os.path.exists(gripper_config):
        raise ValueError(f"Gripper config {gripper_config} does not exist")

    # Handle return_topk logic
    if args.return_topk and args.topk_num_grasps == -1:
        args.topk_num_grasps = 100

    # Get text prompt from parsed args
    text_prompt = args.object_name

    # Configuration for camera capture (same as test_lmp_detection.py)
    camera_serials = ["327122079374", "317422074281"]
    calib_file = Path("transforms/transforms.npy")
    sam2_checkpoint = Path("ckpt/sam2.1_hiera_large.pt")
    sam2_config = Path("sam2.1/sam2.1_hiera_l.yaml")
    clip_model_name = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"

    print("="*60)
    print(f"🔍 GraspGen Test - Capture and Generate Grasps")
    print("="*60)
    print(f"Object to grasp: '{text_prompt}'")
    print(f"Gripper config: {gripper_config}")
    print(f"Camera serials: {camera_serials}")

    try:
        # Load segmentation models
        print("\n📦 Loading AI models...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"   Device: {device}")

        sam_gen, clip_model, clip_proc = load_segmentation_models(
            sam2_checkpoint=sam2_checkpoint,
            sam2_config=sam2_config,
            clip_model_name=clip_model_name,
            device=device
        )
        print("   ✅ Segmentation models loaded!")

        # Create merger with segmentation
        print("\n📷 Initializing cameras...")
        merger = RobotFrameMerger(
            camera_serials=camera_serials,
            calib_file=calib_file,
            max_depth=2.0,
            min_depth=0.1,
            sam_generator=sam_gen,
            clip_model=clip_model,
            clip_processor=clip_proc,
            device=device,
        )
        print("   ✅ Cameras initialized!")

        # Capture with segmentation
        print(f"\n🎯 Detecting '{text_prompt}'...")
        points, colors = merger.capture_merged_pointcloud(text_prompt=text_prompt)

        if points is None or len(points) == 0:
            print("\n❌ No points found! Object not detected.")
            merger.cleanup()
            sys.exit(1)

        print(f"   ✅ Found {len(points)} points!")

        # Normalize colors to [0, 1] range for visualization
        pc_color = colors / 255.0 if colors.max() > 1.0 else colors

        # Load grasp config and initialize GraspGenSampler
        print("\n🤖 Loading GraspGen model...")
        grasp_cfg = load_grasp_cfg(gripper_config)
        gripper_name = grasp_cfg.data.gripper_name
        grasp_sampler = GraspGenSampler(grasp_cfg)
        print(f"   ✅ GraspGen loaded for gripper: {gripper_name}")

        # Create visualizer
        vis = create_visualizer()

        # Center point cloud (don't need grasps for centering in this case)
        T_subtract_pc_mean = tra.translation_matrix(-points.mean(axis=0))
        pc_centered = tra.transform_points(points, T_subtract_pc_mean)

        # Visualize original point cloud
        visualize_pointcloud(vis, "pc", pc_centered, pc_color, size=0.0025)

        # Filter point cloud
        print("\n🔍 Filtering point cloud...")
        pc_filtered, pc_removed = point_cloud_outlier_removal(
            torch.from_numpy(pc_centered)
        )
        pc_filtered = pc_filtered.numpy()
        pc_removed = pc_removed.numpy()
        print(f"   Filtered: {len(pc_filtered)} points")
        print(f"   Removed: {len(pc_removed)} outliers")
        visualize_pointcloud(vis, "pc_removed", pc_removed, [255, 0, 0], size=0.003)

        # Run inference on filtered point cloud
        print(f"\n🎯 Generating grasps (num_grasps={args.num_grasps}, threshold={args.grasp_threshold})...")
        grasps_inferred, grasp_conf_inferred = GraspGenSampler.run_inference(
            pc_filtered,
            grasp_sampler,
            grasp_threshold=args.grasp_threshold,
            num_grasps=args.num_grasps,
            topk_num_grasps=args.topk_num_grasps,
        )

        if len(grasps_inferred) > 0:
            grasp_conf_inferred = grasp_conf_inferred.cpu().numpy()
            grasps_inferred = grasps_inferred.cpu().numpy()
            grasps_inferred[:, 3, 3] = 1
            scores_inferred = get_color_from_score(
                grasp_conf_inferred, use_255_scale=True
            )
            print(
                f"   ✅ Generated {len(grasps_inferred)} grasps!"
            )
            print(
                f"   Scores range: {grasp_conf_inferred.min():.3f} - {grasp_conf_inferred.max():.3f}"
            )

            # Visualize inferred grasps
            print("\n🎨 Visualizing grasps in meshcat...")
            for j, grasp in enumerate(grasps_inferred):
                visualize_grasp(
                    vis,
                    f"grasps_objectpc_filtered/{j:03d}/grasp",
                    grasp,
                    color=scores_inferred[j],
                    gripper_name=gripper_name,
                    linewidth=0.6,
                )

            print("\n✅ Done! Check meshcat visualization.")
            print("   Press Enter to exit...")
            input()

        else:
            print("\n❌ No grasps found from inference!")

        # Cleanup
        merger.cleanup()

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        if 'merger' in locals():
            merger.cleanup()
        sys.exit(1)
