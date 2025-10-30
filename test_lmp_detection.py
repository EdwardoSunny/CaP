#!/usr/bin/env python3
"""
Test script for LMP object detection with visualization.

This script:
1. Uses the LMP segmentation to detect an object
2. Calculates the top surface center
3. Visualizes the point cloud with the center marked

Usage:
    python test_lmp_detection.py "orange disinfecting wipes"
    python test_lmp_detection.py "cup"
"""

import sys
import numpy as np
from pathlib import Path
import torch

# Import the segmentation API
from cap.segment_pc import RobotFrameMerger, load_segmentation_models

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False
    print("Open3D not available. Cannot visualize.")
    sys.exit(1)


def calculate_top_surface_center(points, top_percentile=10):
    """
    Calculate the center of the top surface of an object.
    Same logic as LMP's get_object_center().

    Args:
        points: Nx3 numpy array of 3D points
        top_percentile: Percentage of top points to consider

    Returns:
        center: [x, y, z] center position on top surface
    """
    if points is None or len(points) == 0:
        return None

    # Get the top surface points (top percentile of Z values)
    z_threshold = np.percentile(points[:, 2], 100 - top_percentile)
    top_points = points[points[:, 2] >= z_threshold]

    if len(top_points) == 0:
        print("Warning: No top surface points found, using simple centroid")
        center = np.mean(points, axis=0)
    else:
        # Get XY center of top surface
        center_xy = np.mean(top_points[:, :2], axis=0)
        # Z is the highest point
        center_z = np.max(points[:, 2])
        center = np.array([center_xy[0], center_xy[1], center_z])

    print(f"\n📍 Object Center Calculation:")
    print(f"   Total points: {len(points)}")
    print(f"   Top surface points (top {top_percentile}%): {len(top_points)}")
    print(f"   Z threshold: {z_threshold:.3f} m")
    print(f"   Center position: [{center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}]")

    return center


def visualize_with_center(points, colors, center):
    """
    Visualize point cloud with center marked as a red sphere.

    Args:
        points: Nx3 numpy array of 3D points
        colors: Nx3 numpy array of RGB colors
        center: [x, y, z] center position
    """
    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Create coordinate frame at robot origin
    robot_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

    # Create red sphere at center position
    center_marker = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
    center_marker.paint_uniform_color([1.0, 0.0, 0.0])  # Red
    center_marker.translate(center)

    # Create small coordinate frame at center
    center_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
    center_frame.translate(center)

    print("\n🎨 Visualization Guide:")
    print("   - WHITE/RGB points: Segmented object")
    print("   - RED sphere: Calculated grasp center")
    print("   - Large RGB axes: Robot origin")
    print("   - Small RGB axes: Object center")
    print("\nClose the window to exit.")

    # Visualize
    o3d.visualization.draw_geometries(
        [pcd, robot_frame, center_marker, center_frame],
        window_name=f"Object Detection - Center at [{center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}]",
        width=1280,
        height=720,
    )


def main():
    # Get text prompt from command line
    if len(sys.argv) < 2:
        print("Usage: python test_lmp_detection.py '<object name>'")
        print("\nExamples:")
        print("  python test_lmp_detection.py 'orange disinfecting wipes'")
        print("  python test_lmp_detection.py 'cup'")
        print("  python test_lmp_detection.py 'bottle'")
        sys.exit(1)

    text_prompt = sys.argv[1]

    # Configuration
    camera_serials = ["327122079374", "317422074281"]
    calib_file = Path("transforms/transforms.npy")
    sam2_checkpoint = Path("ckpt/sam2.1_hiera_large.pt")
    sam2_config = Path("sam2.1/sam2.1_hiera_l.yaml")
    clip_model_name = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"

    print("="*60)
    print(f"🔍 Testing LMP Object Detection")
    print("="*60)
    print(f"Object to find: '{text_prompt}'")
    print(f"Camera serials: {camera_serials}")
    print(f"Calibration file: {calib_file}")

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
        print("   ✅ Models loaded!")

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

        # Calculate top surface center (same as LMP does)
        print("\n🎯 Calculating grasp center...")
        center = calculate_top_surface_center(points, top_percentile=10)

        # Save point cloud
        print("\n💾 Saving point cloud...")
        merger.save_pointcloud(points, colors, filename=f"detected_{text_prompt.replace(' ', '_')}")
        print(f"   ✅ Saved to: detected_{text_prompt.replace(' ', '_')}.ply")

        # Visualize
        print("\n👁️  Opening visualization...")
        visualize_with_center(points, colors, center)

        print("\n✅ Test complete!")
        print(f"\n📊 Summary:")
        print(f"   Object: '{text_prompt}'")
        print(f"   Points: {len(points)}")
        print(f"   Grasp Center: [{center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}] meters")
        print(f"   (X, Y, Z in robot base frame)")

        # Cleanup
        merger.cleanup()

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        if 'merger' in locals():
            merger.cleanup()
        sys.exit(1)


if __name__ == "__main__":
    main()
