#!/usr/bin/env python3
"""
Simple debug script to visualize point cloud from a single RealSense camera.
No transformations, just raw camera output.
"""

import numpy as np
import pyrealsense2 as rs
import argparse

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False
    print("ERROR: Open3D not available. Please install: pip install open3d")
    exit(1)


def configure_camera(serial_number):
    """Configure a single RealSense camera for better depth quality"""
    ctx = rs.context()
    dev = None

    for device in ctx.query_devices():
        if device.get_info(rs.camera_info.serial_number) == serial_number:
            dev = device
            break

    if dev is None:
        print(f"Camera {serial_number} not found!")
        print("Available cameras:")
        for device in ctx.query_devices():
            print(f"  - {device.get_info(rs.camera_info.serial_number)}")
        return

    print(f"\nConfiguring camera {serial_number}...")

    # Enable Advanced Mode
    adv = rs.rs400_advanced_mode(dev)
    if not adv.is_enabled():
        adv.toggle_advanced_mode(True)
        print(f"  Advanced mode enabled")

    # Find the Stereo Module (depth) sensor
    stereo = next(
        s for s in dev.query_sensors() if "Stereo" in s.get_info(rs.camera_info.name)
    )

    # Find the RGB sensor
    rgb_sensor = next(
        (s for s in dev.query_sensors() if "RGB" in s.get_info(rs.camera_info.name)),
        None
    )

    # === Depth/Stereo Module Configuration ===
    stereo.set_option(rs.option.enable_auto_exposure, 0)

    # stereo.set_option(rs.option.enable_auto_exposure, 1)

    stereo.set_option(rs.option.exposure, 500)

    stereo.set_option(rs.option.gain, 16)

    stereo.set_option(rs.option.laser_power, 320)

    # Enable auto white balance for RGB sensor
    if rgb_sensor:
        rgb_sensor.set_option(rs.option.enable_auto_white_balance, 1)
        print(f"  RGB auto white balance: ON")

        rgb_sensor.set_option(rs.option.enable_auto_exposure, 1)
        print(f"  RGB auto exposure: ON")

    print(f"  Camera {serial_number} configured successfully\n")


def capture_and_visualize(serial_number, use_filters=True, max_depth=2.0, min_depth=0.1):
    """Capture and visualize point cloud from a single camera"""

    # Configure camera first
    configure_camera(serial_number)

    # Initialize pipeline
    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_device(serial_number)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    print(f"Starting camera {serial_number}...")
    profile = pipeline.start(config)

    # Create post-processing filters
    filters = None
    if use_filters:
        print("Setting up post-processing filters...")
        decimation = rs.decimation_filter()
        decimation.set_option(rs.option.filter_magnitude, 2)

        spatial = rs.spatial_filter()
        spatial.set_option(rs.option.filter_magnitude, 2)
        spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
        spatial.set_option(rs.option.filter_smooth_delta, 20)

        temporal = rs.temporal_filter()
        temporal.set_option(rs.option.filter_smooth_alpha, 0.4)
        temporal.set_option(rs.option.filter_smooth_delta, 20)

        hole_filling = rs.hole_filling_filter()
        hole_filling.set_option(rs.option.holes_fill, 1)

        filters = {
            'decimation': decimation,
            'spatial': spatial,
            'temporal': temporal,
            'hole_filling': hole_filling
        }
        print("  - Decimation filter: enabled")
        print("  - Spatial filter: enabled")
        print("  - Temporal filter: enabled")
        print("  - Hole filling filter: enabled")
    else:
        print("Filters disabled - showing raw depth data")

    # Wait for camera to stabilize
    print("\nWaiting for camera to stabilize...")
    import time
    time.sleep(2)

    # Warm up frames (let temporal filter build history)
    if use_filters:
        print("Warming up filters (capturing 30 frames)...")
        for i in range(30):
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            if filters:
                depth_frame = filters['decimation'].process(depth_frame)
                depth_frame = filters['spatial'].process(depth_frame)
                depth_frame = filters['temporal'].process(depth_frame)
                depth_frame = filters['hole_filling'].process(depth_frame)

    print(f"\nCapturing point cloud from camera {serial_number}...")

    # Capture frame
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()

    if not depth_frame or not color_frame:
        print("Failed to capture frames!")
        pipeline.stop()
        return

    # Apply filters if enabled
    if use_filters and filters:
        depth_frame = filters['decimation'].process(depth_frame)
        depth_frame = filters['spatial'].process(depth_frame)
        depth_frame = filters['temporal'].process(depth_frame)
        depth_frame = filters['hole_filling'].process(depth_frame)

    color_image = np.asanyarray(color_frame.get_data())

    # Create point cloud
    pc = rs.pointcloud()
    pc.map_to(color_frame)
    points = pc.calculate(depth_frame)

    # Extract points and colors
    vtx = np.asanyarray(points.get_vertices())
    tex = np.asanyarray(points.get_texture_coordinates())

    points_3d = np.column_stack((vtx["f0"], vtx["f1"], vtx["f2"]))

    # Get colors
    h, w = color_image.shape[:2]
    u = np.clip((tex["f0"] * w).astype(int), 0, w - 1)
    v = np.clip((tex["f1"] * h).astype(int), 0, h - 1)

    colors = color_image[v, u] / 255.0  # Normalize to [0,1]
    colors = colors[:, [2, 1, 0]]  # BGR to RGB

    # Filter valid points
    valid_mask = (
        (points_3d[:, 2] > min_depth)
        & (points_3d[:, 2] < max_depth)
        & ~np.isnan(points_3d).any(axis=1)
        & ~np.isinf(points_3d).any(axis=1)
    )

    valid_points = points_3d[valid_mask]
    valid_colors = colors[valid_mask]

    print(f"\nPoint cloud statistics:")
    print(f"  Total valid points: {len(valid_points)}")
    print(f"  Camera frame bounds (meters):")
    print(f"    X: {valid_points[:,0].min():.3f} to {valid_points[:,0].max():.3f}")
    print(f"    Y: {valid_points[:,1].min():.3f} to {valid_points[:,1].max():.3f}")
    print(f"    Z: {valid_points[:,2].min():.3f} to {valid_points[:,2].max():.3f}")

    # Stop camera
    pipeline.stop()

    # Visualize
    print(f"\nVisualizing point cloud...")
    print("  Camera coordinate frame:")
    print("    X (Red): Right")
    print("    Y (Green): Down")
    print("    Z (Blue): Forward (into the scene)")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(valid_points)
    pcd.colors = o3d.utility.Vector3dVector(valid_colors)

    # Add coordinate frame at camera origin
    camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

    filter_status = "WITH filters" if use_filters else "RAW (no filters)"
    window_title = f"Camera {serial_number} - {filter_status}"

    o3d.visualization.draw_geometries(
        [pcd, camera_frame],
        window_name=window_title,
        width=1280,
        height=720
    )

    print("\nDone!")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize point cloud from a single RealSense camera (raw, no transformations)"
    )
    parser.add_argument(
        "--serial",
        type=str,
        default="327122079374",
        help="Camera serial number (default: 327122079374)"
    )
    parser.add_argument(
        "--no-filters",
        action="store_true",
        help="Disable post-processing filters (show raw depth)"
    )
    parser.add_argument(
        "--max-depth",
        type=float,
        default=2.0,
        help="Maximum depth in meters (default: 2.0)"
    )
    parser.add_argument(
        "--min-depth",
        type=float,
        default=0.1,
        help="Minimum depth in meters (default: 0.1)"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("RealSense Single Camera Debug Visualizer")
    print("=" * 60)
    print(f"Camera serial: {args.serial}")
    print(f"Filters: {'DISABLED' if args.no_filters else 'ENABLED'}")
    print(f"Depth range: {args.min_depth}m to {args.max_depth}m")
    print("=" * 60)

    try:
        capture_and_visualize(
            serial_number=args.serial,
            use_filters=not args.no_filters,
            max_depth=args.max_depth,
            min_depth=args.min_depth
        )
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
