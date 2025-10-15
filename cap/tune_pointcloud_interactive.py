#!/usr/bin/env python3
"""
Interactive point cloud alignment tuner with visual manipulation.
Use Open3D's interactive editing to manually align point clouds by dragging and rotating.
Press 'S' to save the alignment, 'Q' to quit without saving.
"""

import numpy as np
import pyrealsense2 as rs
import os
import sys
import copy

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False
    print("ERROR: Open3D is required for this tool. Install with: pip install open3d")
    sys.exit(1)

CAMERA_SERIALS = ["317422074281", "327122079374"]


def configure_realsense_cameras():
    """Configure RealSense cameras for better depth quality"""
    camera_configs = {
        "317422074281": {
            "auto_exposure": 1,
            "gain": 64,
            "laser_power": 240,
            "exposure": None,
        },
        "327122079374": {
            "auto_exposure": 0,
            "exposure": 500,
            "gain": 16,
            "laser_power": 320,
        }
    }

    ctx = rs.context()
    for dev in ctx.query_devices():
        if dev.get_info(rs.camera_info.serial_number) not in CAMERA_SERIALS:
            continue

        serial = dev.get_info(rs.camera_info.serial_number)
        config = camera_configs.get(serial, camera_configs["317422074281"])

        print(f"\nConfiguring camera {serial}...")

        adv = rs.rs400_advanced_mode(dev)
        if not adv.is_enabled():
            adv.toggle_advanced_mode(True)

        stereo = next(
            s for s in dev.query_sensors() if "Stereo" in s.get_info(rs.camera_info.name)
        )
        rgb_sensor = next(
            (s for s in dev.query_sensors() if "RGB" in s.get_info(rs.camera_info.name)),
            None
        )

        stereo.set_option(rs.option.enable_auto_exposure, config["auto_exposure"])
        if config["auto_exposure"] == 0 and config.get("exposure") is not None:
            stereo.set_option(rs.option.exposure, config["exposure"])
        stereo.set_option(rs.option.gain, config["gain"])
        stereo.set_option(rs.option.laser_power, config["laser_power"])

        if rgb_sensor:
            rgb_sensor.set_option(rs.option.enable_auto_white_balance, 1)
            rgb_sensor.set_option(rs.option.enable_auto_exposure, 1)

        print(f"  Camera {serial} configured successfully")


class InteractiveAlignmentTuner:
    """Interactive point cloud alignment with visual editing"""

    def __init__(self, camera_serials, calib_file="transforms/transforms.npy"):
        self.camera_serials = camera_serials
        self.cameras = {}
        self.current_transform = np.eye(4)
        self.final_offset = None

        # Load calibration transforms
        if not os.path.exists(calib_file):
            raise FileNotFoundError(f"Calibration file {calib_file} not found!")

        self.transforms = np.load(calib_file, allow_pickle=True).item()
        print(f"Loaded transforms for cameras: {list(self.transforms.keys())}")

        # Initialize cameras
        for serial in camera_serials:
            self._init_camera(serial)

    def _init_camera(self, serial_number, width=640, height=480, fps=30):
        """Initialize a RealSense camera"""
        pipeline = rs.pipeline()
        config = rs.config()

        config.enable_device(serial_number)
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

        profile = pipeline.start(config)

        # Create post-processing filters
        filters = [
            rs.spatial_filter(),
            rs.temporal_filter(),
            rs.hole_filling_filter(),
        ]

        self.cameras[serial_number] = {
            'pipeline': pipeline,
            'filters': filters
        }

        print(f"Camera {serial_number} initialized")

    def capture_camera_pointcloud(self, serial, max_depth=2.0, min_depth=0.1):
        """Capture point cloud from a single camera"""
        camera_data = self.cameras[serial]
        pipeline = camera_data['pipeline']
        filters = camera_data['filters']

        # Capture frame
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            print(f"Failed to capture from camera {serial}")
            return None

        # Apply filters
        for filter in filters:
            depth_frame = filter.process(depth_frame)

        color_image = np.asanyarray(color_frame.get_data())

        # Create point cloud
        raw_pcd = rs.pointcloud()
        raw_pcd.map_to(color_frame)
        points = raw_pcd.calculate(depth_frame)

        # Extract points and colors
        points_3d = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)
        tex = np.asanyarray(points.get_texture_coordinates()).view(np.float32).reshape(-1, 2)

        # Get colors
        h, w = color_image.shape[:2]
        u = np.clip((tex[:, 0] * w).astype(np.int32), 0, w - 1)
        v = np.clip((tex[:, 1] * h).astype(np.int32), 0, h - 1)
        colors = color_image[v, u][:, ::-1] / 255.0  # BGR to RGB

        # Filter valid points
        valid_mask = (
            (points_3d[:, 2] > min_depth) &
            (points_3d[:, 2] < max_depth) &
            np.isfinite(points_3d).all(axis=1)
        )

        valid_points = points_3d[valid_mask]
        valid_colors = colors[valid_mask]

        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(valid_points)
        pcd.colors = o3d.utility.Vector3dVector(valid_colors)

        return pcd

    def transform_pointcloud(self, pcd, transform):
        """Apply transformation to point cloud"""
        pcd_copy = o3d.geometry.PointCloud(pcd)

        # Ensure transform is 4x4
        if transform.shape == (3, 4):
            transform_4x4 = np.vstack([transform, [0, 0, 0, 1]])
        else:
            transform_4x4 = transform

        pcd_copy.transform(transform_4x4)
        return pcd_copy

    def extract_transform_params(self, transform):
        """Extract translation and rotation from 4x4 transform matrix"""
        # Extract translation
        tx, ty, tz = transform[:3, 3]

        # Extract rotation (ZYX Euler angles)
        R = transform[:3, :3]

        # Calculate Euler angles (in radians)
        sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)

        singular = sy < 1e-6

        if not singular:
            rx = np.arctan2(R[2, 1], R[2, 2])
            ry = np.arctan2(-R[2, 0], sy)
            rz = np.arctan2(R[1, 0], R[0, 0])
        else:
            rx = np.arctan2(-R[1, 2], R[1, 1])
            ry = np.arctan2(-R[2, 0], sy)
            rz = 0

        # Convert to degrees
        rx_deg = np.degrees(rx)
        ry_deg = np.degrees(ry)
        rz_deg = np.degrees(rz)

        return {
            'tx': tx, 'ty': ty, 'tz': tz,
            'rx': rx_deg, 'ry': ry_deg, 'rz': rz_deg
        }

    def run_interactive_alignment(self):
        """Run interactive alignment with manual editing"""
        print("\n" + "="*70)
        print("Interactive Point Cloud Alignment - Visual Editing Mode")
        print("="*70)
        print("\nCapturing point clouds from both cameras...")

        import time
        time.sleep(1)

        # Capture from both cameras
        pcd1 = self.capture_camera_pointcloud(self.camera_serials[0])
        pcd2 = self.capture_camera_pointcloud(self.camera_serials[1])

        if pcd1 is None or pcd2 is None:
            print("Failed to capture point clouds!")
            return

        print(f"\nCamera 1 ({self.camera_serials[0]}): {len(pcd1.points)} points")
        print(f"Camera 2 ({self.camera_serials[1]}): {len(pcd2.points)} points")

        # Apply calibration transforms to robot frame
        tcr1 = self.transforms[self.camera_serials[0]]["tcr"]
        tcr2 = self.transforms[self.camera_serials[1]]["tcr"]

        pcd1_robot = self.transform_pointcloud(pcd1, tcr1)
        pcd2_robot = self.transform_pointcloud(pcd2, tcr2)

        print("\n" + "="*70)
        print("Interactive Editing Instructions")
        print("="*70)
        print("\n✨ Opening interactive alignment editor...")
        print("\n" + "="*70)

        # Create visualization
        vis = o3d.visualization.VisualizerWithEditing()
        vis.create_window(window_name="Point Cloud Alignment - Edit Camera 2", width=1280, height=720)

        # Add camera 1 (base - won't be editable)
        vis.add_geometry(pcd1_robot)

        # Add camera 2 (editable)
        vis.add_geometry(pcd2_robot)

        # Add coordinate frame
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        vis.add_geometry(coord_frame)

        print("\n✅ Visualization opened!")
        print("📝 Tip: Click on the Camera 2 point cloud to select it, then use the")
        print("        transformation widget (arrows and arcs) to adjust alignment.")
        print("\nActually, let me use the manual registration mode instead...")
        vis.destroy_window()

        # Use manual registration which is better for this
        self.run_manual_registration(pcd1_robot, pcd2_robot)

    def run_manual_registration(self, pcd1_robot, pcd2_robot_original):
        """Use Open3D's pick points registration for manual alignment"""

        print("\n" + "="*70)
        print("Manual Alignment Mode - Using Transformation Editing")
        print("="*70)

        # Create a copy we can transform
        pcd2_robot = copy.deepcopy(pcd2_robot_original)

        # Create combined for visualization
        pcd1_display = copy.deepcopy(pcd1_robot)
        pcd2_display = copy.deepcopy(pcd2_robot)

        # Add slight color tint to distinguish them
        colors1 = np.asarray(pcd1_display.colors)
        colors1 = colors1 * 0.8 + np.array([0.2, 0.0, 0.0])  # Slight red tint
        pcd1_display.colors = o3d.utility.Vector3dVector(colors1)

        colors2 = np.asarray(pcd2_display.colors)
        colors2 = colors2 * 0.8 + np.array([0.0, 0.2, 0.0])  # Slight green tint
        pcd2_display.colors = o3d.utility.Vector3dVector(colors2)

        # Now use the edit mode
        print("\n🎮 Opening interactive editing mode...")
        print("   Camera 1 (base) and Camera 2 (adjustable) with original RGB colors")

        # Use ICP with manual review
        result = self.interactive_transform_with_widget(pcd1_robot, pcd2_robot_original)

        if result is not None:
            self.final_offset = result
            self.save_manual_offset(result)

    def interactive_transform_with_widget(self, pcd1, pcd2):
        """Interactive transformation with live visual feedback using keyboard"""

        # Parameters for adjustment
        trans_step = 0.001  # 1mm
        rot_step = 0.5      # 0.5 degrees

        print("\n" + "="*70)
        print("Real-Time Visual Alignment Editor")
        print("="*70)
        print("\n🎮 Controls (in the 3D viewer window):")
        print("\n  ARROW KEYS - Translation:")
        print("    ↑/↓  - Move along X axis (forward/back)")
        print("    ←/→  - Move along Y axis (left/right)")
        print("    PgUp/PgDn - Move along Z axis (up/down)")
        print("\n  W/A/S/D - Rotation:")
        print("    W/S  - Rotate around X axis (pitch)")
        print("    A/D  - Rotate around Y axis (yaw)")
        print("    Q/E  - Rotate around Z axis (roll)")
        print("\n  Other:")
        print("    +/-  - Increase/decrease step size (10x)")
        print("    R    - Reset to original position")
        print("    ESC  - Save and exit")
        print("    X    - Exit without saving")
        print("="*70)
        print("\n✨ The visualization will update in REAL-TIME as you press keys!")
        print("📍 Red coordinate frame = Robot origin")

        offset_params = {'tx': 0, 'ty': 0, 'tz': 0, 'rx': 0, 'ry': 0, 'rz': 0}

        def create_transform_from_params(params):
            tx, ty, tz = params['tx'], params['ty'], params['tz']
            rx, ry, rz = params['rx'], params['ry'], params['rz']

            rx_rad = np.radians(rx)
            ry_rad = np.radians(ry)
            rz_rad = np.radians(rz)

            Rx = np.array([
                [1, 0, 0],
                [0, np.cos(rx_rad), -np.sin(rx_rad)],
                [0, np.sin(rx_rad), np.cos(rx_rad)]
            ])

            Ry = np.array([
                [np.cos(ry_rad), 0, np.sin(ry_rad)],
                [0, 1, 0],
                [-np.sin(ry_rad), 0, np.cos(ry_rad)]
            ])

            Rz = np.array([
                [np.cos(rz_rad), -np.sin(rz_rad), 0],
                [np.sin(rz_rad), np.cos(rz_rad), 0],
                [0, 0, 1]
            ])

            R = Rz @ Ry @ Rx
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = [tx, ty, tz]
            return T

        # Create visualization with keyboard callback
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window(window_name="Point Cloud Alignment - Live Editor", width=1280, height=720)

        # Add geometries
        pcd1_vis = copy.deepcopy(pcd1)
        pcd2_vis = copy.deepcopy(pcd2)
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

        vis.add_geometry(pcd1_vis)
        vis.add_geometry(pcd2_vis)
        vis.add_geometry(coord_frame)

        # State to track if we should save
        should_save = [False]
        should_exit = [False]

        def update_visualization():
            """Update pcd2 with current transformation"""
            transform = create_transform_from_params(offset_params)
            pcd2_new = copy.deepcopy(pcd2)
            pcd2_new.transform(transform)

            pcd2_vis.points = pcd2_new.points
            pcd2_vis.colors = pcd2_new.colors
            vis.update_geometry(pcd2_vis)

            # Print status
            print(f"\r✏️  tx={offset_params['tx']:.4f}m, ty={offset_params['ty']:.4f}m, "
                  f"tz={offset_params['tz']:.4f}m, rx={offset_params['rx']:.2f}°, "
                  f"ry={offset_params['ry']:.2f}°, rz={offset_params['rz']:.2f}° "
                  f"[steps: {trans_step*1000:.1f}mm, {rot_step:.1f}°]", end='', flush=True)

        def make_callback(param, delta):
            def callback(vis):
                offset_params[param] += delta
                update_visualization()
                return False
            return callback

        def make_step_callback(multiplier):
            def callback(vis):
                nonlocal trans_step, rot_step
                trans_step *= multiplier
                rot_step *= multiplier
                print(f"\n🔧 Step size: {trans_step*1000:.1f}mm, {rot_step:.1f}°")
                return False
            return callback

        def reset_callback(vis):
            offset_params.update({'tx': 0, 'ty': 0, 'tz': 0, 'rx': 0, 'ry': 0, 'rz': 0})
            update_visualization()
            print("\n🔄 Reset to identity")
            return False

        def save_callback(vis):
            should_save[0] = True
            print("\n\n💾 Saving...")
            vis.close()
            return False

        def exit_callback(vis):
            should_exit[0] = True
            print("\n\n❌ Exiting without saving...")
            vis.close()
            return False

        # Register arrow keys for translation
        vis.register_key_callback(265, make_callback('tx', trans_step))    # Up arrow
        vis.register_key_callback(264, make_callback('tx', -trans_step))   # Down arrow
        vis.register_key_callback(263, make_callback('ty', trans_step))    # Left arrow
        vis.register_key_callback(262, make_callback('ty', -trans_step))   # Right arrow
        vis.register_key_callback(266, make_callback('tz', trans_step))    # Page Up
        vis.register_key_callback(267, make_callback('tz', -trans_step))   # Page Down

        # Register WASD for rotation
        vis.register_key_callback(ord('W'), make_callback('rx', rot_step))
        vis.register_key_callback(ord('S'), make_callback('rx', -rot_step))
        vis.register_key_callback(ord('A'), make_callback('ry', rot_step))
        vis.register_key_callback(ord('D'), make_callback('ry', -rot_step))
        vis.register_key_callback(ord('Q'), make_callback('rz', rot_step))
        vis.register_key_callback(ord('E'), make_callback('rz', -rot_step))

        # Register step size adjustment
        vis.register_key_callback(ord('+'), make_step_callback(10.0))
        vis.register_key_callback(ord('='), make_step_callback(10.0))  # Also accept =
        vis.register_key_callback(ord('-'), make_step_callback(0.1))

        # Register reset and save
        vis.register_key_callback(ord('R'), reset_callback)
        vis.register_key_callback(256, save_callback)  # ESC key
        vis.register_key_callback(ord('X'), exit_callback)

        print("\n✅ Visualization ready! Start adjusting with keyboard...")
        print("📍 Initial view:")
        update_visualization()

        # Run the visualizer
        vis.run()
        vis.destroy_window()

        print("\n")  # New line after status updates

        if should_exit[0]:
            return None
        elif should_save[0]:
            return create_transform_from_params(offset_params)
        else:
            # Window was closed normally, ask what to do
            save = input("\n💾 Save the alignment? (y/n): ").strip().lower()
            if save == 'y':
                return create_transform_from_params(offset_params)
            else:
                return None

    def save_manual_offset(self, transform):
        """Save the manual offset transformation"""
        params = self.extract_transform_params(transform)

        save_path = "transforms/manual_offset.npy"
        os.makedirs("transforms", exist_ok=True)

        save_data = {
            'transform': transform,
            'params': params,
            'camera_serials': self.camera_serials,
            'description': 'Manual offset for fine-tuning point cloud alignment'
        }

        np.save(save_path, save_data)
        print(f"\n✅ Manual offset saved to {save_path}")
        print(f"📊 Parameters:")
        print(f"   Translation: tx={params['tx']:.4f}m, ty={params['ty']:.4f}m, tz={params['tz']:.4f}m")
        print(f"   Rotation: rx={params['rx']:.2f}°, ry={params['ry']:.2f}°, rz={params['rz']:.2f}°")
        print("\n🎯 This offset will be automatically applied when you run:")
        print("   - segment_pc.py")
        print("   - pc.py")

    def cleanup(self):
        """Stop all cameras"""
        for camera_data in self.cameras.values():
            camera_data['pipeline'].stop()
        print("\nAll cameras stopped")


def main():
    """Main function"""
    CAMERA_SERIALS = ["317422074281", "327122079374"]
    CALIB_FILE = "./transforms/transforms.npy"

    # Configure cameras
    configure_realsense_cameras()

    try:
        # Initialize tuner
        tuner = InteractiveAlignmentTuner(
            camera_serials=CAMERA_SERIALS,
            calib_file=CALIB_FILE
        )

        # Run interactive alignment
        tuner.run_interactive_alignment()

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        if 'tuner' in locals():
            tuner.cleanup()


if __name__ == "__main__":
    main()
