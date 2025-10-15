#!/usr/bin/env python3
"""
Interactive tool to manually tune point cloud alignment between two cameras.
Allows adjusting XYZ translation and rotation to fine-tune alignment after applying transforms.npy.
Saves the manual offset to transforms/manual_offset.npy for use in other scripts.
"""

import numpy as np
import pyrealsense2 as rs
import os
import sys

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
            print(f"  Advanced mode enabled")

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


class PointCloudTuner:
    """Interactive point cloud alignment tuner"""

    def __init__(self, camera_serials, calib_file="transforms/transforms.npy"):
        self.camera_serials = camera_serials
        self.cameras = {}
        self.manual_offset = self._identity_transform()

        # Load calibration transforms
        if not os.path.exists(calib_file):
            raise FileNotFoundError(f"Calibration file {calib_file} not found!")

        self.transforms = np.load(calib_file, allow_pickle=True).item()
        print(f"Loaded transforms for cameras: {list(self.transforms.keys())}")

        # Initialize cameras
        for serial in camera_serials:
            self._init_camera(serial)

    def _identity_transform(self):
        """Create identity transformation matrix"""
        return np.eye(4)

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
            # Convert 3x4 to 4x4 by adding [0, 0, 0, 1] row
            transform_4x4 = np.vstack([transform, [0, 0, 0, 1]])
        else:
            transform_4x4 = transform

        pcd_copy.transform(transform_4x4)
        return pcd_copy

    def create_offset_transform(self, tx=0, ty=0, tz=0, rx=0, ry=0, rz=0):
        """
        Create a transformation matrix from translation and rotation parameters.

        Args:
            tx, ty, tz: Translation in meters
            rx, ry, rz: Rotation in degrees (Euler angles)

        Returns:
            4x4 transformation matrix
        """
        # Convert degrees to radians
        rx_rad = np.radians(rx)
        ry_rad = np.radians(ry)
        rz_rad = np.radians(rz)

        # Rotation matrices
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

        # Combined rotation (ZYX order)
        R = Rz @ Ry @ Rx

        # Create 4x4 transformation matrix
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [tx, ty, tz]

        return T

    def interactive_tune(self):
        """Interactive tuning interface"""
        print("\n" + "="*60)
        print("Interactive Point Cloud Alignment Tuner")
        print("="*60)
        print("\nCapturing point clouds from both cameras...")

        import time
        time.sleep(1)  # Let cameras stabilize

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

        # Keep original RGB colors for better visualization

        print("\n" + "="*60)
        print("Tuning Interface")
        print("="*60)
        print("\nControls:")
        print("  Camera 1 (base): " + self.camera_serials[0])
        print("  Camera 2 (will be adjusted): " + self.camera_serials[1])
        print("  Both point clouds shown with original RGB colors")
        print("\nAdjustment parameters (applied to Camera 2):")
        print("  Translation: tx, ty, tz (meters)")
        print("  Rotation: rx, ry, rz (degrees)")
        print("\nCommands:")
        print("  view        - Show current alignment")
        print("  set <param> <value> - Set parameter (e.g., 'set tx 0.01')")
        print("  adjust <param> <delta> - Adjust parameter (e.g., 'adjust tx 0.001')")
        print("  reset       - Reset all adjustments")
        print("  save        - Save current offset and exit")
        print("  quit        - Exit without saving")
        print("="*60)

        # Current offset parameters
        offset_params = {'tx': 0, 'ty': 0, 'tz': 0, 'rx': 0, 'ry': 0, 'rz': 0}

        # Initial view
        print("\nShowing initial alignment (close window to continue)...")
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        o3d.visualization.draw_geometries(
            [pcd1_robot, pcd2_robot, coord_frame],
            window_name="Initial Alignment - Both cameras with RGB colors"
        )

        while True:
            print(f"\nCurrent offset: tx={offset_params['tx']:.4f}, ty={offset_params['ty']:.4f}, "
                  f"tz={offset_params['tz']:.4f}, rx={offset_params['rx']:.2f}, "
                  f"ry={offset_params['ry']:.2f}, rz={offset_params['rz']:.2f}")

            cmd = input("\n> ").strip().lower()

            if cmd == 'quit':
                print("Exiting without saving.")
                break

            elif cmd == 'view':
                # Apply current offset to camera 2
                offset_tf = self.create_offset_transform(**offset_params)
                pcd2_adjusted = self.transform_pointcloud(pcd2_robot, offset_tf)

                print("Showing adjusted alignment (close window to continue)...")
                o3d.visualization.draw_geometries(
                    [pcd1_robot, pcd2_adjusted, coord_frame],
                    window_name="Adjusted Alignment - Both cameras with RGB colors"
                )

            elif cmd == 'reset':
                offset_params = {'tx': 0, 'ty': 0, 'tz': 0, 'rx': 0, 'ry': 0, 'rz': 0}
                print("Reset all parameters to zero.")

            elif cmd.startswith('set '):
                try:
                    parts = cmd.split()
                    param = parts[1]
                    value = float(parts[2])
                    if param in offset_params:
                        offset_params[param] = value
                        print(f"Set {param} = {value}")
                    else:
                        print(f"Unknown parameter: {param}")
                except (IndexError, ValueError) as e:
                    print(f"Invalid command. Usage: set <param> <value>")

            elif cmd.startswith('adjust '):
                try:
                    parts = cmd.split()
                    param = parts[1]
                    delta = float(parts[2])
                    if param in offset_params:
                        offset_params[param] += delta
                        print(f"Adjusted {param} by {delta}, new value: {offset_params[param]}")
                    else:
                        print(f"Unknown parameter: {param}")
                except (IndexError, ValueError) as e:
                    print(f"Invalid command. Usage: adjust <param> <delta>")

            elif cmd == 'save':
                # Create final transformation
                self.manual_offset = self.create_offset_transform(**offset_params)

                # Save to file
                save_path = "transforms/manual_offset.npy"
                os.makedirs("transforms", exist_ok=True)

                # Save with metadata
                save_data = {
                    'transform': self.manual_offset,
                    'params': offset_params,
                    'camera_serials': self.camera_serials,
                    'description': 'Manual offset for fine-tuning point cloud alignment'
                }

                np.save(save_path, save_data)
                print(f"\nManual offset saved to {save_path}")
                print(f"Parameters: {offset_params}")
                print("\nThis offset will be automatically applied after transforms.npy")
                print("when you run segment_pc.py or other scripts.")
                break

            else:
                print("Unknown command. Type 'view', 'set', 'adjust', 'reset', 'save', or 'quit'.")

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
        tuner = PointCloudTuner(
            camera_serials=CAMERA_SERIALS,
            calib_file=CALIB_FILE
        )

        # Run interactive tuning
        tuner.interactive_tune()

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        if 'tuner' in locals():
            tuner.cleanup()


if __name__ == "__main__":
    main()
