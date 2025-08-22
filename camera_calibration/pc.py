#!/usr/bin/env python3
"""
Simple script to capture from both cameras, transform to robot frame, and merge.
This is the clean, production-ready version for getting a single merged point cloud.
"""

import numpy as np
import pyrealsense2 as rs
import os

CAMERA_SERIALS = ["317422074281", "317422075456"]

ctx = rs.context()
for dev in ctx.query_devices():
    if dev.get_info(rs.camera_info.serial_number) not in CAMERA_SERIALS:
        continue

    # Enable Advanced Mode per device
    adv = rs.rs400_advanced_mode(dev)
    if not adv.is_enabled():
        adv.toggle_advanced_mode(True)

    # Find the Stereo Module sensor
    stereo = next(
        s for s in dev.query_sensors() if "Stereo" in s.get_info(rs.camera_info.name)
    )

    # Minimal manual depth config
    stereo.set_option(rs.option.enable_auto_exposure, 0)
    # stereo.set_option(rs.option.exposure, 1000)     # µs
    stereo.set_option(rs.option.gain, 16)
    stereo.set_option(rs.option.laser_power, 70)  # 0–360

try:
    import open3d as o3d

    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False
    print("Open3D not available. Will save as text file.")


class RobotFrameMerger:
    """Merge point clouds from multiple cameras in robot coordinate frame"""

    def __init__(
        self,
        camera_serials,
        calib_file="calib/transforms.npy",
        icp_file="calib/icp_tf.npy",
        calib_units="mm",
        point_cloud_units="m",
    ):
        self.camera_serials = camera_serials
        self.cameras = {}

        # Unit conversion setup
        self.calib_units = calib_units
        self.point_cloud_units = point_cloud_units
        self.unit_scale = self._get_unit_scale(calib_units, point_cloud_units)

        print(
            f"Unit conversion: {calib_units} -> {point_cloud_units} (scale: {self.unit_scale})"
        )

        # Load calibration transforms
        if not os.path.exists(calib_file):
            raise FileNotFoundError(f"Calibration file {calib_file} not found!")

        self.transforms = np.load(calib_file, allow_pickle=True).item()
        print(f"Loaded transforms for cameras: {list(self.transforms.keys())}")

        # Convert calibration transforms to point cloud units
        self._convert_transform_units()

        # Load ICP transforms if available (for fine alignment)
        self.icp_transforms = {}
        if os.path.exists(icp_file):
            self.icp_transforms = np.load(icp_file, allow_pickle=True).item()
            self._convert_icp_transform_units()
            print(
                f"Loaded ICP transforms for cameras: {list(self.icp_transforms.keys())}"
            )

        # Initialize cameras
        for serial in camera_serials:
            self._init_camera(serial)

    def _get_unit_scale(self, from_units, to_units):
        """Calculate scale factor for unit conversion"""
        unit_factors = {
            "mm": 0.001,  # mm to meters
            "m": 1.0,  # meters to meters
            "cm": 0.01,  # cm to meters
            "inch": 0.0254,  # inches to meters
        }

        if from_units not in unit_factors or to_units not in unit_factors:
            raise ValueError(
                f"Unsupported units. Supported: {list(unit_factors.keys())}"
            )

        # Scale factor to convert from_units to to_units
        return unit_factors[from_units] / unit_factors[to_units]

    def _convert_transform_units(self):
        """Convert calibration transforms from calibration units to point cloud units"""
        if self.unit_scale == 1.0:
            return  # No conversion needed

        print(
            f"Converting calibration transforms from {self.calib_units} to {self.point_cloud_units}..."
        )

        for serial, transform_data in self.transforms.items():
            if "tcr" in transform_data:
                # Extract translation part and scale it
                tcr = transform_data["tcr"].copy()
                tcr[:3, 3] *= self.unit_scale  # Scale translation components
                self.transforms[serial]["tcr"] = tcr

                print(f"Camera {serial}: scaled translation by {self.unit_scale}")
                print(f"  New translation: {tcr[:3, 3]}")

    def _convert_icp_transform_units(self):
        """Convert ICP transforms from calibration units to point cloud units"""
        if self.unit_scale == 1.0:
            return  # No conversion needed

        print(
            f"Converting ICP transforms from {self.calib_units} to {self.point_cloud_units}..."
        )

        for serial, icp_tf in self.icp_transforms.items():
            # Scale translation part
            icp_tf[:3, 3] *= self.unit_scale
            print(f"Camera {serial}: scaled ICP translation by {self.unit_scale}")

    def _init_camera(self, serial_number, width=640, height=480, fps=30):
        """Initialize a RealSense camera"""
        pipeline = rs.pipeline()
        config = rs.config()

        config.enable_device(serial_number)
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

        profile = pipeline.start(config)

        print(f"Camera {serial_number} initialized")
        self.cameras[serial_number] = pipeline

    def capture_single_camera(
        self, serial, max_depth=2.0, min_depth=0.1, max_points=100000
    ):
        """Capture point cloud from a single camera"""
        pipeline = self.cameras[serial]

        # Capture frame
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            print(f"Failed to capture from camera {serial}")
            return None, None

        # Create point cloud using RealSense
        pc = rs.pointcloud()
        pc.map_to(color_frame)
        points = pc.calculate(depth_frame)

        # Extract points and colors
        vtx = np.asanyarray(points.get_vertices())
        tex = np.asanyarray(points.get_texture_coordinates())

        points_3d = np.column_stack((vtx["f0"], vtx["f1"], vtx["f2"]))

        # Get colors
        color_image = np.asanyarray(color_frame.get_data())
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

        print(f"Camera {serial}: {len(valid_points)} valid points captured")

        # Downsample if too many points
        if len(valid_points) > max_points:
            indices = np.random.choice(len(valid_points), max_points, replace=False)
            valid_points = valid_points[indices]
            valid_colors = valid_colors[indices]
            print(f"Camera {serial}: downsampled to {len(valid_points)} points")

        return valid_points, valid_colors

    def transform_to_robot_frame(self, points, camera_serial):
        """Transform points from camera coordinates to robot coordinates"""
        if camera_serial not in self.transforms:
            raise ValueError(f"No calibration found for camera {camera_serial}")

        # Apply main calibration transform (camera to robot) - already unit-corrected
        tcr = self.transforms[camera_serial]["tcr"]

        # Convert to homogeneous coordinates
        ones = np.ones((points.shape[0], 1))
        points_homo = np.hstack([points, ones])

        # Transform to robot frame
        points_robot = (tcr @ points_homo.T).T[:, :3]

        # Apply ICP refinement if available - already unit-corrected
        if camera_serial in self.icp_transforms:
            icp_tf = self.icp_transforms[camera_serial]
            ones = np.ones((points_robot.shape[0], 1))
            points_homo = np.hstack([points_robot, ones])
            points_robot = (icp_tf @ points_homo.T).T[:, :3]
            print(f"Camera {camera_serial}: Applied ICP refinement")

        return points_robot

    def crop_workspace(self, points, min_bound=None, max_bound=None):
        """Crop point cloud to workspace bounds (optional)"""
        if min_bound is None:
            min_bound = [0.2, -0.35, 0.10]  # Default workspace bounds in meters
        if max_bound is None:
            max_bound = [0.9, 0.3, 0.5]

        mask = (
            (points[:, 0] > min_bound[0])
            & (points[:, 0] < max_bound[0])
            & (points[:, 1] > min_bound[1])
            & (points[:, 1] < max_bound[1])
            & (points[:, 2] > min_bound[2])
            & (points[:, 2] < max_bound[2])
        )

        return mask

    def capture_merged_pointcloud(
        self, crop_workspace=False, workspace_bounds=None, color_by_camera=False
    ):
        """Capture and merge point clouds from all cameras in robot frame"""
        print(
            f"Capturing merged point cloud in robot frame ({self.point_cloud_units})..."
        )

        all_points_robot = []
        all_colors = []

        # Define distinct colors for each camera (only used if color_by_camera=True)
        camera_colors = [
            [1.0, 0.2, 0.2],  # Red for first camera
            [0.2, 0.8, 0.2],  # Green for second camera
            [0.2, 0.2, 1.0],  # Blue for third camera (if any)
            [1.0, 0.8, 0.2],  # Orange for fourth camera (if any)
        ]

        # Wait for cameras to stabilize
        import time

        time.sleep(1)

        # Process each camera
        for idx, serial in enumerate(self.camera_serials):
            print(f"\nProcessing camera {serial}...")

            # Capture from camera
            points_cam, colors_cam = self.capture_single_camera(serial)

            if points_cam is None:
                print(f"Skipping camera {serial} - capture failed")
                continue

            try:
                # Transform to robot frame
                points_robot = self.transform_to_robot_frame(points_cam, serial)

                print(f"Camera {serial} robot frame bounds ({self.point_cloud_units}):")
                print(
                    f"  X: {points_robot[:,0].min():.3f} to {points_robot[:,0].max():.3f}"
                )
                print(
                    f"  Y: {points_robot[:,1].min():.3f} to {points_robot[:,1].max():.3f}"
                )
                print(
                    f"  Z: {points_robot[:,2].min():.3f} to {points_robot[:,2].max():.3f}"
                )

                # Optional workspace cropping
                if crop_workspace:
                    if workspace_bounds is not None:
                        min_bound = workspace_bounds[:3]
                        max_bound = workspace_bounds[3:]
                    else:
                        min_bound = max_bound = None

                    crop_mask = self.crop_workspace(points_robot, min_bound, max_bound)
                    points_robot = points_robot[crop_mask]
                    colors_cam = colors_cam[crop_mask]
                    print(f"Camera {serial}: {len(points_robot)} points after cropping")

                # Color points by camera if requested (otherwise keep original RGB)
                if color_by_camera:
                    camera_color = camera_colors[idx % len(camera_colors)]
                    colors_cam = np.full((len(points_robot), 3), camera_color)
                    print(
                        f"Camera {serial}: colored with {camera_color} ({'Red' if idx==0 else 'Green' if idx==1 else 'Blue'})"
                    )
                else:
                    print(f"Camera {serial}: keeping original RGB colors")

                all_points_robot.append(points_robot)
                all_colors.append(colors_cam)

            except Exception as e:
                print(f"Failed to transform camera {serial}: {e}")
                continue

        # Merge all point clouds
        if not all_points_robot:
            print("No valid point clouds captured!")
            return None, None, None

        merged_points = np.vstack(all_points_robot)
        merged_colors = np.vstack(all_colors)

        # Also return individual camera data for separate visualization
        camera_data = []
        for i, serial in enumerate(self.camera_serials):
            if i < len(all_points_robot):
                camera_data.append(
                    {
                        "serial": serial,
                        "points": all_points_robot[i],
                        "colors": all_colors[i],
                        "color_name": f"Camera {i+1}",
                    }
                )

        print(f"\n✓ Successfully merged point cloud:")
        print(f"  Total points: {len(merged_points)}")
        print(f"  Robot frame bounds ({self.point_cloud_units}):")
        print(
            f"    X: {merged_points[:,0].min():.3f} to {merged_points[:,0].max():.3f}"
        )
        print(
            f"    Y: {merged_points[:,1].min():.3f} to {merged_points[:,1].max():.3f}"
        )
        print(
            f"    Z: {merged_points[:,2].min():.3f} to {merged_points[:,2].max():.3f}"
        )

        return merged_points, merged_colors, camera_data

    def save_pointcloud(
        self, points, colors, filename="merged_robot_frame", camera_data=None
    ):
        """Save merged point cloud"""
        if points is None:
            print("No point cloud to save")
            return

        # Save as PLY if Open3D available
        if HAS_OPEN3D:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors)

            ply_file = f"{filename}.ply"
            o3d.io.write_point_cloud(ply_file, pcd)
            print(f"Saved: {ply_file}")

        # Always save as text file too
        txt_file = f"{filename}.txt"
        with open(txt_file, "w") as f:
            f.write(
                f"# X Y Z R G B (Robot Frame Coordinates in {self.point_cloud_units})\n"
            )
            for i in range(len(points)):
                f.write(
                    f"{points[i,0]:.6f} {points[i,1]:.6f} {points[i,2]:.6f} "
                    f"{colors[i,0]:.3f} {colors[i,1]:.3f} {colors[i,2]:.3f}\n"
                )
        print(f"Saved: {txt_file}")

        # Save individual camera point clouds if data provided
        if camera_data:
            for cam_info in camera_data:
                serial = cam_info["serial"]
                cam_points = cam_info["points"]
                cam_colors = cam_info["colors"]
                cam_filename = f"{filename}_camera_{serial}"

                if HAS_OPEN3D:
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(cam_points)
                    pcd.colors = o3d.utility.Vector3dVector(cam_colors)
                    o3d.io.write_point_cloud(f"{cam_filename}.ply", pcd)
                    print(f"Saved individual camera: {cam_filename}.ply")

    def visualize_cameras_separately(self, camera_data):
        """Visualize individual camera point clouds"""
        if not HAS_OPEN3D:
            print("Open3D not available for visualization")
            return

        if not camera_data:
            print("No camera data to visualize")
            return

        for cam_info in camera_data:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(cam_info["points"])
            pcd.colors = o3d.utility.Vector3dVector(cam_info["colors"])

            # Add coordinate frame
            robot_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

            window_name = f"Camera {cam_info['serial']} - {cam_info['color_name']} ({self.point_cloud_units})"
            print(f"Showing {window_name}...")
            o3d.visualization.draw_geometries(
                [pcd, robot_frame], window_name=window_name
            )

    def visualize_pointcloud(self, points, colors, camera_data=None):
        """Visualize the merged point cloud"""
        if not HAS_OPEN3D:
            print("Open3D not available for visualization")
            return

        if points is None:
            print("No point cloud to visualize")
            return

        # Create point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        # Add coordinate frame at robot origin
        robot_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

        print(
            f"Visualizing merged point cloud in robot frame ({self.point_cloud_units})..."
        )
        print("- Coordinate frame shows robot origin")
        print("- Red=X, Green=Y, Blue=Z axes")

        # Add text labels if camera data provided
        window_title = f"Merged Point Cloud - Robot Frame ({self.point_cloud_units})"
        if camera_data:
            camera_info = ", ".join(
                [f"{cd['color_name']}={cd['serial'][-4:]}" for cd in camera_data]
            )
            window_title += f" ({camera_info})"

        o3d.visualization.draw_geometries([pcd, robot_frame], window_name=window_title)

    def cleanup(self):
        """Stop all cameras"""
        for pipeline in self.cameras.values():
            pipeline.stop()
        print("All cameras stopped")


def main():
    """Main function - easy to use interface"""

    # Configuration
    CAMERA_SERIALS = ["317422074281", "317422075456"]
    CALIB_FILE = "calib/transforms.npy"  # Use original calibration file (in mm)
    ICP_FILE = "calib/icp_tf.npy"

    # Optional workspace cropping bounds: [min_x, min_y, min_z, max_x, max_y, max_z] in meters
    WORKSPACE_BOUNDS = [0.15, -0.4, 0.08, 1.0, 0.35, 0.6]

    try:
        # Initialize merger with implicit unit conversion
        print("Initializing robot frame merger with automatic unit conversion...")
        merger = RobotFrameMerger(
            camera_serials=CAMERA_SERIALS,
            calib_file=CALIB_FILE,
            icp_file=ICP_FILE,
            calib_units="mm",  # Calibration data is in millimeters
            point_cloud_units="m",  # RealSense outputs meters
        )

        # Capture merged point cloud
        result = merger.capture_merged_pointcloud(
            crop_workspace=False,  # Set to True if you want workspace cropping
            workspace_bounds=WORKSPACE_BOUNDS,
            color_by_camera=False,  # Keep original RGB colors from cameras
        )

        if result is not None and len(result) == 3:
            merged_points, merged_colors, camera_data = result
        else:
            merged_points, merged_colors, camera_data = None, None, None

        if merged_points is not None:
            # Save point cloud
            merger.save_pointcloud(
                points=merged_points,
                colors=merged_colors,
                filename="merged_robot_frame_colored",
                camera_data=camera_data,
            )

            # Show individual cameras first
            print("\nShowing individual cameras...")
            merger.visualize_cameras_separately(camera_data)

            # Then show merged view
            print("\nShowing merged view...")
            merger.visualize_pointcloud(merged_points, merged_colors, camera_data)

            print(f"\n🎉 Success! Merged {len(merged_points)} points in robot frame")
            print("Point cloud saved with original RGB colors from cameras")
            print(
                f"Units automatically converted from {merger.calib_units} to {merger.point_cloud_units}"
            )

        else:
            print("❌ Failed to create merged point cloud")

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()

    finally:
        if "merger" in locals():
            merger.cleanup()


if __name__ == "__main__":
    main()
