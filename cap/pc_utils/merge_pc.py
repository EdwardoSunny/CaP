#!/usr/bin/env python3
"""
Multi-camera point cloud capture and merger using calibration transforms.
Captures point clouds from all connected RealSense cameras and merges them
using the same calibration data as the robot pointing system.

Requirements:
- Multiple RealSense cameras connected
- Calibration file (calib/transforms.npy) must exist
- Open3D for point cloud processing (pip install open3d)

Usage:
python multicamera_pointcloud.py
"""

import numpy as np
import pyrealsense2 as rs
import open3d as o3d
import os
import time


class MultiCameraPointCloudMerger:
    def __init__(self):
        # Load calibration data
        if not os.path.exists("calib/transforms.npy"):
            raise FileNotFoundError(
                "Calibration file not found. Run calibration first!"
            )

        self.transforms = np.load("calib/transforms.npy", allow_pickle=True).item()
        print(f"Loaded transforms for {len(self.transforms)} cameras")

        # Get connected camera serials
        ctx = rs.context()
        devices = ctx.query_devices()
        self.serials = [dev.get_info(rs.camera_info.serial_number) for dev in devices]
        print(f"Found {len(self.serials)} connected cameras: {self.serials}")

        # Check which cameras have calibration data
        self.calibrated_serials = []
        for serial in self.serials:
            if serial in self.transforms:
                self.calibrated_serials.append(serial)
                print(f"Camera {serial}: Calibrated ✓")
            else:
                print(f"Camera {serial}: No calibration data ✗")

        if not self.calibrated_serials:
            raise ValueError("No cameras with calibration data found!")

        # Create pipelines for calibrated cameras only
        self.pipelines = []
        self.intrinsics = []

        for serial in self.calibrated_serials:
            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_device(serial)
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

            profile = pipeline.start(config)

            # Get camera intrinsics
            depth_stream = profile.get_stream(rs.stream.depth)
            intrinsic = depth_stream.as_video_stream_profile().get_intrinsics()
            self.intrinsics.append(intrinsic)

            self.pipelines.append(pipeline)
            print(f"Initialized pipeline for camera {serial}")

        # Create point cloud object
        self.pc = rs.pointcloud()

    def capture_single_pointcloud(self, pipeline_idx):
        """Capture point cloud from a single camera"""
        pipeline = self.pipelines[pipeline_idx]
        serial = self.calibrated_serials[pipeline_idx]

        # Capture frames
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            print(f"Failed to capture frames from camera {serial}")
            return None

        # Generate point cloud
        self.pc.map_to(color_frame)
        points = self.pc.calculate(depth_frame)

        # Convert to numpy arrays
        vertices = np.asanyarray(points.get_vertices())
        colors = np.asanyarray(points.get_texture_coordinates())

        # Convert structured array to regular array
        points_3d = vertices.view(np.float32).reshape(-1, 3)

        # Get color data
        color_data = np.asanyarray(color_frame.get_data())

        # Filter out invalid points (zeros)
        valid_mask = np.all(points_3d != 0, axis=1)
        points_3d = points_3d[valid_mask]

        # Get corresponding colors
        if len(points_3d) > 0:
            # Sample colors from the color image
            h, w = color_data.shape[:2]
            u_coords = np.clip(
                (colors.view(np.float32).reshape(-1, 2)[valid_mask, 0] * w).astype(int),
                0,
                w - 1,
            )
            v_coords = np.clip(
                (colors.view(np.float32).reshape(-1, 2)[valid_mask, 1] * h).astype(int),
                0,
                h - 1,
            )

            point_colors = color_data[v_coords, u_coords] / 255.0  # Normalize to [0,1]
        else:
            point_colors = np.array([]).reshape(0, 3)

        return points_3d, point_colors

    def transform_pointcloud(self, points_3d, serial):
        """Transform point cloud from camera coordinates to robot coordinates"""
        if serial not in self.transforms:
            print(f"No transform available for camera {serial}")
            return points_3d

        tcr = self.transforms[serial]["tcr"]  # Camera-to-Robot transform

        # Convert points to millimeters
        points_mm = points_3d * 1000.0

        # Add homogeneous coordinate
        points_homogeneous = np.hstack([points_mm, np.ones((points_mm.shape[0], 1))])

        # Apply transformation
        transformed_points = (tcr @ points_homogeneous.T).T

        # Remove homogeneous coordinate and convert back to meters
        transformed_points = transformed_points[:, :3] / 1000.0

        return transformed_points

    def capture_and_merge_pointclouds(self, output_filename="merged_pointcloud.ply"):
        """Capture point clouds from all cameras and merge them"""
        print("Warming up cameras...")
        for _ in range(30):
            for pipeline in self.pipelines:
                try:
                    pipeline.wait_for_frames()
                except:
                    pass

        print("Capturing point clouds...")

        all_points = []
        all_colors = []

        for i, serial in enumerate(self.calibrated_serials):
            print(f"Capturing from camera {serial}...")

            try:
                points_3d, colors = self.capture_single_pointcloud(i)

                if points_3d is not None and len(points_3d) > 0:
                    # Transform to robot coordinates
                    transformed_points = self.transform_pointcloud(points_3d, serial)

                    all_points.append(transformed_points)
                    all_colors.append(colors)

                    print(f"Camera {serial}: {len(transformed_points)} points captured")
                else:
                    print(f"Camera {serial}: No valid points captured")

            except Exception as e:
                print(f"Error capturing from camera {serial}: {e}")
                continue

        if not all_points:
            print("No valid point clouds captured!")
            return None

        # Merge all point clouds
        merged_points = np.vstack(all_points)
        merged_colors = np.vstack(all_colors)

        print(f"Total merged points: {len(merged_points)}")

        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(merged_points)
        pcd.colors = o3d.utility.Vector3dVector(merged_colors)

        # Optional: Remove statistical outliers
        print("Removing outliers...")
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        print(f"Points after outlier removal: {len(pcd.points)}")

        # Save merged point cloud
        o3d.io.write_point_cloud(output_filename, pcd)
        print(f"Saved merged point cloud: {output_filename}")

        return pcd

    def capture_individual_and_merged(self):
        """Capture individual point clouds and create merged version"""
        print("Warming up cameras...")
        for _ in range(30):
            for pipeline in self.pipelines:
                try:
                    pipeline.wait_for_frames()
                except:
                    pass

        print("Capturing individual point clouds...")

        individual_clouds = []
        all_points = []
        all_colors = []

        for i, serial in enumerate(self.calibrated_serials):
            print(f"Capturing from camera {serial}...")

            try:
                points_3d, colors = self.capture_single_pointcloud(i)

                if points_3d is not None and len(points_3d) > 0:
                    # Save individual untransformed point cloud
                    pcd_individual = o3d.geometry.PointCloud()
                    pcd_individual.points = o3d.utility.Vector3dVector(points_3d)
                    pcd_individual.colors = o3d.utility.Vector3dVector(colors)

                    individual_filename = f"pointcloud_camera_{serial}.ply"
                    o3d.io.write_point_cloud(individual_filename, pcd_individual)
                    print(f"Saved individual: {individual_filename}")

                    # Transform for merged cloud
                    transformed_points = self.transform_pointcloud(points_3d, serial)
                    all_points.append(transformed_points)
                    all_colors.append(colors)

                    individual_clouds.append(pcd_individual)

            except Exception as e:
                print(f"Error capturing from camera {serial}: {e}")
                continue

        # Create merged point cloud
        if all_points:
            merged_points = np.vstack(all_points)
            merged_colors = np.vstack(all_colors)

            pcd_merged = o3d.geometry.PointCloud()
            pcd_merged.points = o3d.utility.Vector3dVector(merged_points)
            pcd_merged.colors = o3d.utility.Vector3dVector(merged_colors)

            # Remove outliers
            pcd_merged, _ = pcd_merged.remove_statistical_outlier(
                nb_neighbors=20, std_ratio=2.0
            )

            # Save merged point cloud
            o3d.io.write_point_cloud("merged_pointcloud.ply", pcd_merged)
            print(f"Saved merged point cloud: merged_pointcloud.ply")
            print(f"Total points in merged cloud: {len(pcd_merged.points)}")

            return individual_clouds, pcd_merged
        else:
            print("No point clouds to merge!")
            return individual_clouds, None

    def cleanup(self):
        """Clean up pipelines"""
        for pipeline in self.pipelines:
            try:
                pipeline.stop()
            except:
                pass


def main():
    """Main function"""
    try:
        merger = MultiCameraPointCloudMerger()

        print("\nChoose an option:")
        print("1. Capture and save merged point cloud only")
        print("2. Capture individual point clouds and merged version")

        choice = input("Enter choice (1 or 2): ").strip()

        if choice == "1":
            merger.capture_and_merge_pointclouds()
        elif choice == "2":
            merger.capture_individual_and_merged()
        else:
            print("Invalid choice. Capturing individual and merged...")
            merger.capture_individual_and_merged()

        print("\nPoint cloud capture completed!")

    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure:")
        print("1. RealSense cameras are connected")
        print("2. Calibration file exists (calib/transforms.npy)")
        print("3. Open3D is installed (pip install open3d)")

    finally:
        try:
            merger.cleanup()
        except:
            pass


if __name__ == "__main__":
    main()
