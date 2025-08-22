#!/usr/bin/env python3
"""
Sample code for converting 2D image coordinates to 3D robot coordinates
and moving the robot to the selected position.

Usage:
1. Run the script
2. Click on any point in the camera image
3. Robot will move to that 3D location

Requirements:
- Calibrated camera (transforms.npy must exist)
- Robot connected and operational
- Red marker visible for initial verification
"""

import numpy as np
import cv2
import os
from rs_streamer import RealsenseStreamer
from multicam import XarmEnv
import time


# 317422074281
class ImageToRobotConverter:
    def __init__(self, camera_serial="317422075456"):
        self.camera_serial = camera_serial

        # Initialize camera
        self.camera = RealsenseStreamer(camera_serial)

        # Initialize robot
        self.robot = XarmEnv()

        # Load calibration data
        if not os.path.exists("calib/transforms.npy"):
            raise FileNotFoundError(
                "Calibration file not found. Run calibration first!"
            )

        self.transforms = np.load("calib/transforms.npy", allow_pickle=True).item()
        self.tcr = self.transforms[camera_serial]["tcr"]  # Camera-to-Robot transform

        print("System initialized successfully!")
        print("Click on the image to move robot to that position")
        print("Press 'q' to quit, 'h' to go home")

        # Mouse callback variables
        self.clicked_point = None
        self.current_rgb = None
        self.current_depth_frame = None

    def mouse_callback(self, event, x, y, flags, param):
        """Mouse callback function for clicking on image"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.clicked_point = (x, y)
            print(f"Clicked at pixel: ({x}, {y})")

    def pixel_to_3d_camera(self, u, v, depth_frame):
        """Convert 2D pixel + depth to 3D camera coordinates"""
        try:
            # Get 3D point in camera coordinates (meters)
            point_3d_camera = np.array(self.camera.deproject((u, v), depth_frame))
            return point_3d_camera
        except Exception as e:
            print(f"Error in pixel to 3D conversion: {e}")
            return None

    def camera_to_robot_coords(self, point_3d_camera):
        """Transform 3D camera coordinates to robot coordinates"""
        # Convert to millimeters
        point_3d_camera_mm = point_3d_camera * 1000.0

        # Apply transformation matrix
        point_homogeneous = np.append(point_3d_camera_mm, 1)
        robot_homogeneous = self.tcr @ point_homogeneous
        point_3d_robot = robot_homogeneous[:3]

        return point_3d_robot

    def move_robot_to_point(self, robot_coords):
        """Move robot to specified 3D coordinates"""
        try:
            # Get current orientation (keep same orientation)
            _, current_orientation = self.robot.pose_ee()

            print(f"Moving robot to: {robot_coords}")
            print(f"Current orientation: {current_orientation}")

            # Move robot to position
            ret = self.robot.move_to_ee_pose(robot_coords, current_orientation)

            if ret == 0:
                print("Robot moved successfully!")
            else:
                print(f"Robot movement failed with error: {ret}")

        except Exception as e:
            print(f"Error moving robot: {e}")

    def process_click(self, u, v):
        """Process a mouse click at pixel (u,v)"""
        if self.current_depth_frame is None:
            print("No depth frame available")
            return

        # Convert 2D pixel to 3D camera coordinates
        point_3d_camera = self.pixel_to_3d_camera(u, v, self.current_depth_frame)
        if point_3d_camera is None:
            print("Failed to get 3D camera coordinates")
            return

        print(f"3D Camera coordinates: {point_3d_camera}")

        # Transform to robot coordinates
        robot_coords = self.camera_to_robot_coords(point_3d_camera)
        print(f"3D Robot coordinates: {robot_coords}")

        # Check if coordinates are reasonable (basic safety check)
        if (
            robot_coords[0] < 100
            or robot_coords[0] > 800
            or abs(robot_coords[1]) > 500
            or robot_coords[2] < 100
            or robot_coords[2] > 600
        ):
            print("WARNING: Coordinates seem outside safe workspace!")
            print("Coordinates:", robot_coords)
            response = input("Continue anyway? (y/n): ")
            if response.lower() != "y":
                print("Movement cancelled")
                return

        # Move robot to the calculated position
        self.move_robot_to_point(robot_coords)

    def run(self):
        """Main loop for image display and interaction"""
        cv2.namedWindow("Camera Feed")
        cv2.setMouseCallback("Camera Feed", self.mouse_callback)

        while True:
            try:
                # Capture RGB-D image
                _, rgb_image, depth_frame, depth_img_vis = self.camera.capture_rgbd()
                self.current_rgb = rgb_image
                self.current_depth_frame = depth_frame

                # Draw crosshair if point was clicked
                display_image = rgb_image.copy()
                if self.clicked_point is not None:
                    u, v = self.clicked_point
                    cv2.circle(display_image, (u, v), 5, (0, 255, 0), -1)
                    cv2.line(display_image, (u - 10, v), (u + 10, v), (0, 255, 0), 2)
                    cv2.line(display_image, (u, v - 10), (u, v + 10), (0, 255, 0), 2)

                # Display image
                cv2.imshow("Camera Feed", display_image)

                # Process keyboard input
                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    print("Quitting...")
                    break
                elif key == ord("h"):
                    print("Moving robot home...")
                    self.robot.go_home()
                elif key == ord(" ") and self.clicked_point is not None:
                    # Process the clicked point
                    u, v = self.clicked_point
                    self.process_click(u, v)
                    self.clicked_point = None

                # Auto-process click (alternative to spacebar)
                if self.clicked_point is not None:
                    u, v = self.clicked_point
                    time.sleep(0.5)  # Small delay to see the crosshair
                    self.process_click(u, v)
                    self.clicked_point = None

            except KeyboardInterrupt:
                print("Interrupted by user")
                break
            except Exception as e:
                print(f"Error in main loop: {e}")
                continue

        cv2.destroyAllWindows()
        print("Program ended")


def main():
    """Main function"""
    try:
        converter = ImageToRobotConverter()
        converter.run()
    except Exception as e:
        print(f"Error initializing system: {e}")
        print("Make sure:")
        print("1. Camera is connected")
        print("2. Robot is connected and operational")
        print("3. Calibration file exists (calib/transforms.npy)")


if __name__ == "__main__":
    main()
