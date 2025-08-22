#!/usr/bin/env python3
"""
Script to use parsed coordinates from JSONL file with the robot movement system.
This script reads 2D image coordinates from parsed_coordinates.jsonl and uses
the ImageToRobotConverter to move the robot to those positions.
"""

import json
import time
import sys
import os

# Add the camera_calibration directory to path to import test.py
sys.path.append("/home/u-ril/sangyun/camera_calibration")
from test import ImageToRobotConverter


class CoordinateProcessor:
    def __init__(self, jsonl_file_path, camera_serial="317422074281", z_offset=0.0):
        self.jsonl_file_path = jsonl_file_path
        self.coordinates_data = []
        self.converter = None
        self.camera_serial = camera_serial
        self.z_offset = z_offset  # Z-axis offset in millimeters

        # Load coordinates from JSONL file
        self.load_coordinates()

    def load_coordinates(self):
        """Load coordinates from JSONL file"""
        try:
            with open(self.jsonl_file_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:  # Skip empty lines
                        data = json.loads(line)
                        self.coordinates_data.append(data)

            print(f"Loaded {len(self.coordinates_data)} coordinate entries")

        except FileNotFoundError:
            print(f"Error: Could not find file {self.jsonl_file_path}")
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
            sys.exit(1)

    def initialize_converter(self):
        """Initialize the ImageToRobotConverter"""
        try:
            self.converter = ImageToRobotConverter(self.camera_serial)
            print("Robot converter initialized successfully!")
            return True
        except Exception as e:
            print(f"Error initializing converter: {e}")
            return False

    def convert_normalized_to_pixel(
        self, normalized_coord, image_width=640, image_height=480
    ):
        """Convert normalized coordinates (0-1) to pixel coordinates"""
        x_norm, y_norm = normalized_coord
        u = int(x_norm * image_width)
        v = int(y_norm * image_height)
        return u, v

    def set_z_offset(self, offset):
        """Set Z-axis offset in millimeters"""
        self.z_offset = offset
        print(f"Z-axis offset set to: {self.z_offset} mm")

    def apply_z_offset(self, robot_position):
        """Apply Z-axis offset to robot position"""
        if robot_position is None:
            return None

        # Apply offset to Z coordinate (assuming robot_position is [x, y, z, rx, ry, rz])
        modified_position = robot_position.copy()
        modified_position[2] += self.z_offset  # Add offset to Z coordinate

        print(f"Original Z: {robot_position[2]:.2f} mm")
        print(f"Z offset: {self.z_offset} mm")
        print(f"Modified Z: {modified_position[2]:.2f} mm")

        return modified_position

    def process_single_coordinate(self, coord_data, apply_offset=True):
        """Process a single coordinate entry"""
        if self.converter is None:
            print("Error: Converter not initialized!")
            return False

        question_id = coord_data["question_id"]
        coordinate = coord_data["coordinate"]  # [x_norm, y_norm]

        print(f"\nProcessing Question ID: {question_id}")
        print(f"Normalized coordinate: {coordinate}")
        print(f"Z-axis offset: {self.z_offset} mm (apply_offset={apply_offset})")

        # Convert normalized coordinates to pixel coordinates
        # You may need to adjust image dimensions based on your camera setup
        u, v = self.convert_normalized_to_pixel(coordinate)
        print(f"Pixel coordinate: ({u}, {v})")

        # Capture current frame for depth information
        try:
            print("Capturing RGBD frame...")
            _, rgb_image, depth_frame, _ = self.converter.camera.capture_rgbd()
            self.converter.current_depth_frame = depth_frame
            print("Frame captured successfully")

            # If z-offset is enabled, we need to modify the robot movement
            if apply_offset and self.z_offset != 0.0:
                # Get the target position without moving first
                # Convert 2D pixel to 3D camera coordinates
                point_3d_camera = self.converter.pixel_to_3d_camera(
                    u, v, self.converter.current_depth_frame
                )
                if point_3d_camera is None:
                    print("Failed to get 3D camera coordinates")
                    return False

                # Transform to robot coordinates
                robot_coords = self.converter.camera_to_robot_coords(point_3d_camera)
                print(f"Original robot coordinates: {robot_coords}")

                # Apply z-offset to the target position
                modified_position = self.apply_z_offset(robot_coords)

                # Move robot to the modified position
                print(f"Moving robot to modified position: {modified_position}")
                self.converter.move_robot_to_point(modified_position)
            else:
                # Process the coordinate normally (this will move the robot)
                self.converter.process_click(u, v)

        except Exception as e:
            print(f"Error processing coordinate: {e}")
            return False

        return True

    def process_all_coordinates(self, delay_between_moves=3.0):
        """Process all coordinates from the JSONL file"""
        if not self.initialize_converter():
            return

        print(f"\nStarting to process {len(self.coordinates_data)} coordinates...")
        print(f"Delay between moves: {delay_between_moves} seconds")

        for i, coord_data in enumerate(self.coordinates_data):
            print(f"\n{'='*50}")
            print(f"Processing coordinate {i+1}/{len(self.coordinates_data)}")

            # Process the coordinate
            success = self.process_single_coordinate(coord_data)

            if not success:
                response = input(
                    "Error occurred. Continue with next coordinate? (y/n/q): "
                )
                if response.lower() == "q":
                    break
                elif response.lower() != "y":
                    continue

            # Wait before processing next coordinate (except for the last one)
            if i < len(self.coordinates_data) - 1:
                print(f"Waiting {delay_between_moves} seconds before next move...")
                time.sleep(delay_between_moves)

        print("\nFinished processing all coordinates!")

    def process_interactive(self):
        """Process coordinates interactively, allowing user to select which ones to process"""
        if not self.initialize_converter():
            return

        while True:
            print(f"\n{'='*50}")
            print("Available coordinates:")
            for i, coord_data in enumerate(self.coordinates_data):
                print(
                    f"{i}: Question ID {coord_data['question_id']} - {coord_data['coordinate']}"
                )

            print(f"{len(self.coordinates_data)}: Process all coordinates")
            print("q: Quit")
            print("h: Move robot home")
            print("z: Set Z-axis offset")
            print(f"Current Z-offset: {self.z_offset} mm")

            choice = input("\nSelect coordinate index to process: ").strip()

            if choice.lower() == "q":
                break
            elif choice.lower() == "h":
                print("Moving robot home...")
                self.converter.robot.go_home()
                continue
            elif choice.lower() == "z":
                try:
                    new_offset = input(
                        f"Enter new Z-axis offset in mm (current: {self.z_offset}): "
                    ).strip()
                    if new_offset:
                        self.set_z_offset(float(new_offset))
                except ValueError:
                    print("Invalid offset value! Please enter a number.")
                continue

            try:
                index = int(choice)
                if index == len(self.coordinates_data):
                    # Process all coordinates
                    delay = input(
                        "Enter delay between moves (seconds, default 3.0): "
                    ).strip()
                    delay = float(delay) if delay else 3.0
                    self.process_all_coordinates(delay)
                elif 0 <= index < len(self.coordinates_data):
                    # Process single coordinate
                    self.process_single_coordinate(self.coordinates_data[index])
                else:
                    print("Invalid index!")
            except ValueError:
                print("Invalid input! Please enter a number.")
            except KeyboardInterrupt:
                print("\nInterrupted by user")
                break


def main():
    """Main function"""
    jsonl_file = "/home/u-ril/sangyun/parsed_coordinates.jsonl"

    print("2D Image Point to Robot Movement System")
    print("=" * 50)

    # Check if file exists
    if not os.path.exists(jsonl_file):
        print(f"Error: File {jsonl_file} not found!")
        sys.exit(1)

    # Ask for initial Z-offset
    try:
        z_offset_input = input(
            "Enter initial Z-axis offset in mm (default: 0.0): "
        ).strip()
        z_offset = float(z_offset_input) if z_offset_input else 0.0
    except ValueError:
        print("Invalid offset value, using default 0.0 mm")
        z_offset = 0.0

    processor = CoordinateProcessor(jsonl_file, z_offset=z_offset)

    print("\nSelect mode:")
    print("1: Interactive mode (select coordinates individually)")
    print("2: Batch mode (process all coordinates automatically)")

    while True:
        choice = input("Enter choice (1 or 2): ").strip()

        if choice == "1":
            processor.process_interactive()
            break
        elif choice == "2":
            delay = input("Enter delay between moves (seconds, default 3.0): ").strip()
            delay = float(delay) if delay else 3.0
            processor.process_all_coordinates(delay)
            break
        else:
            print("Invalid choice! Please enter 1 or 2.")


if __name__ == "__main__":
    main()
