#!/usr/bin/env python3
"""
Simple script to capture from both cameras, transform to robot frame, merge, and segment.
This does exactly what pc.py does but adds SAM+CLIP segmentation to extract specific objects.
"""

import numpy as np
import pyrealsense2 as rs
import os
import sys
import time
import torch
from PIL import Image
from camera_exposure_config import DEPTH_EXPOSURE, RGB_EXPOSURE

# Add SAM2 to path
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "segment/segment-anything2")
)

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from transformers import AutoModel, AutoProcessor

try:
    import open3d as o3d

    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False
    print("Open3D not available. Will save as text file.")


# --- Segmentation Helper Functions ---


def extract_segmented_objects(image, masks):
    """Extract segmented objects from image using masks"""
    segmented_objects = []
    for mask in masks:
        mask_3d = np.stack([mask, mask, mask], axis=-1)
        segmented = image * mask_3d
        segmented_objects.append(segmented)
    return segmented_objects


def find_best_matching_mask_index(
    segmented_images, text_prompt, model, processor, device
):
    """Find the mask that best matches the text prompt using CLIP"""
    if not segmented_images:
        return None
    inputs = processor(
        text=[text_prompt], images=segmented_images, return_tensors="pt", padding=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        similarities = torch.cosine_similarity(
            outputs.image_embeds, outputs.text_embeds, dim=1
        )
    best_idx = similarities.argmax().item()
    print(f"CLIP Similarity Scores: {similarities.cpu().numpy().round(2)}")
    print(f"Best match is index {best_idx} with score {similarities[best_idx]:.3f}")
    return best_idx


class RobotFrameMerger:
    """Merge point clouds from multiple cameras in robot coordinate frame with segmentation"""

    def __init__(
        self,
        camera_serials,
        calib_file="../transforms/transforms.npy",
        max_depth=2.0,
        min_depth=0.1,
        sam_generator=None,
        clip_model=None,
        clip_processor=None,
        device="cuda",
    ):
        self.camera_serials = camera_serials
        self.cameras = {}
        self.max_depth = max_depth
        self.min_depth = min_depth

        # Segmentation models
        self.sam_generator = sam_generator
        self.clip_model = clip_model
        self.clip_processor = clip_processor
        self.device = device

        # Load calibration transforms
        if not os.path.exists(calib_file):
            raise FileNotFoundError(f"Calibration file {calib_file} not found!")

        self.transforms = np.load(calib_file, allow_pickle=True).item()
        print(f"Loaded transforms for cameras: {list(self.transforms.keys())}")

        # Initialize cameras
        for serial in camera_serials:
            self._init_camera(serial)

    def _init_camera(self, serial_number):
        """Initialize a RealSense camera with auto defaults"""
        pipeline = rs.pipeline()
        config = rs.config()

        config.enable_device(serial_number)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        profile = pipeline.start(config)

        # Configure camera with auto defaults
        self._configure_camera(pipeline)

        # Create post-processing filters (simplified approach like robot_calib)
        filters = [
            rs.spatial_filter(),
            rs.temporal_filter(),
            rs.hole_filling_filter(),
        ]

        self.cameras[serial_number] = {"pipeline": pipeline, "filters": filters}

        print(f"Camera {serial_number} initialized")

    def _configure_camera(self, pipeline):
        """Configure camera with simple auto defaults - no JSON, no config files"""
        # Get the active profile and device
        profile = pipeline.get_active_profile()
        device = profile.get_device()

        # Find the Stereo Module (depth) sensor
        stereo = next(
            s
            for s in device.query_sensors()
            if "Stereo" in s.get_info(rs.camera_info.name)
        )

        # Find the RGB sensor
        rgb_sensor = next(
            (
                s
                for s in device.query_sensors()
                if "RGB" in s.get_info(rs.camera_info.name)
            ),
            None,
        )

        # Depth sensor settings
        if DEPTH_EXPOSURE is not None:
            stereo.set_option(rs.option.enable_auto_exposure, 0)
            stereo.set_option(rs.option.exposure, DEPTH_EXPOSURE)
        else:
            stereo.set_option(rs.option.enable_auto_exposure, 1)

        # RGB sensor settings
        if rgb_sensor:
            rgb_sensor.set_option(rs.option.enable_auto_white_balance, 1)

            if RGB_EXPOSURE is not None:
                rgb_sensor.set_option(rs.option.enable_auto_exposure, 0)
                rgb_sensor.set_option(rs.option.exposure, RGB_EXPOSURE)
            else:
                rgb_sensor.set_option(rs.option.enable_auto_exposure, 1)

    def capture_single_camera(self, serial, text_prompt=None):
        """Capture point cloud from a single camera with optional segmentation"""
        camera_data = self.cameras[serial]
        pipeline = camera_data["pipeline"]
        filters = camera_data["filters"]

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

        # Create point cloud (same as robot_calib)
        raw_pcd = rs.pointcloud()
        raw_pcd.map_to(color_frame)
        points = raw_pcd.calculate(depth_frame)

        # Extract points and colors (same as robot_calib)
        points_3d = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)
        tex = (
            np.asanyarray(points.get_texture_coordinates())
            .view(np.float32)
            .reshape(-1, 2)
        )

        # Get colors
        h, w = color_image.shape[:2]
        u = np.clip((tex[:, 0] * w).astype(np.int32), 0, w - 1)
        v = np.clip((tex[:, 1] * h).astype(np.int32), 0, h - 1)
        colors = color_image[v, u][:, ::-1] / 255.0  # BGR to RGB

        # Filter valid points (same as robot_calib)
        valid_mask = (
            (points_3d[:, 2] > self.min_depth)
            & (points_3d[:, 2] < self.max_depth)
            & np.isfinite(points_3d).all(axis=1)
        )

        # Apply segmentation if text_prompt is provided
        if text_prompt and self.sam_generator and self.clip_model:
            print(f"Camera {serial}: Applying segmentation for '{text_prompt}'...")

            # Convert color image for SAM (BGR to RGB)
            color_image_rgb = color_image[:, :, ::-1].copy()

            # Run SAM to get masks
            print(f"  Running SAM on image from camera {serial}...")
            masks_data = self.sam_generator.generate(color_image_rgb)

            if not masks_data:
                print(f"  Warning: SAM found no masks for camera {serial}")
                segmentation_mask = valid_mask  # Fall back to just valid points
            else:
                masks = np.array([m["segmentation"] for m in masks_data])
                print(f"  Found {len(masks)} masks")

                # Run CLIP to find best matching mask
                print(f"  Running CLIP to find '{text_prompt}'...")
                segmented_images = extract_segmented_objects(color_image_rgb, masks)
                best_mask_idx = find_best_matching_mask_index(
                    segmented_images,
                    text_prompt,
                    self.clip_model,
                    self.clip_processor,
                    self.device,
                )

                if best_mask_idx is None:
                    print(f"  Warning: Could not find matching mask")
                    segmentation_mask = valid_mask
                else:
                    # Apply the selected mask
                    target_mask = masks[best_mask_idx]
                    point_in_segment_mask = target_mask[v, u]
                    segmentation_mask = valid_mask & point_in_segment_mask
                    print(f"  Segmentation mask applied")
        else:
            segmentation_mask = valid_mask

        valid_points = points_3d[segmentation_mask]
        valid_colors = colors[segmentation_mask]

        print(f"Camera {serial}: {len(valid_points)} valid points captured")

        return valid_points, valid_colors

    def transform_to_robot_frame(self, points, camera_serial):
        """Transform points from camera coordinates to robot coordinates"""
        if camera_serial not in self.transforms:
            raise ValueError(f"No calibration found for camera {camera_serial}")

        # Get tcr transform
        tcr = self.transforms[camera_serial]["tcr"]

        # Ensure transform is 4x4
        if tcr.shape == (3, 4):
            tcr_4x4 = np.vstack([tcr, [0, 0, 0, 1]])
        else:
            tcr_4x4 = tcr

        # Convert to homogeneous coordinates
        ones = np.ones((points.shape[0], 1))
        points_homo = np.hstack([points, ones])

        # Transform to robot frame
        points_robot = (tcr_4x4 @ points_homo.T).T[:, :3]

        return points_robot

    def capture_merged_pointcloud(self, text_prompt=None):
        """Capture and merge point clouds from all cameras in robot frame"""
        print("Capturing merged point cloud in robot frame...")

        all_points_robot = []
        all_colors = []

        # Wait for cameras to stabilize
        time.sleep(1)

        # Warm up filters (let temporal filter build history)
        print("Warming up filters (capturing 30 frames)...")
        for _ in range(30):
            for serial in self.camera_serials:
                camera_data = self.cameras[serial]
                pipeline = camera_data["pipeline"]
                filters = camera_data["filters"]

                frames = pipeline.wait_for_frames()
                depth_frame = frames.get_depth_frame()
                if depth_frame:
                    for filter in filters:
                        depth_frame = filter.process(depth_frame)

        # Process each camera
        for serial in self.camera_serials:
            print(f"\nProcessing camera {serial}...")

            # Capture from camera (with optional segmentation)
            result = self.capture_single_camera(serial, text_prompt=text_prompt)

            if result is None:
                print(f"Skipping camera {serial} - capture failed")
                continue

            points_cam, colors_cam = result

            try:
                # Transform to robot frame
                points_robot = self.transform_to_robot_frame(points_cam, serial)

                print(f"Camera {serial} robot frame bounds:")
                print(
                    f"  X: {points_robot[:,0].min():.3f} to {points_robot[:,0].max():.3f}"
                )
                print(
                    f"  Y: {points_robot[:,1].min():.3f} to {points_robot[:,1].max():.3f}"
                )
                print(
                    f"  Z: {points_robot[:,2].min():.3f} to {points_robot[:,2].max():.3f}"
                )

                all_points_robot.append(points_robot)
                all_colors.append(colors_cam)

            except Exception as e:
                print(f"Failed to transform camera {serial}: {e}")
                continue

        # Merge all point clouds
        if not all_points_robot:
            print("No valid point clouds captured!")
            return None, None

        merged_points = np.vstack(all_points_robot)
        merged_colors = np.vstack(all_colors)

        print(f"\nSuccessfully merged point cloud:")
        print(f"  Total points: {len(merged_points)}")
        print(f"  Robot frame bounds:")
        print(
            f"    X: {merged_points[:,0].min():.3f} to {merged_points[:,0].max():.3f}"
        )
        print(
            f"    Y: {merged_points[:,1].min():.3f} to {merged_points[:,1].max():.3f}"
        )
        print(
            f"    Z: {merged_points[:,2].min():.3f} to {merged_points[:,2].max():.3f}"
        )

        return merged_points, merged_colors

    def save_pointcloud(self, points, colors, filename="merged_robot_frame"):
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
            f.write("# X Y Z R G B (Robot Frame Coordinates in meters)\n")
            for i in range(len(points)):
                f.write(
                    f"{points[i,0]:.6f} {points[i,1]:.6f} {points[i,2]:.6f} "
                    f"{colors[i,0]:.3f} {colors[i,1]:.3f} {colors[i,2]:.3f}\n"
                )
        print(f"Saved: {txt_file}")

    def visualize_pointcloud(self, points, colors):
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

        print("Visualizing merged point cloud in robot frame...")
        print("- Coordinate frame shows robot origin")
        print("- Red=X, Green=Y, Blue=Z axes")

        o3d.visualization.draw_geometries(
            [pcd, robot_frame], window_name="Merged Point Cloud - Robot Frame"
        )

    def cleanup(self):
        """Stop all cameras"""
        for camera_data in self.cameras.values():
            camera_data["pipeline"].stop()
        print("All cameras stopped")


def main():
    """Main function - easy to use interface"""

    # Hardcoded camera serials - edit these for your setup
    camera_serials = ["327122079374", "317422074281"]
    calib_file = "../transforms/transforms.npy"

    # Depth range settings
    max_depth = 2.0  # meters
    min_depth = 0.1  # meters

    # Segmentation prompt (set this to what you want to segment, or None to disable)
    TEXT_PROMPT = "Orange disinfecting wipes"

    # Segmentation model settings
    SAM2_CHECKPOINT = "ckpt/sam2.1_hiera_large.pt"
    SAM2_MODEL_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"  # Path relative to parent directory
    CLIP_MODEL = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"

    print(f"Camera serials: {camera_serials}")
    print(f"Calibration file: {calib_file}")
    print(f"Depth range: {min_depth}m to {max_depth}m")
    print(
        f"Depth exposure: {'AUTO' if DEPTH_EXPOSURE is None else f'{DEPTH_EXPOSURE} µs'}"
    )
    print(
        f"RGB exposure: {'AUTO' if RGB_EXPOSURE is None else f'{RGB_EXPOSURE} µs'}"
    )
    if TEXT_PROMPT:
        print(f"Segmentation: '{TEXT_PROMPT}'")

    # Initialize segmentation models if text prompt is provided
    sam_generator = None
    clip_model = None
    clip_processor = None
    device = "cpu"

    if TEXT_PROMPT:
        print("\nInitializing segmentation models...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")

        print("Loading CLIP model...")
        clip_model = AutoModel.from_pretrained(CLIP_MODEL).to(device)
        clip_processor = AutoProcessor.from_pretrained(CLIP_MODEL)
        print("CLIP model loaded.")

        print("Loading SAM2 model...")
        # Change to parent directory temporarily for SAM2 to find configs
        import os
        original_dir = os.getcwd()
        os.chdir("..")
        sam2 = build_sam2(
            SAM2_MODEL_CFG, SAM2_CHECKPOINT, apply_postprocessing=False, device=device
        )
        os.chdir(original_dir)
        sam_generator = SAM2AutomaticMaskGenerator(
            model=sam2,
            points_per_side=32,
            points_per_batch=128,
            pred_iou_thresh=0.88,
            stability_score_thresh=0.95,
            min_mask_region_area=200.0,
        )
        print("SAM2 model loaded.")
        print("All models initialized successfully.")

    try:
        # Initialize merger
        print("\nInitializing robot frame merger...")
        merger = RobotFrameMerger(
            camera_serials=camera_serials,
            calib_file=calib_file,
            max_depth=max_depth,
            min_depth=min_depth,
            sam_generator=sam_generator,
            clip_model=clip_model,
            clip_processor=clip_processor,
            device=device,
        )

        # Capture merged point cloud
        merged_points, merged_colors = merger.capture_merged_pointcloud(
            text_prompt=TEXT_PROMPT  # Set to None to disable segmentation
        )

        if merged_points is not None:
            # Save point cloud
            filename_suffix = "_segmented" if TEXT_PROMPT else "_colored"
            merger.save_pointcloud(
                points=merged_points,
                colors=merged_colors,
                filename=f"merged_robot_frame{filename_suffix}",
            )

            # Visualize
            merger.visualize_pointcloud(merged_points, merged_colors)

            print(f"\nSuccess! Merged {len(merged_points)} points in robot frame")
            if TEXT_PROMPT:
                print(f"Point cloud saved with segmented object: '{TEXT_PROMPT}'")
            else:
                print("Point cloud saved with original RGB colors from cameras")

        else:
            print("Failed to create merged point cloud")

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()

    finally:
        if "merger" in locals():
            merger.cleanup()


if __name__ == "__main__":
    main()
