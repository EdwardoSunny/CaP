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
from camera_utils import load_camera_config, configure_realsense_cameras

# Add SAM2 to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'segment/segment-anything2'))

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


def find_best_matching_mask_index(segmented_images, text_prompt, model, processor, device):
    """Find the mask that best matches the text prompt using CLIP"""
    if not segmented_images:
        return None
    inputs = processor(text=[text_prompt], images=segmented_images, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        similarities = torch.cosine_similarity(outputs.image_embeds, outputs.text_embeds, dim=1)
    best_idx = similarities.argmax().item()
    print(f"CLIP Similarity Scores: {similarities.cpu().numpy().round(2)}")
    print(f"Best match is index {best_idx} with score {similarities[best_idx]:.3f}")
    return best_idx


class RobotFrameMerger:
    """Merge point clouds from multiple cameras in robot coordinate frame with segmentation"""

    def __init__(
        self,
        camera_serials,
        calib_file="calib/transforms.npy",
        icp_file="calib/icp_tf.npy",
        manual_offset_file="transforms/manual_offset.npy",
        calib_units="mm",
        point_cloud_units="m",
        sam_generator=None,
        clip_model=None,
        clip_processor=None,
        device="cuda",
    ):
        self.camera_serials = camera_serials
        self.cameras = {}

        # Unit conversion setup
        self.calib_units = calib_units
        self.point_cloud_units = point_cloud_units
        self.unit_scale = self._get_unit_scale(calib_units, point_cloud_units)

        # Segmentation models
        self.sam_generator = sam_generator
        self.clip_model = clip_model
        self.clip_processor = clip_processor
        self.device = device

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

        # Load manual offset if available (for fine-tuning alignment)
        self.manual_offset = None
        if os.path.exists(manual_offset_file):
            offset_data = np.load(manual_offset_file, allow_pickle=True).item()
            self.manual_offset = offset_data['transform']
            print(f"Loaded manual offset from {manual_offset_file}")
            if 'params' in offset_data:
                params = offset_data['params']
                print(f"  Manual offset parameters: tx={params['tx']:.4f}, ty={params['ty']:.4f}, "
                      f"tz={params['tz']:.4f}, rx={params['rx']:.2f}, ry={params['ry']:.2f}, rz={params['rz']:.2f}")

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
        """Initialize a RealSense camera with post-processing filters"""
        pipeline = rs.pipeline()
        config = rs.config()

        config.enable_device(serial_number)
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

        profile = pipeline.start(config)

        # Create post-processing filters for better depth quality (simplified approach)
        filters = [
            rs.spatial_filter(),
            rs.temporal_filter(),
            rs.hole_filling_filter(),
        ]

        # Store pipeline and filters with camera
        self.cameras[serial_number] = {
            'pipeline': pipeline,
            'filters': filters
        }

        print(f"Camera {serial_number} initialized with depth filters")
        print(f"  - Spatial smoothing: enabled")
        print(f"  - Temporal smoothing: enabled")
        print(f"  - Hole filling: enabled")

    def capture_single_camera(
        self, serial, text_prompt=None, max_depth=2.0, min_depth=0.1, max_points=100000
    ):
        """Capture point cloud from a single camera with optional segmentation"""
        camera_data = self.cameras[serial]
        pipeline = camera_data['pipeline']
        filters = camera_data['filters']

        # Capture frame
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            print(f"Failed to capture from camera {serial}")
            return None, None

        # Apply post-processing filters to depth frame (simplified approach)
        for filter in filters:
            depth_frame = filter.process(depth_frame)

        color_image = np.asanyarray(color_frame.get_data())

        # Create point cloud using RealSense (same as realsense_toolbox)
        raw_pcd = rs.pointcloud()
        raw_pcd.map_to(color_frame)
        points = raw_pcd.calculate(depth_frame)

        # Extract points and colors (same as realsense_toolbox)
        points_3d = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)
        tex = np.asanyarray(points.get_texture_coordinates()).view(np.float32).reshape(-1, 2)

        # Get colors
        h, w = color_image.shape[:2]
        u = np.clip((tex[:, 0] * w).astype(np.int32), 0, w - 1)
        v = np.clip((tex[:, 1] * h).astype(np.int32), 0, h - 1)

        colors = color_image[v, u][:, ::-1] / 255.0  # BGR to RGB and normalize to [0,1]

        # Filter valid points (same as realsense_toolbox)
        valid_mask = (
            (points_3d[:, 2] > min_depth)
            & (points_3d[:, 2] < max_depth)
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
                masks = np.array([m['segmentation'] for m in masks_data])
                print(f"  Found {len(masks)} masks")

                # Run CLIP to find best matching mask
                print(f"  Running CLIP to find '{text_prompt}'...")
                segmented_images = extract_segmented_objects(color_image_rgb, masks)
                best_mask_idx = find_best_matching_mask_index(
                    segmented_images, text_prompt, self.clip_model, self.clip_processor, self.device
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

        # Apply manual offset if available (only to second camera for fine-tuning alignment)
        # The manual offset is typically tuned for the second camera to align with the first
        if self.manual_offset is not None and camera_serial == self.camera_serials[1]:
            ones = np.ones((points_robot.shape[0], 1))
            points_homo = np.hstack([points_robot, ones])
            points_robot = (self.manual_offset @ points_homo.T).T[:, :3]
            print(f"Camera {camera_serial}: Applied manual offset for fine-tuning")

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
        self, text_prompt=None, crop_workspace=False, workspace_bounds=None, color_by_camera=False
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

            # Capture from camera (with optional segmentation)
            points_cam, colors_cam = self.capture_single_camera(serial, text_prompt=text_prompt)

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

        print(f"\nSuccessfully merged point cloud:")
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
        for camera_data in self.cameras.values():
            camera_data['pipeline'].stop()
        print("All cameras stopped")


def main():
    """Main function - easy to use interface"""

    # Load configuration
    config = load_camera_config()
    camera_serials = config['camera_serials']
    calib_config = config['calibration']
    seg_config = config['segmentation']
    ws_config = config['workspace_bounds']

    # Configure cameras
    configure_realsense_cameras(config)

    TEXT_PROMPT = seg_config['text_prompt']

    # Optional workspace cropping bounds
    WORKSPACE_BOUNDS = None
    if ws_config['enabled']:
        WORKSPACE_BOUNDS = [ws_config['min_x'], ws_config['min_y'], ws_config['min_z'],
                           ws_config['max_x'], ws_config['max_y'], ws_config['max_z']]

    # Initialize segmentation models if text prompt is provided
    sam_generator = None
    clip_model = None
    clip_processor = None
    device = "cpu"

    if TEXT_PROMPT:
        print("Initializing segmentation models...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")

        print("Loading CLIP model...")
        clip_model = AutoModel.from_pretrained(seg_config['clip_model']).to(device)
        clip_processor = AutoProcessor.from_pretrained(seg_config['clip_model'])
        print("CLIP model loaded.")

        print("Loading SAM2 model...")
        sam2_checkpoint = seg_config['sam2_checkpoint']
        model_cfg = seg_config['sam2_model_cfg']
        sam2 = build_sam2(model_cfg, sam2_checkpoint, apply_postprocessing=False, device=device)
        sam_generator = SAM2AutomaticMaskGenerator(
            model=sam2,
            points_per_side=seg_config['points_per_side'],
            points_per_batch=seg_config['points_per_batch'],
            pred_iou_thresh=seg_config['pred_iou_thresh'],
            stability_score_thresh=seg_config['stability_score_thresh'],
            min_mask_region_area=seg_config['min_mask_region_area'],
        )
        print("SAM2 model loaded.")
        print("All models initialized successfully.")

    try:
        # Initialize merger with implicit unit conversion
        print("Initializing robot frame merger with automatic unit conversion...")
        merger = RobotFrameMerger(
            camera_serials=camera_serials,
            calib_file=calib_config['transforms_file'],
            icp_file=calib_config['icp_file'],
            manual_offset_file=calib_config['manual_offset_file'],
            calib_units=calib_config['calib_units'],
            point_cloud_units=calib_config['point_cloud_units'],
            sam_generator=sam_generator,
            clip_model=clip_model,
            clip_processor=clip_processor,
            device=device,
        )

        # Capture merged point cloud
        result = merger.capture_merged_pointcloud(
            text_prompt=TEXT_PROMPT,  # Set to None to disable segmentation
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
            filename_suffix = "_segmented" if TEXT_PROMPT else "_colored"
            merger.save_pointcloud(
                points=merged_points,
                colors=merged_colors,
                filename=f"merged_robot_frame{filename_suffix}",
                camera_data=camera_data,
            )

            # Show individual cameras first
            print("\nShowing individual cameras...")
            merger.visualize_cameras_separately(camera_data)

            # Then show merged view
            print("\nShowing merged view...")
            merger.visualize_pointcloud(merged_points, merged_colors, camera_data)

            print(f"\nSuccess! Merged {len(merged_points)} points in robot frame")
            if TEXT_PROMPT:
                print(f"Point cloud saved with segmented object: '{TEXT_PROMPT}'")
            else:
                print("Point cloud saved with original RGB colors from cameras")
            print(
                f"Units automatically converted from {merger.calib_units} to {merger.point_cloud_units}"
            )

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
