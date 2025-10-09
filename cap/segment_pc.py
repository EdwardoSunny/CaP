#!/usr/bin/env python3
"""
Captures from multiple cameras, segments an object using SAM+CLIP,
visualizes the result from each camera, then merges the results.
The final merged cloud is cleaned using statistical outlier removal,
saved, and visualized.
"""
import os
import sys
import time
import numpy as np
import pyrealsense2 as rs
from PIL import Image

# Add SAM2 to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'segment/segment-anything2'))

import torch
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from transformers import AutoModel, AutoProcessor

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False
    print("Open3D not available. Visualization and PLY saving will be disabled.")

# Configure camera advanced mode settings
CAMERA_SERIALS = ["317422074281", "327122079374"]

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
    stereo.set_option(rs.option.gain, 16)
    stereo.set_option(rs.option.laser_power, 70)  # 0–360

# --- Segmentation Helper Functions ---

def extract_segmented_objects(image, masks):
    segmented_objects = []
    for mask in masks:
        mask_3d = np.stack([mask, mask, mask], axis=-1)
        segmented = image * mask_3d
        segmented_objects.append(segmented)
    return segmented_objects

def find_best_matching_mask_index(segmented_images, text_prompt, model, processor, device):
    if not segmented_images: return None
    inputs = processor(text=[text_prompt], images=segmented_images, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        similarities = torch.cosine_similarity(outputs.image_embeds, outputs.text_embeds, dim=1)
    best_idx = similarities.argmax().item()
    print(f"CLIP Similarity Scores: {similarities.cpu().numpy().round(2)}")
    print(f"Best match is index {best_idx} with score {similarities[best_idx]:.3f}")
    return best_idx


# --- Main Point Cloud Processing Class ---

class SegmentedPointCloudMerger:
    def __init__(
        self,
        camera_serials,
        calib_file="transforms/transforms.npy",
        icp_file="calib/icp_tf.npy",
        calib_units="m",
        point_cloud_units="m",
    ):
        self.camera_serials = camera_serials
        self.cameras = {}

        # Unit conversion setup
        self.calib_units = calib_units
        self.point_cloud_units = point_cloud_units
        self.unit_scale = self._get_unit_scale(calib_units, point_cloud_units)

        print(f"Unit conversion: {calib_units} -> {point_cloud_units} (scale: {self.unit_scale})")

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
            print(f"Loaded ICP transforms for cameras: {list(self.icp_transforms.keys())}")

        # Initialize cameras
        for serial in camera_serials:
            self._init_camera(serial)

    def _get_unit_scale(self, from_units, to_units):
        """Calculate scale factor for unit conversion"""
        unit_factors = {
            "mm": 0.001,  # mm to meters
            "m": 1.0,      # meters to meters
            "cm": 0.01,    # cm to meters
            "inch": 0.0254,  # inches to meters
        }

        if from_units not in unit_factors or to_units not in unit_factors:
            raise ValueError(f"Unsupported units. Supported: {list(unit_factors.keys())}")

        # Scale factor to convert from_units to to_units
        return unit_factors[from_units] / unit_factors[to_units]

    def _convert_transform_units(self):
        """Convert calibration transforms from calibration units to point cloud units"""
        if self.unit_scale == 1.0:
            return  # No conversion needed

        print(f"Converting calibration transforms from {self.calib_units} to {self.point_cloud_units}...")

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

        print(f"Converting ICP transforms from {self.calib_units} to {self.point_cloud_units}...")

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

    def capture_and_segment_single_camera(self, serial, text_prompt, sam_gen, clip_mod, clip_proc, device_str):
        pipeline = self.cameras[serial]

        # Wait for frames with retry mechanism
        max_retries = 10
        for attempt in range(max_retries):
            try:
                frames = pipeline.wait_for_frames(timeout_ms=5000)
                align = rs.align(rs.stream.color)
                frames = align.process(frames)
                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()
                if depth_frame and color_frame:
                    break
            except RuntimeError as e:
                if attempt < max_retries - 1:
                    print(f"  Retry {attempt + 1}/{max_retries} for camera {serial}...")
                    time.sleep(0.5)
                else:
                    print(f"  Failed to capture from camera {serial} after {max_retries} attempts")
                    return None, None

        if not depth_frame or not color_frame:
            return None, None
        
        color_image_bgr = np.asanyarray(color_frame.get_data())
        color_image_rgb = color_image_bgr[:, :, ::-1].copy()

        print(f"Running SAM on image from camera {serial}...")
        masks_data = sam_gen.generate(color_image_rgb)
        if not masks_data:
            print(f"Warning: SAM found no masks for camera {serial}."); return None, None
        masks = np.array([m['segmentation'] for m in masks_data])
        print(f"Found {len(masks)} masks.")

        print(f"Running CLIP to find '{text_prompt}'...")
        segmented_images = extract_segmented_objects(color_image_rgb, masks)
        best_mask_idx = find_best_matching_mask_index(segmented_images, text_prompt, clip_mod, clip_proc, device_str)
        if best_mask_idx is None: return None, None
        target_mask = masks[best_mask_idx]

        pc = rs.pointcloud(); pc.map_to(color_frame); points_rs = pc.calculate(depth_frame)
        vtx = np.asanyarray(points_rs.get_vertices()).view(np.float32).reshape(-1, 3)
        tex = np.asanyarray(points_rs.get_texture_coordinates()).view(np.float32).reshape(-1, 2)
        valid_mask_depth = (vtx[:, 2] > 0.1) & (vtx[:, 2] < 2.0) & ~np.isinf(vtx).any(axis=1) & ~np.isnan(vtx).any(axis=1)

        h, w, _ = color_image_rgb.shape
        u = np.clip((tex[:, 0] * w).astype(int), 0, w - 1); v = np.clip((tex[:, 1] * h).astype(int), 0, h - 1)
        point_in_segment_mask = target_mask[v, u]
        final_mask = valid_mask_depth & point_in_segment_mask
        
        segmented_points = vtx[final_mask]
        colors_rgb = color_image_rgb[v[final_mask], u[final_mask]] / 255.0
        print(f"Filtered to {len(segmented_points)} points for the object from camera {serial}.")
        return segmented_points, colors_rgb

    def capture_merged_segmented_pointcloud(self, text_prompt, sam_gen, clip_mod, clip_proc, device_str, use_icp=True):
        """Capture and merge segmented point clouds from all cameras in robot frame"""
        print(f"Capturing merged point cloud in robot frame ({self.point_cloud_units})...")

        all_points_robot, all_colors_robot = [], []
        print("\n" + "="*50 + "\nStarting capture and segmentation process...\n" + "="*50)

        # Wait for cameras to stabilize
        time.sleep(1)

        for serial in self.camera_serials:
            print(f"\n--- Processing Camera: {serial} ---")
            points_cam, colors_cam = self.capture_and_segment_single_camera(
                serial, text_prompt, sam_gen, clip_mod, clip_proc, device_str
            )
            if points_cam is not None and len(points_cam) > 0:
                try:
                    # Transform to robot frame
                    points_robot = self.transform_to_robot_frame(points_cam, serial)

                    print(f"Camera {serial} robot frame bounds ({self.point_cloud_units}):")
                    print(f"  X: {points_robot[:,0].min():.3f} to {points_robot[:,0].max():.3f}")
                    print(f"  Y: {points_robot[:,1].min():.3f} to {points_robot[:,1].max():.3f}")
                    print(f"  Z: {points_robot[:,2].min():.3f} to {points_robot[:,2].max():.3f}")

                    self.visualize_individual_cloud(points_robot, colors_cam, serial)
                    all_points_robot.append(points_robot)
                    all_colors_robot.append(colors_cam)

                except Exception as e:
                    print(f"Failed to transform camera {serial}: {e}")
                    continue

        if not all_points_robot:
            print("\nError: Could not segment the object from any camera.")
            return None, None

        # Apply ICP refinement if requested and we have 2+ cameras
        if use_icp and len(all_points_robot) >= 2:
            print("\n" + "="*50 + "\nApplying ICP refinement...\n" + "="*50)
            all_points_robot = self._apply_icp_refinement(all_points_robot, all_colors_robot)

        merged_points = np.vstack(all_points_robot)
        merged_colors = np.vstack(all_colors_robot)

        print(f"\n✓ Successfully merged point cloud:")
        print(f"  Total points: {len(merged_points)}")
        print(f"  Robot frame bounds ({self.point_cloud_units}):")
        print(f"    X: {merged_points[:,0].min():.3f} to {merged_points[:,0].max():.3f}")
        print(f"    Y: {merged_points[:,1].min():.3f} to {merged_points[:,1].max():.3f}")
        print(f"    Z: {merged_points[:,2].min():.3f} to {merged_points[:,2].max():.3f}")

        return merged_points, merged_colors

    def _apply_icp_refinement(self, points_list, colors_list):
        """Apply ICP to align point clouds from different cameras."""
        if not HAS_OPEN3D:
            print("Open3D not available, skipping ICP refinement")
            return points_list

        # Use first camera as anchor
        anchor_pcd = o3d.geometry.PointCloud()
        anchor_pcd.points = o3d.utility.Vector3dVector(points_list[0])
        anchor_pcd.colors = o3d.utility.Vector3dVector(colors_list[0])

        # Estimate normals for anchor
        anchor_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.03, max_nn=30))

        refined_points = [points_list[0]]  # Keep anchor unchanged

        for i in range(1, len(points_list)):
            source_pcd = o3d.geometry.PointCloud()
            source_pcd.points = o3d.utility.Vector3dVector(points_list[i])
            source_pcd.colors = o3d.utility.Vector3dVector(colors_list[i])
            source_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.03, max_nn=30))

            print(f"Refining camera {i+1} to align with camera 1...")

            # Run ICP
            reg = o3d.pipelines.registration.registration_icp(
                source=source_pcd,
                target=anchor_pcd,
                max_correspondence_distance=0.03,
                init=np.eye(4),
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane()
            )

            print(f"  ICP fitness: {reg.fitness:.3f}, RMSE: {reg.inlier_rmse:.4f}")

            # Apply transformation
            source_pcd.transform(reg.transformation)
            refined_points.append(np.asarray(source_pcd.points))

        return refined_points

    def visualize_individual_cloud(self, points, colors, camera_serial):
        """Visualize individual camera point cloud in robot frame"""
        if not HAS_OPEN3D or points is None or len(points) == 0:
            print(f"Skipping visualization for camera {camera_serial} (no points or Open3D not found).")
            return

        print(f"\nDisplaying segmented point cloud from Camera {camera_serial}. Close the window to continue...")
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        # Add coordinate frame at robot origin
        robot_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

        window_name = f"Camera {camera_serial} - Robot Frame ({self.point_cloud_units})"
        o3d.visualization.draw_geometries([pcd, robot_frame], window_name=window_name)

    def save_pointcloud(self, points, colors, filename="segmented_object"):
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
            f.write(f"# X Y Z R G B (Robot Frame Coordinates in {self.point_cloud_units})\n")
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

        print(f"Visualizing merged point cloud in robot frame ({self.point_cloud_units})...")
        print("- Coordinate frame shows robot origin")
        print("- Red=X, Green=Y, Blue=Z axes")

        window_title = f"Merged Point Cloud - Robot Frame ({self.point_cloud_units})"
        o3d.visualization.draw_geometries([pcd, robot_frame], window_name=window_title)

    def clean_save_and_visualize(self, points, colors, filename="segmented_object"):
        """Clean, save, and visualize the final point cloud"""
        if not HAS_OPEN3D or points is None:
            print("Cannot save or visualize (no points or Open3D not found).")
            return

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        print("\nApplying Statistical Outlier Removal to the final cloud...")
        # nb_neighbors: How many neighbors to consider for mean distance calculation.
        # std_ratio: Standard deviation multiplier. A lower value is more aggressive in removing points.
        cleaned_pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        num_removed = len(pcd.points) - len(cleaned_pcd.points)
        print(f"Removed {num_removed} outlier points.")

        # Now, proceed with the cleaned point cloud
        pcd_downsampled = cleaned_pcd.voxel_down_sample(voxel_size=0.005)
        print(f"Downsampled final cloud to {len(pcd_downsampled.points)} points.")

        ply_file = f"{filename}.ply"
        o3d.io.write_point_cloud(ply_file, pcd_downsampled)
        print(f"Saved cleaned and downsampled point cloud to: {ply_file}")

        # Visualize the final, cleaned, and downsampled result
        print("Displaying final merged and cleaned point cloud. Close the window to exit.")
        robot_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        o3d.visualization.draw_geometries([pcd_downsampled, robot_frame], window_name=f"Final Merged & Cleaned Object ({self.point_cloud_units})")

    def cleanup(self):
        """Stop all cameras"""
        for pipeline in self.cameras.values():
            pipeline.stop()
        print("All cameras stopped")

def main():
    # --- 1. INITIALIZE SEGMENTATION MODELS GLOBALLY ---
    print("Initializing segmentation models...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Loading CLIP model...")
    clip_model = AutoModel.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K").to(device)
    clip_processor = AutoProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
    print("CLIP model loaded.")

    print("Loading SAM2 model...")
    sam2_checkpoint = "ckpt/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    sam2 = build_sam2(model_cfg, sam2_checkpoint, apply_postprocessing=False, device=device)
    sam_generator = SAM2AutomaticMaskGenerator(
        model=sam2, points_per_side=32, points_per_batch=128,
        pred_iou_thresh=0.88, stability_score_thresh=0.95, min_mask_region_area=200.0,
    )
    print("SAM2 model loaded.")
    print("All models initialized successfully.")

    # --- CONFIGURATION ---
    TEXT_PROMPT = "can of beans"
    CAMERA_SERIALS = ["327122079374", "317422074281"]
    OUTPUT_FILENAME = "segmented_camera_pcd"

    merger = None
    try:
        # Initialize merger with implicit unit conversion
        print("\nInitializing robot frame merger with automatic unit conversion...")
        merger = SegmentedPointCloudMerger(
            camera_serials=CAMERA_SERIALS,
            calib_file="transforms/transforms.npy",
            calib_units="m",
            point_cloud_units="m"
        )

        # Capture merged point cloud
        merged_points, merged_colors = merger.capture_merged_segmented_pointcloud(
            text_prompt=TEXT_PROMPT,
            sam_gen=sam_generator,
            clip_mod=clip_model,
            clip_proc=clip_processor,
            device_str=device,
            use_icp=True
        )

        if merged_points is not None:
            # Save the raw merged point cloud first
            merger.save_pointcloud(
                points=merged_points,
                colors=merged_colors,
                filename=f"{OUTPUT_FILENAME}_raw"
            )

            # Then clean, save, and visualize
            merger.clean_save_and_visualize(merged_points, merged_colors, filename=OUTPUT_FILENAME)

            print(f"\n🎉 Success! Merged {len(merged_points)} points in robot frame")
            print("Point cloud saved with segmented RGB colors from cameras")
            print(f"Units automatically converted from {merger.calib_units} to {merger.point_cloud_units}")

        else:
            print("❌ Failed to create merged point cloud")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

    finally:
        if merger:
            merger.cleanup()

if __name__ == "__main__":
    main()
