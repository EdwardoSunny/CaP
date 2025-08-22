#!/usr/bin/env python3
"""
Captures from multiple cameras, segments an object using SAM+CLIP,
visualizes the result from each camera, then merges the results.
The final merged cloud is cleaned using statistical outlier removal,
saved, and visualized.
"""
import os
import time
import numpy as np
import pyrealsense2 as rs
from PIL import Image

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
        calib_file="calib/transforms.npy",
        icp_file="calib/icp_tf.npy",
        calib_units="mm",
        point_cloud_units="m",
    ):
        self.camera_serials = camera_serials
        self.cameras = {}
        self.unit_scale = self._get_unit_scale(calib_units, point_cloud_units)
        print(f"\nUnit conversion: {calib_units} -> {point_cloud_units} (scale: {self.unit_scale})")
        self.transforms = self._load_transforms(calib_file)
        self.icp_transforms = self._load_transforms(icp_file) if os.path.exists(icp_file) else {}
        self._convert_transform_units(self.transforms)
        self._convert_transform_units(self.icp_transforms)
        for serial in camera_serials: self._init_camera(serial)

    def _get_unit_scale(self, from_units, to_units):
        return {"mm": 0.001, "m": 1.0, "cm": 0.01}[from_units] / {"mm": 0.001, "m": 1.0, "cm": 0.01}[to_units]

    def _load_transforms(self, file_path):
        if not os.path.exists(file_path): raise FileNotFoundError(f"Transform file {file_path} not found!")
        transforms = np.load(file_path, allow_pickle=True).item()
        print(f"Loaded transforms from {file_path}: {list(transforms.keys())}")
        return transforms

    def _convert_transform_units(self, transforms):
        if self.unit_scale == 1.0: return
        for _, transform in transforms.items():
            if isinstance(transform, dict) and "tcr" in transform: transform["tcr"][:3, 3] *= self.unit_scale
            else: transform[:3, 3] *= self.unit_scale

    def _init_camera(self, serial_number, width=640, height=480, fps=30):
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(serial_number)
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        pipeline.start(config)
        self.cameras[serial_number] = pipeline
        print(f"Camera {serial_number} initialized.")

    def transform_to_robot_frame(self, points, camera_serial):
        tcr = self.transforms[camera_serial]["tcr"]
        points_homo = np.hstack([points, np.ones((points.shape[0], 1))])
        points_robot = (tcr @ points_homo.T).T[:, :3]
        if camera_serial in self.icp_transforms:
            icp_tf = self.icp_transforms[camera_serial]
            points_homo = np.hstack([points_robot, np.ones((points_robot.shape[0], 1))])
            points_robot = (icp_tf @ points_homo.T).T[:, :3]
        return points_robot

    def capture_and_segment_single_camera(self, serial, text_prompt, sam_gen, clip_mod, clip_proc, device_str):
        pipeline = self.cameras[serial]
        frames = pipeline.wait_for_frames(); align = rs.align(rs.stream.color); frames = align.process(frames)
        depth_frame = frames.get_depth_frame(); color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame: return None, None
        
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

    def capture_merged_segmented_pointcloud(self, text_prompt, sam_gen, clip_mod, clip_proc, device_str):
        all_points_robot, all_colors_robot = [], []
        print("\n" + "="*50 + "\nStarting capture and segmentation process...\n" + "="*50)
        time.sleep(2)
        for serial in self.camera_serials:
            print(f"\n--- Processing Camera: {serial} ---")
            points_cam, colors_cam = self.capture_and_segment_single_camera(
                serial, text_prompt, sam_gen, clip_mod, clip_proc, device_str
            )
            if points_cam is not None and len(points_cam) > 0:
                points_robot = self.transform_to_robot_frame(points_cam, serial)
                self.visualize_individual_cloud(points_robot, colors_cam, serial)
                all_points_robot.append(points_robot)
                all_colors_robot.append(colors_cam)
        
        if not all_points_robot:
            print("\nError: Could not segment the object from any camera."); return None, None
            
        merged_points = np.vstack(all_points_robot)
        merged_colors = np.vstack(all_colors_robot)
        print("\n" + "="*50 + "\n✓ Merging Process Complete!\n" + f"  Total merged points: {len(merged_points)}\n" + "="*50)
        return merged_points, merged_colors

    def visualize_individual_cloud(self, points, colors, camera_serial):
        if not HAS_OPEN3D or points is None or len(points) == 0:
            print(f"Skipping visualization for camera {camera_serial} (no points or Open3D not found).")
            return
        print(f"\nDisplaying segmented point cloud from Camera {camera_serial}. Close the window to continue...")
        pcd = o3d.geometry.PointCloud(); pcd.points = o3d.utility.Vector3dVector(points); pcd.colors = o3d.utility.Vector3dVector(colors)
        robot_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        o3d.visualization.draw_geometries([pcd, robot_frame], window_name=f"Segmented Cloud from Camera {camera_serial}")

    def clean_save_and_visualize(self, points, colors, filename="segmented_object"):
        if not HAS_OPEN3D or points is None:
            print("Cannot save or visualize (no points or Open3D not found)."); return
        
        pcd = o3d.geometry.PointCloud(); pcd.points = o3d.utility.Vector3dVector(points); pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # --- NEW: STATISTICAL OUTLIER REMOVAL ---
        print("\nApplying Statistical Outlier Removal to the final cloud...")
        # nb_neighbors: How many neighbors to consider for mean distance calculation.
        # std_ratio: Standard deviation multiplier. A lower value is more aggressive in removing points.
        cleaned_pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        num_removed = len(pcd.points) - len(cleaned_pcd.points)
        print(f"Removed {num_removed} outlier points.")
        # --- END NEW SECTION ---
        
        # Now, proceed with the cleaned point cloud
        pcd_downsampled = cleaned_pcd.voxel_down_sample(voxel_size=0.005)
        print(f"Downsampled final cloud to {len(pcd_downsampled.points)} points.")

        ply_file = f"{filename}.ply"
        o3d.io.write_point_cloud(ply_file, pcd_downsampled)
        print(f"Saved cleaned and downsampled point cloud to: {ply_file}")
        
        # Visualize the final, cleaned, and downsampled result
        print("Displaying final merged and cleaned point cloud. Close the window to exit.")
        robot_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        o3d.visualization.draw_geometries([pcd_downsampled, robot_frame], window_name="Final Merged & Cleaned Object")

    def cleanup(self):
        for pipeline in self.cameras.values():
            pipeline.stop()
        print("All cameras stopped.")

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
    TEXT_PROMPT = "a red cube"
    CAMERA_SERIALS = ["317422074281", "317422075456"]
    OUTPUT_FILENAME = "segmented_camera_pcd"

    merger = None
    try:
        merger = SegmentedPointCloudMerger(camera_serials=CAMERA_SERIALS, calib_units="mm", point_cloud_units="m")
        merged_points, merged_colors = merger.capture_merged_segmented_pointcloud(
            text_prompt=TEXT_PROMPT, sam_gen=sam_generator,
            clip_mod=clip_model, clip_proc=clip_processor, device_str=device
        )
        if merged_points is not None:
            merger.clean_save_and_visualize(merged_points, merged_colors, filename=OUTPUT_FILENAME)

    except Exception as e:
        print(f"\nAn error occurred: {e}"); import traceback; traceback.print_exc()
    finally:
        if merger: merger.cleanup()

if __name__ == "__main__":
    main()
