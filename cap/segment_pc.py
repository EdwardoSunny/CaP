#!/usr/bin/env python3
"""
Capture from multiple RealSense cameras, transform to robot frame, merge,
and segment objects using Molmo (VLM point prediction) + SAM2 (automatic masks).

Pipeline per camera:
1. Capture aligned RGB + depth.
2. Run SAM2AutomaticMaskGenerator to produce all candidate masks unconditionally.
3. Send RGB to Molmo API → get (x, y) point(s) on the target object.
4. Use Molmo points to vote on which auto-generated mask corresponds to the object.
5. Apply the selected mask to select 3D points belonging to the object.
"""

import base64
import io
import logging
import re
import numpy as np
import pyrealsense2 as rs
import os
import sys
import time
import torch
from PIL import Image
from openai import OpenAI

# Handle both package import and direct script execution
try:
    from .camera_exposure_config import DEPTH_EXPOSURE, RGB_EXPOSURE
except ImportError:
    from camera_exposure_config import DEPTH_EXPOSURE, RGB_EXPOSURE

# Add SAM2 to path
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "segment/segment-anything2")
)

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

try:
    import open3d as o3d

    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False
    print("Open3D not available. Will save as text file.")

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Molmo VLM helpers
# ---------------------------------------------------------------------------

# Default Molmo endpoint — override via MOLMO_BASE_URL env var
MOLMO_BASE_URL = os.environ.get("MOLMO_BASE_URL", "http://scai4.cs.ucla.edu:8000/v1")
MOLMO_MODEL = os.environ.get("MOLMO_MODEL", "allenai/Molmo2-8B")


def _encode_image_to_base64(image_rgb: np.ndarray) -> str:
    """Encode an HWC uint8 RGB numpy image to a base64 JPEG string."""
    pil_img = Image.fromarray(image_rgb)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _parse_molmo_points(response_text: str) -> list[tuple[float, float]]:
    """Parse Molmo ``<points>`` tags and return list of (norm_x, norm_y) in [0, 1000].

    Molmo returns points like:
        Single point:  <points coords="x1 y1">description</points>
        Multi-point:   <points coords="x1 y1 x2 y2 ...">description</points>

    Known quirks handled:
    - Molmo often prepends a ``(1, 1)`` corner artifact. Stripped when other
      real points exist (any number >= 15).
    - When asked for multiple points, Molmo sometimes interleaves index numbers
      (1, 2, 3...) in the coordinate list. Stripped via ``n >= 15`` filter.

    Also handles bare ``<point>`` tags:
        <point x="123" y="456" alt="...">description</point>
    """
    points = []

    # <points coords="x1 y1 x2 y2 ...">
    for match in re.finditer(r'<points coords="([^"]+)">', response_text):
        raw_nums = list(map(float, match.group(1).split()))

        # Filter out small index numbers that Molmo interleaves
        # (e.g. "500 510 2 630 533 3 ..." or "1 1 380 85").
        # Valid coordinates are typically >= 15 in the 0-1000 grid.
        # Always filter; if nothing survives >= 15, fall back to raw.
        filtered = [n for n in raw_nums if n >= 15]
        if len(filtered) >= 2:
            nums = filtered
        else:
            nums = raw_nums

        # Pair up remaining numbers
        points.extend((nums[i], nums[i + 1]) for i in range(0, len(nums) - 1, 2))

    # <point x="..." y="..." ...>
    for match in re.finditer(r'<point x="([^"]+)" y="([^"]+)"', response_text):
        points.append((float(match.group(1)), float(match.group(2))))

    return points


def _query_molmo_single_point(
    b64: str,
    text_prompt: str,
    client: OpenAI,
    model: str,
    call_idx: int,
) -> list[tuple[float, float]]:
    """Send a single Molmo request asking for one point. Returns parsed points."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Point at the {text_prompt}"},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                        },
                    ],
                }
            ],
            max_tokens=128,
        )
        raw = response.choices[0].message.content or ""
        logger.info(f"Molmo call #{call_idx} response: {raw}")
        return _parse_molmo_points(raw)
    except Exception as e:
        logger.error(f"Molmo call #{call_idx} failed: {e}")
        return []


def query_molmo_for_points(
    image_rgb: np.ndarray,
    text_prompt: str,
    base_url: str = MOLMO_BASE_URL,
    model: str = MOLMO_MODEL,
    n_calls: int = 6,
) -> list[tuple[float, float]]:
    """Ask Molmo to point at *text_prompt* in *image_rgb*.

    Fires *n_calls* independent single-point requests in parallel (via threads)
    to get diverse, independent point predictions. Each call asks Molmo to
    point at the object once, avoiding the multi-point index-interleaving issue.

    Returns list of (norm_x, norm_y) in the 0-1000 Molmo coordinate space.
    Empty list if all calls fail or Molmo finds nothing.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    client = OpenAI(base_url=base_url, api_key="not-needed")
    b64 = _encode_image_to_base64(image_rgb)

    all_points: list[tuple[float, float]] = []

    with ThreadPoolExecutor(max_workers=n_calls) as pool:
        futures = {
            pool.submit(_query_molmo_single_point, b64, text_prompt, client, model, i): i
            for i in range(n_calls)
        }
        for future in as_completed(futures):
            idx = futures[future]
            try:
                pts = future.result()
                all_points.extend(pts)
            except Exception as e:
                logger.error(f"Molmo call #{idx} raised: {e}")

    logger.info(f"Molmo: {len(all_points)} points from {n_calls} parallel calls")
    return all_points


def molmo_points_to_pixel(
    molmo_points: list[tuple[float, float]], img_w: int, img_h: int
) -> np.ndarray:
    """Convert Molmo normalised-1000 points to pixel coords (Nx2, XY order)."""
    if not molmo_points:
        return np.empty((0, 2), dtype=np.float32)
    arr = np.array(molmo_points, dtype=np.float32)
    arr[:, 0] = arr[:, 0] / 1000.0 * img_w
    arr[:, 1] = arr[:, 1] / 1000.0 * img_h
    return arr


class RobotFrameMerger:
    """Merge point clouds from multiple cameras in robot coordinate frame with segmentation.

    Uses SAM2AutomaticMaskGenerator for unconditional mask generation, then
    Molmo (VLM) points to vote on which mask corresponds to the target object.
    """

    def __init__(
        self,
        camera_serials,
        calib_file,
        max_depth=2.0,
        min_depth=0.1,
        sam_mask_generator=None,
        device="cuda",
    ):
        """
        Initialize RobotFrameMerger with SAM2 automatic masks + Molmo voting.

        Args:
            camera_serials: List of camera serial numbers
            calib_file: Path to calibration transforms file (transforms.npy)
            max_depth: Maximum depth in meters (default: 2.0)
            min_depth: Minimum depth in meters (default: 0.1)
            sam_mask_generator: SAM2AutomaticMaskGenerator instance (optional, for segmentation)
            device: Device for models ('cuda' or 'cpu')
        """
        self.camera_serials = camera_serials
        self.cameras = {}
        self.max_depth = max_depth
        self.min_depth = min_depth

        # Segmentation models
        self.sam_mask_generator = sam_mask_generator
        self.device = device

        # Load calibration transforms
        calib_path = str(calib_file)  # Handle both str and Path objects
        if not os.path.exists(calib_path):
            raise FileNotFoundError(f"Calibration file {calib_path} not found!")

        self.transforms = np.load(calib_path, allow_pickle=True).item()
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

        # Align depth to color frame — MUST match calibration which uses
        # rs.align(rs.stream.color) before deprojecting. Without this, point
        # clouds are in the depth/IR sensor frame instead of the color frame,
        # causing ~50mm offset (the color-to-IR baseline).
        align = rs.align(rs.stream.color)

        # Create post-processing filters (simplified approach like robot_calib)
        filters = [
            rs.spatial_filter(),
            rs.temporal_filter(),
            rs.hole_filling_filter(),
        ]

        self.cameras[serial_number] = {"pipeline": pipeline, "filters": filters, "align": align}

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
        """Capture point cloud from a single camera with optional SAM auto-mask + Molmo voting.

        When *text_prompt* is provided the pipeline is:
        1. Run SAM2AutomaticMaskGenerator on the RGB image → all candidate masks.
        2. Send the RGB image to Molmo asking it to point at the object → pixel points.
        3. Vote: select the auto-generated mask that contains the most Molmo points.
        4. Apply the selected mask to select 3D points belonging to the object.
        """
        camera_data = self.cameras[serial]
        pipeline = camera_data["pipeline"]
        filters = camera_data["filters"]
        align = camera_data["align"]

        # Capture frame and align depth to color (matches calibration)
        frames = pipeline.wait_for_frames()
        frames = align.process(frames)
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            logger.warning(f"Failed to capture from camera {serial}")
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

        # ----- SAM automatic masks + Molmo voting -----
        if text_prompt and self.sam_mask_generator:
            logger.info(f"Camera {serial}: Segmenting '{text_prompt}' via SAM auto-masks + Molmo voting...")

            # Convert BGR -> RGB for SAM and Molmo
            color_image_rgb = color_image[:, :, ::-1].copy()

            # Step 1: Generate ALL masks unconditionally with SAM2
            logger.info(f"  Running SAM2 automatic mask generation...")
            sam_results = self.sam_mask_generator.generate(color_image_rgb)
            logger.info(f"  SAM2 generated {len(sam_results)} masks")

            # Extract masks and scores from auto-generated results
            # Each result dict has: segmentation (HxW bool), predicted_iou, stability_score, area, bbox
            all_masks = np.array([r["segmentation"] for r in sam_results]) if sam_results else None
            all_scores = np.array([r["predicted_iou"] for r in sam_results]) if sam_results else None

            target_mask = None
            best_mask_idx = None

            # Step 2: Ask Molmo where the object is
            logger.info(f"  Querying Molmo for '{text_prompt}'...")
            molmo_pts = query_molmo_for_points(color_image_rgb, text_prompt)

            if not molmo_pts:
                logger.warning(f"  Molmo returned no points for '{text_prompt}'")
            elif all_masks is None or len(all_masks) == 0:
                logger.warning(f"  SAM2 generated no masks")
            else:
                # Step 3: Convert normalised coords to pixel coords
                pixel_points = molmo_points_to_pixel(molmo_pts, w, h)
                logger.info(f"  Molmo returned {len(pixel_points)} points, voting across {len(all_masks)} masks...")

                # Vote: pick the mask that contains the most Molmo points
                px_int = pixel_points.astype(int)
                px_x = np.clip(px_int[:, 0], 0, w - 1)
                px_y = np.clip(px_int[:, 1], 0, h - 1)

                point_counts = []
                for mi in range(len(all_masks)):
                    m = all_masks[mi].astype(bool)
                    count = m[px_y, px_x].sum()
                    point_counts.append(count)
                point_counts = np.array(point_counts)

                best_mask_idx = int(np.argmax(point_counts))
                target_mask = all_masks[best_mask_idx]
                logger.info(f"  Selected mask #{best_mask_idx}/{len(all_masks)} "
                             f"({point_counts[best_mask_idx]}/{len(pixel_points)} Molmo pts, "
                             f"iou={all_scores[best_mask_idx]:.3f}, "
                             f"area={sam_results[best_mask_idx]['area']}px)")

            # --- Debug visualisation ---
            self._show_segmentation_debug(
                serial, color_image_rgb, text_prompt,
                molmo_pts, all_masks, all_scores, best_mask_idx,
            )

            # Step 4: Apply mask to select 3D points
            if target_mask is not None:
                point_in_segment_mask = target_mask[v, u].astype(bool)
                segmentation_mask = valid_mask & point_in_segment_mask
                n_seg_pts = segmentation_mask.sum()
                logger.info(f"  Segmentation mask applied ({n_seg_pts} points in segment)")
            else:
                logger.warning(f"  No valid mask obtained, using all valid points")
                segmentation_mask = valid_mask
        else:
            segmentation_mask = valid_mask

        valid_points = points_3d[segmentation_mask]
        valid_colors = colors[segmentation_mask]

        logger.info(f"Camera {serial}: {len(valid_points)} valid points captured")

        return valid_points, valid_colors

    def _show_segmentation_debug(
        self, serial, color_image_rgb, text_prompt,
        molmo_pts, all_masks, all_scores, best_mask_idx,
    ):
        """Show interactive segmentation debug window with three panels:

        1. Original image + Molmo point(s) drawn as red dots
        2. All SAM auto-generated masks overlaid (colour-coded, up to 20 shown)
        3. The selected (best) mask highlighted in green with red contour

        The window blocks until the user presses a key or closes it.
        Also saves a PNG backup to ``segmentation_debug_cam_{serial}.png``.
        """
        try:
            import cv2 as _cv2
            from scipy import ndimage

            img_h, img_w = color_image_rgb.shape[:2]

            # ---- Panel 1: Original image + Molmo points ----
            panel_orig = np.ascontiguousarray(color_image_rgb[:, :, ::-1])  # RGB->BGR
            _cv2.putText(panel_orig, f"Camera {serial}", (10, 30),
                         _cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            _cv2.putText(panel_orig, f"'{text_prompt}'", (10, 60),
                         _cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            if molmo_pts:
                pixel_points = molmo_points_to_pixel(molmo_pts, img_w, img_h)
                for px, py in pixel_points:
                    _cv2.circle(panel_orig, (int(px), int(py)), 8, (0, 0, 255), -1)
                    _cv2.circle(panel_orig, (int(px), int(py)), 10, (255, 255, 255), 2)
                _cv2.putText(panel_orig, f"Molmo: {len(molmo_pts)} pt(s)", (10, 90),
                             _cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            else:
                _cv2.putText(panel_orig, "Molmo: NO POINTS", (10, 90),
                             _cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # ---- Panel 2: All SAM auto-generated masks ----
            if all_masks is not None and len(all_masks) > 0:
                overlay = color_image_rgb.copy()
                # Use a larger colour palette for many auto-generated masks
                np.random.seed(42)  # deterministic colours
                n_masks = len(all_masks)
                mask_colors = []
                for i in range(n_masks):
                    hue = int(180 * i / max(n_masks, 1))
                    hsv = np.uint8([[[hue, 200, 220]]])
                    bgr = _cv2.cvtColor(hsv, _cv2.COLOR_HSV2BGR)[0, 0]
                    mask_colors.append(tuple(int(c) for c in bgr[::-1]))  # BGR->RGB

                # Sort by area ascending so smaller masks are drawn on top
                order = np.argsort([all_masks[i].sum() for i in range(n_masks)])
                # Only draw up to 20 masks to keep viz readable
                drawn = 0
                for mi in reversed(order):
                    if drawn >= 20:
                        break
                    m = all_masks[mi].astype(bool)
                    c = np.array(mask_colors[mi], dtype=np.uint8)
                    overlay[m] = (overlay[m] * 0.45 + c * 0.55).astype(np.uint8)
                    contour = ndimage.binary_dilation(m) ^ m
                    overlay[contour] = c
                    drawn += 1

                panel_masks = np.ascontiguousarray(overlay[:, :, ::-1])
                label = f"SAM auto: {n_masks} masks"
                if best_mask_idx is not None:
                    label += f" | sel=#{best_mask_idx}"
                _cv2.putText(panel_masks, label, (10, 30),
                             _cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            else:
                panel_masks = np.ascontiguousarray(color_image_rgb[:, :, ::-1])
                _cv2.putText(panel_masks, "SAM: NO MASKS", (10, 30),
                             _cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # ---- Panel 3: Selected (best) mask ----
            if all_masks is not None and best_mask_idx is not None:
                selected = color_image_rgb.copy()
                mask_bool = all_masks[best_mask_idx].astype(bool)
                green = np.array([0, 255, 0], dtype=np.uint8)
                selected[mask_bool] = (selected[mask_bool] * 0.4 + green * 0.6).astype(np.uint8)
                contour = ndimage.binary_dilation(mask_bool) ^ mask_bool
                selected[contour] = [255, 0, 0]
                panel_selected = np.ascontiguousarray(selected[:, :, ::-1])
                sel_label = f"Selected: #{best_mask_idx} (iou={all_scores[best_mask_idx]:.3f})"
            else:
                panel_selected = np.ascontiguousarray(color_image_rgb[:, :, ::-1])
                sel_label = f"NO MASK for '{text_prompt}'"
            _cv2.putText(panel_selected, sel_label, (10, 30),
                         _cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # ---- Combine and show ----
            combined = np.hstack([panel_orig, panel_masks, panel_selected])

            # Save PNG backup
            out_path = f"segmentation_debug_cam_{serial}.png"
            _cv2.imwrite(out_path, combined)
            logger.info(f"  Debug viz saved to {out_path}")

            # Interactive window disabled — PNG backup is saved above.
            # To re-enable: uncomment the lines below.
            # window_name = f"Segmentation - Camera {serial} - Press any key"
            # _cv2.imshow(window_name, combined)
            # logger.info(f"  Showing segmentation debug for camera {serial}. Press any key to continue...")
            # _cv2.waitKey(0)
            # _cv2.destroyWindow(window_name)
        except Exception as viz_e:
            logger.warning(f"  Could not show segmentation visualization: {viz_e}")

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
                align = camera_data["align"]

                frames = pipeline.wait_for_frames()
                frames = align.process(frames)
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


def load_sam_mask_generator(sam2_checkpoint, sam2_config, device="cuda"):
    """
    Load SAM2 automatic mask generator for unconditional mask generation.

    Args:
        sam2_checkpoint: Path to SAM2 checkpoint file
        sam2_config: Path to SAM2 config YAML file (or just "sam2.1/sam2.1_hiera_l")
        device: Device to load model on ('cuda' or 'cpu')

    Returns:
        SAM2AutomaticMaskGenerator instance
    """
    print("Loading SAM2 model for automatic mask generation...")

    # Handle both Path objects and strings, convert to proper format for build_sam2
    from pathlib import Path
    sam2_config_path = Path(sam2_config)
    sam2_checkpoint_path = Path(sam2_checkpoint)

    # If given an absolute/relative path, need to change directory for Hydra
    if sam2_config_path.is_absolute() or str(sam2_config_path).startswith('..'):
        original_dir = os.getcwd()

        if sam2_config_path.is_absolute():
            config_root = sam2_config_path.parent.parent.parent
        else:
            config_root = Path(original_dir) / sam2_config_path.parent.parent.parent
            config_root = config_root.resolve()

        os.chdir(config_root)

        config_name = f"{sam2_config_path.parent.name}/{sam2_config_path.stem}"
        checkpoint_path = str(sam2_checkpoint_path.relative_to(config_root))

        sam2 = build_sam2(
            config_name, checkpoint_path, apply_postprocessing=False, device=device
        )

        os.chdir(original_dir)
    else:
        sam2 = build_sam2(
            str(sam2_config), str(sam2_checkpoint), apply_postprocessing=False, device=device
        )

    mask_generator = SAM2AutomaticMaskGenerator(
        model=sam2,
        points_per_side=32,
        points_per_batch=128,
        pred_iou_thresh=0.88,
        stability_score_thresh=0.95,
        min_mask_region_area=200.0,
    )
    print("SAM2 automatic mask generator loaded.")

    return mask_generator


# Keep backward-compatible alias
load_sam_predictor = load_sam_mask_generator


def main():
    """
    Main function - example usage when running as standalone script.

    When importing as a package, use load_sam_mask_generator() and RobotFrameMerger:

        from pathlib import Path
        from cap.segment_pc import RobotFrameMerger, load_sam_mask_generator

        # Load SAM2 automatic mask generator
        mask_gen = load_sam_mask_generator(
            sam2_checkpoint=Path("ckpt/sam2.1_hiera_large.pt"),
            sam2_config=Path("configs/sam2.1/sam2.1_hiera_l.yaml"),
            device="cuda"
        )

        # Create merger (SAM auto-masks + Molmo voting when text_prompt is given)
        merger = RobotFrameMerger(
            camera_serials=["327122079374", "317222072157"],
            calib_file=Path("transforms/transforms.npy"),
            sam_mask_generator=mask_gen,
            device="cuda"
        )

        # Capture with SAM auto-masks + Molmo voting segmentation
        points, colors = merger.capture_merged_pointcloud(text_prompt="cup")
    """
    from pathlib import Path

    # When run as script from cap/ directory, paths are relative to parent
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    # Configuration
    camera_serials = ["327122079374", "317222072157"]
    calib_file = project_root / "transforms" / "transforms.npy"
    max_depth = 2.0  # meters
    min_depth = 0.1  # meters

    # Segmentation settings
    TEXT_PROMPT = "tissue box"  # Set to None to disable segmentation
    sam2_checkpoint = project_root / "ckpt" / "sam2.1_hiera_large.pt"
    sam2_config = project_root / "configs" / "sam2.1" / "sam2.1_hiera_l.yaml"

    print(f"Camera serials: {camera_serials}")
    print(f"Calibration file: {calib_file}")
    print(f"Depth range: {min_depth}m to {max_depth}m")
    print(
        f"Depth exposure: {'AUTO' if DEPTH_EXPOSURE is None else f'{DEPTH_EXPOSURE} us'}"
    )
    print(
        f"RGB exposure: {'AUTO' if RGB_EXPOSURE is None else f'{RGB_EXPOSURE} us'}"
    )
    if TEXT_PROMPT:
        print(f"Segmentation: '{TEXT_PROMPT}'")

    # Initialize SAM2 automatic mask generator
    sam_mask_generator = None
    device = "cpu"

    if TEXT_PROMPT:
        print("\nInitializing SAM2 automatic mask generator...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")

        sam_mask_generator = load_sam_mask_generator(
            sam2_checkpoint=sam2_checkpoint,
            sam2_config=sam2_config,
            device=device,
        )
        print("SAM2 automatic mask generator initialized.")

    try:
        # Initialize merger
        print("\nInitializing robot frame merger...")
        merger = RobotFrameMerger(
            camera_serials=camera_serials,
            calib_file=calib_file,
            max_depth=max_depth,
            min_depth=min_depth,
            sam_mask_generator=sam_mask_generator,
            device=device,
        )

        # Capture merged point cloud
        merged_points, merged_colors = merger.capture_merged_pointcloud(
            text_prompt=TEXT_PROMPT
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
