"""Fast-FoundationStereo backend for replacing rs.pointcloud()/rs.align depth.

Provides a drop-in replacement for the on-device active-stereo depth path used
by ``rs.align(rs.stream.color)`` + ``rs.pointcloud()``: capture an IR pair +
color frame from a D4xx, run the Fast-FoundationStereo passive matcher, and
return points expressed in the **color camera frame** with per-pixel RGB
colors — same layout as the original RealSense flow.

Usage from a calling module:

    from calib_tools.utils import ffs_backend as ffs

    engine = ffs.FFSDepthEngine()              # lazy-loads the model on first use
    inputs = ffs.capture_emitter_off(pipeline, depth_sensor)
    pts, colors, depth_color, disp = ffs.points_and_colors_from_ir(
        inputs['ir1'], inputs['ir2'], inputs['color'],
        intr_ir, intr_col, ext_ir1_ir2, ext_ir1_col, engine,
        zfar=2.0)

The helpers can also be wired into a custom pipeline directly:
    disp        = engine.disparity(ir1, ir2)
    depth_ir    = ffs.disp_to_depth(disp, intr_ir.fx, baseline)
    depth_color = ffs.align_depth_ir_to_color(depth_ir, intr_ir, intr_col,
                                              ext_ir1_col, color_image.shape)
    pts, colors = ffs.deproject_color_aligned(depth_color, intr_col, color_image)
"""
from __future__ import annotations

import os
import sys
import threading
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pyrealsense2 as rs
import torch


# ---------------------------------------------------------------------------
# Locate the Fast-FoundationStereo clone and add it to sys.path
# ---------------------------------------------------------------------------

def _find_ffs_root() -> Path:
    """Resolve a Fast-FoundationStereo clone — look near this file and at env."""
    here = Path(__file__).resolve()
    candidates: list[Path] = []
    for env_var in ('FFS_DIR', 'FAST_FOUNDATION_STEREO'):
        if os.environ.get(env_var):
            candidates.append(Path(os.environ[env_var]))
    for ancestor in here.parents:
        candidates.append(ancestor / 'Fast-FoundationStereo')
    for c in candidates:
        if (c / 'core' / 'foundation_stereo.py').exists():
            return c
    raise RuntimeError(
        'Cannot find Fast-FoundationStereo. Set FFS_DIR or place a clone next '
        'to the project root.')


FFS_ROOT = _find_ffs_root()
if str(FFS_ROOT) not in sys.path:
    sys.path.insert(0, str(FFS_ROOT))

from core.utils.utils import InputPadder  # noqa: E402  (relies on sys.path)
from Utils import AMP_DTYPE  # noqa: E402

_DEFAULT_WEIGHTS = FFS_ROOT / 'weights' / '23-36-37' / 'model_best_bp2_serialize.pth'


# ---------------------------------------------------------------------------
# Inference engine
# ---------------------------------------------------------------------------

class FFSDepthEngine:
    """Lazy-loaded FFS inference helper. Process-global cache so multiple
    callers share the same loaded model."""

    _model = None
    _model_path: Optional[str] = None
    _lock = threading.Lock()

    def __init__(self, model_path: Optional[str] = None,
                 valid_iters: int = 8, max_disp: int = 192,
                 device: str = 'cuda', amp_dtype: torch.dtype = AMP_DTYPE):
        self.model_path = str(model_path or _DEFAULT_WEIGHTS)
        self.valid_iters = valid_iters
        self.max_disp = max_disp
        self.device = device
        self.amp_dtype = amp_dtype

    def _ensure_model(self):
        with FFSDepthEngine._lock:
            if (FFSDepthEngine._model is None
                    or FFSDepthEngine._model_path != self.model_path):
                model = torch.load(self.model_path, map_location='cpu',
                                   weights_only=False)
                model.to(self.device).eval()
                FFSDepthEngine._model = model
                FFSDepthEngine._model_path = self.model_path
            FFSDepthEngine._model.args.valid_iters = self.valid_iters
            FFSDepthEngine._model.args.max_disp = self.max_disp
        return FFSDepthEngine._model

    @torch.no_grad()
    def disparity(self, ir1: np.ndarray, ir2: np.ndarray) -> np.ndarray:
        """Run FFS on a Y8 IR pair, return float32 disparity (H, W)."""
        model = self._ensure_model()
        if ir1.ndim == 2:
            ir1 = np.repeat(ir1[..., None], 3, axis=2)
            ir2 = np.repeat(ir2[..., None], 3, axis=2)
        H, W = ir1.shape[:2]
        t0 = torch.as_tensor(ir1).to(self.device).float()[None].permute(0, 3, 1, 2)
        t1 = torch.as_tensor(ir2).to(self.device).float()[None].permute(0, 3, 1, 2)
        padder = InputPadder(t0.shape, divis_by=32, force_square=False)
        t0, t1 = padder.pad(t0, t1)
        with torch.amp.autocast('cuda', enabled=True, dtype=self.amp_dtype):
            disp = model.forward(t0, t1, iters=self.valid_iters,
                                 test_mode=True,
                                 optimize_build_volume='pytorch1')
        disp = padder.unpad(disp.float()).cpu().numpy().reshape(H, W)
        return disp.clip(0, None)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def baseline_from_extrinsics(ext_ir1_ir2) -> float:
    """Stereo baseline (m) from the IR1 → IR2 extrinsics record."""
    return float(abs(ext_ir1_ir2.translation[0]))


def disp_to_depth(disp: np.ndarray, fx: float, baseline: float,
                  zfar: Optional[float] = None) -> np.ndarray:
    depth = np.zeros_like(disp, dtype=np.float32)
    valid = disp > 0
    depth[valid] = fx * baseline / disp[valid]
    if zfar is not None:
        depth[depth > zfar] = 0.0
    return depth


def _rotation_matrix(ext) -> np.ndarray:
    # librealsense stores rotation column-major.
    return np.asarray(ext.rotation, dtype=np.float32).reshape(3, 3).T


def align_depth_ir_to_color(depth_ir: np.ndarray, intr_ir, intr_col,
                            ext_ir1_col, color_shape) -> np.ndarray:
    """Forward-project an IR1-frame depth map into the color camera frame
    with z-buffering. Returns (Hc, Wc) float32 depth aligned to the color
    sensor; pixels with no source IR depth are 0."""
    Hc, Wc = color_shape[:2]
    H, W = depth_ir.shape
    R = _rotation_matrix(ext_ir1_col)
    t = np.asarray(ext_ir1_col.translation, dtype=np.float32).reshape(3)

    valid = depth_ir > 0
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    z_i = depth_ir.astype(np.float32)
    x_i = (u - intr_ir.ppx) * z_i / intr_ir.fx
    y_i = (v - intr_ir.ppy) * z_i / intr_ir.fy
    P_i = np.stack([x_i, y_i, z_i], axis=-1).reshape(-1, 3)
    P_c = P_i @ R.T + t

    z_c = P_c[:, 2]
    valid_proj = z_c > 1e-3
    safe_z = np.where(valid_proj, z_c, 1.0)
    u_c = intr_col.fx * P_c[:, 0] / safe_z + intr_col.ppx
    v_c = intr_col.fy * P_c[:, 1] / safe_z + intr_col.ppy
    u_ci = np.round(u_c).astype(np.int64)
    v_ci = np.round(v_c).astype(np.int64)
    in_bounds = (u_ci >= 0) & (u_ci < Wc) & (v_ci >= 0) & (v_ci < Hc)
    keep = valid.reshape(-1) & valid_proj & in_bounds

    flat = np.full(Hc * Wc, np.inf, dtype=np.float32)
    np.minimum.at(flat, v_ci[keep] * Wc + u_ci[keep], z_c[keep])
    depth_aligned = flat.reshape(Hc, Wc)
    depth_aligned[depth_aligned == np.inf] = 0.0
    return depth_aligned


def deproject_color_aligned(depth_aligned: np.ndarray, intr_col,
                            color_image_bgr: np.ndarray
                            ) -> Tuple[np.ndarray, np.ndarray]:
    """Deproject a color-aligned depth map. Returns ``(points, colors)``
    flattened to (H*W, 3): points in the color camera frame (m), colors as
    float32 RGB in [0, 1] using the same BGR→RGB swap the original
    ``rs.pointcloud()`` flow used (``color[v, u][:, ::-1] / 255.0``).

    Pixels with depth == 0 land at (0, 0, 0) — keep the existing min/max
    depth filters in the caller to drop them."""
    Hc, Wc = depth_aligned.shape
    # Match legacy rs.pointcloud() output dtypes exactly: float32 points,
    # float64 colors (the original `color[v, u][:, ::-1] / 255.0` divides a
    # uint8 array by a Python float so it promotes to float64).
    u = np.arange(Wc, dtype=np.float32)
    v = np.arange(Hc, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)
    z = depth_aligned.astype(np.float32)
    x = (uu - np.float32(intr_col.ppx)) * z / np.float32(intr_col.fx)
    y = (vv - np.float32(intr_col.ppy)) * z / np.float32(intr_col.fy)
    pts = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    color_rgb = color_image_bgr[:, :, ::-1] / 255.0
    colors = color_rgb.reshape(-1, 3)
    return pts, colors


def points_and_colors_from_ir(ir1, ir2, color_bgr, intr_ir, intr_col,
                              ext_ir1_ir2, ext_ir1_col, engine: FFSDepthEngine,
                              zfar: float = 5.0):
    """End-to-end: IR pair + color image → (Nx3 points in color frame,
    Nx3 RGB in [0, 1], Hc×Wc aligned depth, H×W FFS disparity)."""
    disp = engine.disparity(ir1, ir2)
    baseline = baseline_from_extrinsics(ext_ir1_ir2)
    depth_ir = disp_to_depth(disp, intr_ir.fx, baseline, zfar=zfar)
    depth_color = align_depth_ir_to_color(depth_ir, intr_ir, intr_col,
                                          ext_ir1_col, color_bgr.shape)
    pts, colors = deproject_color_aligned(depth_color, intr_col, color_bgr)
    return pts, colors, depth_color, disp


# ---------------------------------------------------------------------------
# RealSense capture helpers
# ---------------------------------------------------------------------------

def enable_ffs_streams(config: rs.config, width: int = 640, height: int = 480,
                       fps: int = 30, color_format=rs.format.bgr8):
    """Add the streams FFS needs to an rs.config: IR1, IR2 and color.

    Caller must have already enabled the device. Depth can additionally be
    enabled by the caller if the on-device active-stereo depth is also wanted.
    """
    config.enable_stream(rs.stream.infrared, 1, width, height, rs.format.y8, fps)
    config.enable_stream(rs.stream.infrared, 2, width, height, rs.format.y8, fps)
    config.enable_stream(rs.stream.color, width, height, color_format, fps)


def query_intrinsics_extrinsics(profile: rs.pipeline_profile):
    """Return a dict with intr_ir, intr_col, ext_ir1_ir2, ext_ir1_col."""
    ir1_p = profile.get_stream(rs.stream.infrared, 1).as_video_stream_profile()
    ir2_p = profile.get_stream(rs.stream.infrared, 2).as_video_stream_profile()
    col_p = profile.get_stream(rs.stream.color).as_video_stream_profile()
    return {
        'intr_ir': ir1_p.get_intrinsics(),
        'intr_col': col_p.get_intrinsics(),
        'ext_ir1_ir2': ir1_p.get_extrinsics_to(ir2_p),
        'ext_ir1_col': ir1_p.get_extrinsics_to(col_p),
    }


def capture_emitter_off(pipeline: rs.pipeline, depth_sensor: rs.depth_sensor,
                        warmup: int = 15, restore_emitter: bool = True):
    """Toggle the IR projector off, snapshot one synced frame, restore.

    The pipeline must already have IR1, IR2, and color streams active.
    Returns a dict with ir1, ir2, color (numpy arrays).
    """
    if not depth_sensor.supports(rs.option.emitter_enabled):
        raise RuntimeError('Depth sensor does not expose emitter_enabled')
    initial = depth_sensor.get_option(rs.option.emitter_enabled)
    depth_sensor.set_option(rs.option.emitter_enabled, 0.0)
    try:
        for _ in range(warmup):
            pipeline.wait_for_frames()
        f = pipeline.wait_for_frames()
        ir1 = np.asanyarray(f.get_infrared_frame(1).get_data()).copy()
        ir2 = np.asanyarray(f.get_infrared_frame(2).get_data()).copy()
        color = np.asanyarray(f.get_color_frame().get_data()).copy()
    finally:
        if restore_emitter:
            depth_sensor.set_option(rs.option.emitter_enabled, initial)
    return {'ir1': ir1, 'ir2': ir2, 'color': color}
