"""Minimal live RealSense + Fast-FoundationStereo smoke test.

Opens one RealSense D4xx, captures a single synced IR1/IR2/color frame with
the IR projector OFF, runs Fast-FoundationStereo on the IR pair, and writes:

    output_ffs_test/<serial>_ffs.ply       — point cloud (color-aligned)
    output_ffs_test/<serial>_ffs_depth.npy — color-aligned depth (meters)
    output_ffs_test/<serial>_ffs_disp.npy  — raw FFS disparity

No transforms.npy / multi-camera / robot frame needed. If this writes a
non-empty PLY, FFS + your RealSense are working together.

Usage:
    uv run python test_ffs_realsense.py
    uv run python test_ffs_realsense.py --serial 327122079374
    uv run python test_ffs_realsense.py --viz       # open3d viewer after
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pyrealsense2 as rs

sys.path.insert(0, str(Path(__file__).parent))
from cap import ffs_backend as ffs


def pick_serial(requested: str | None) -> str:
    ctx = rs.context()
    devices = list(ctx.query_devices())
    if not devices:
        raise RuntimeError("No RealSense devices connected.")
    if requested is not None:
        for d in devices:
            if d.get_info(rs.camera_info.serial_number) == requested:
                return requested
        raise RuntimeError(
            f"Serial {requested!r} not found. Connected: "
            + ", ".join(d.get_info(rs.camera_info.serial_number) for d in devices))
    serial = devices[0].get_info(rs.camera_info.serial_number)
    print(f"Auto-picked first RealSense: {serial}")
    return serial


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--serial", default=None, help="RealSense serial (defaults to first found).")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--zfar", type=float, default=2.0, help="Clip depth beyond this (m).")
    ap.add_argument("--out-dir", default="output_ffs_test")
    ap.add_argument("--viz", action="store_true", help="Open the PLY in an open3d viewer after capture.")
    args = ap.parse_args()

    serial = pick_serial(args.serial)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Configure RealSense
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    ffs.enable_ffs_streams(config, width=args.width, height=args.height, fps=args.fps)
    profile = pipeline.start(config)
    print(f"Pipeline started: {args.width}x{args.height}@{args.fps}")

    try:
        # Grab intrinsics/extrinsics from the active profile
        rs_calib = ffs.query_intrinsics_extrinsics(profile)
        intr_ir = rs_calib["intr_ir"]
        intr_col = rs_calib["intr_col"]
        print(f"Color intrinsics:  fx={intr_col.fx:.1f}  fy={intr_col.fy:.1f}  "
              f"ppx={intr_col.ppx:.1f}  ppy={intr_col.ppy:.1f}")
        print(f"IR1   intrinsics:  fx={intr_ir.fx:.1f}  fy={intr_ir.fy:.1f}  "
              f"ppx={intr_ir.ppx:.1f}  ppy={intr_ir.ppy:.1f}")
        baseline = ffs.baseline_from_extrinsics(rs_calib["ext_ir1_ir2"])
        print(f"IR stereo baseline: {baseline*1000:.2f} mm")

        # Capture with emitter off
        depth_sensor = profile.get_device().first_depth_sensor()
        inputs = ffs.capture_emitter_off(pipeline, depth_sensor)
        ir1, ir2, color = inputs["ir1"], inputs["ir2"], inputs["color"]
        print(f"Captured frames: IR1={ir1.shape}  IR2={ir2.shape}  color={color.shape}")

        # Load FFS engine (lazy — first call loads checkpoint, ~few seconds)
        engine = ffs.FFSDepthEngine()
        print(f"FFS engine: weights={engine.model_path}")
        print("Running FFS disparity…")
        pts, colors, depth_color, disp = ffs.points_and_colors_from_ir(
            ir1, ir2, color, intr_ir, intr_col,
            rs_calib["ext_ir1_ir2"], rs_calib["ext_ir1_col"],
            engine, zfar=args.zfar)
        print(f"Disparity stats: min={disp.min():.2f}  max={disp.max():.2f}  "
              f"mean={disp.mean():.2f}")
        print(f"Aligned depth: shape={depth_color.shape}  "
              f"valid_pixels={(depth_color>0).sum():,}  "
              f"depth range=[{depth_color[depth_color>0].min():.3f}, "
              f"{depth_color.max():.3f}] m")
        print(f"Point cloud: N={len(pts):,}  "
              f"z=[{pts[:,2].min():.3f}, {pts[:,2].max():.3f}] m")

        # Save outputs
        ply_path = out_dir / f"{serial}_ffs.ply"
        depth_path = out_dir / f"{serial}_ffs_depth.npy"
        disp_path = out_dir / f"{serial}_ffs_disp.npy"

        # Write PLY using open3d
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
        o3d.io.write_point_cloud(str(ply_path), pcd)
        np.save(depth_path, depth_color)
        np.save(disp_path, disp)
        print(f"\nWrote {ply_path}")
        print(f"Wrote {depth_path}")
        print(f"Wrote {disp_path}")

        if args.viz:
            print("\nOpening in open3d viewer (close window to exit)…")
            o3d.visualization.draw_geometries(
                [pcd, o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)],
                window_name=f"FFS PC — {serial}")
    finally:
        pipeline.stop()


if __name__ == "__main__":
    main()
