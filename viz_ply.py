"""Quick open3d viewer for the PLYs written by test_ffs_realsense.py /
test_ffs_segment_pc.py.

Usage:
    uv run python viz_ply.py output_ffs_test/<serial>_ffs.ply
    uv run python viz_ply.py output_ffs_test/*.ply              # multiple
    uv run python viz_ply.py --no-axes output_ffs_test/foo.ply
    uv run python viz_ply.py --axes-size 0.2 output_ffs_test/foo.ply
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import open3d as o3d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="One or more .ply files (globs OK).")
    ap.add_argument("--axes-size", type=float, default=0.1,
                    help="Coordinate-frame axis length in meters (default 0.1).")
    ap.add_argument("--no-axes", action="store_true", help="Hide the origin axes.")
    ap.add_argument("--point-size", type=float, default=2.0)
    args = ap.parse_args()

    # Expand globs
    files: list[str] = []
    for p in args.paths:
        matched = sorted(glob.glob(p)) or [p]
        for m in matched:
            if Path(m).is_file():
                files.append(m)
            else:
                print(f"  (skip, not a file: {m})")
    if not files:
        print("No files matched.")
        sys.exit(1)

    geometries = []
    if not args.no_axes:
        geometries.append(
            o3d.geometry.TriangleMesh.create_coordinate_frame(size=args.axes_size))

    for fp in files:
        pcd = o3d.io.read_point_cloud(fp)
        n = np.asarray(pcd.points).shape[0]
        if n == 0:
            print(f"[!] {fp}: empty point cloud")
            continue
        pts = np.asarray(pcd.points)
        z = pts[:, 2]
        print(f"  {fp}: N={n:,}  "
              f"x=[{pts[:,0].min():.3f},{pts[:,0].max():.3f}]  "
              f"y=[{pts[:,1].min():.3f},{pts[:,1].max():.3f}]  "
              f"z=[{z.min():.3f},{z.max():.3f}] m")
        geometries.append(pcd)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="PLY viewer — " + ", ".join(Path(f).name for f in files))
    for g in geometries:
        vis.add_geometry(g)
    opt = vis.get_render_option()
    opt.point_size = args.point_size
    opt.background_color = np.array([0.05, 0.05, 0.05])
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()
