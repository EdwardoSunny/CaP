"""Parity check for the FFS-backed segment_pc pipeline.

Runs RobotFrameMerger.capture_merged_pointcloud() without SAM/Molmo (no
text prompt) and prints stats so the FFS-backed output can be compared to
the previous rs.pointcloud() flow.
"""
import os
import sys
import numpy as np
import open3d as o3d

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cap.segment_pc import RobotFrameMerger


def main():
    serials = ['327122079374', '317222072157']
    calib_file = os.path.join(os.path.dirname(__file__), 'transforms',
                              'transforms.npy')
    out_dir = os.path.join(os.path.dirname(__file__), 'output_ffs_test')
    os.makedirs(out_dir, exist_ok=True)

    merger = RobotFrameMerger(
        camera_serials=serials,
        calib_file=calib_file,
        max_depth=2.0,
        min_depth=0.1,
    )

    for serial in serials:
        result = merger.capture_single_camera(serial)
        if result is None:
            print(f"[{serial}] FAILED")
            continue
        pts, colors = result
        z = pts[:, 2]
        print(f"[{serial}] N={len(pts):,}  z={z.min():.3f}/{np.median(z):.3f}/{z.max():.3f} m  "
              f"colors=[{colors.min():.3f}, {colors.max():.3f}]  "
              f"shape pts={pts.shape}, colors={colors.shape}")

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
        o3d.io.write_point_cloud(f"{out_dir}/{serial}_camera.ply", pcd)

    merged_pts, merged_colors = merger.capture_merged_pointcloud()
    if merged_pts is not None:
        print(f"[merged] N={len(merged_pts):,}  bounds: x=[{merged_pts[:,0].min():.3f},{merged_pts[:,0].max():.3f}]  "
              f"y=[{merged_pts[:,1].min():.3f},{merged_pts[:,1].max():.3f}]  "
              f"z=[{merged_pts[:,2].min():.3f},{merged_pts[:,2].max():.3f}]")
        m_pcd = o3d.geometry.PointCloud()
        m_pcd.points = o3d.utility.Vector3dVector(merged_pts.astype(np.float64))
        m_pcd.colors = o3d.utility.Vector3dVector(merged_colors.astype(np.float64))
        o3d.io.write_point_cloud(f"{out_dir}/merged_robot.ply", m_pcd)
    print(f"Saved to {out_dir}")


if __name__ == '__main__':
    main()
