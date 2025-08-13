import open3d as o3d
import glob

# Load all point cloud files
ply_files = glob.glob("pointcloud_*.ply")

if not ply_files:
    print("No point cloud files found!")
    exit()

# Load and visualize each point cloud
point_clouds = []
for i, file in enumerate(ply_files):
    print(f"Loading {file}...")
    pcd = o3d.io.read_point_cloud(file)
    
    # Optional: downsample for better performance
    pcd = pcd.voxel_down_sample(voxel_size=0.005)
    
    # Optional: remove outliers
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    
    point_clouds.append(pcd)

# Visualize separately
for i, pcd in enumerate(point_clouds):
    print(f"Visualizing point cloud {i}...")
    o3d.visualization.draw_geometries([pcd], window_name=f"Point Cloud {i}")

# Or visualize all together (uncomment line below)
# o3d.visualization.draw_geometries(point_clouds)
