from segment.gen_masks import get_masks
from segment.utils import capture_rs_pc, capture_rs_rgb, get_obs_objects
import open3d as o3d

camera_serials = ["317422074281", "317422075456"]
calibration_file = "/home/u-ril/edward/CaP/camera_calibration/calib/transforms.npy"

objects_dict = get_obs_objects(camera_serials, "gpt-4.1-mini")
object_name_str = ", ".join(objects_dict[camera_serials[0]])

frame = capture_rs_rgb(camera_serials[0])

masks = get_masks(frame, object_name_str)

points, colors = capture_rs_pc(camera_serials[0], calibration_file)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors)

pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

# Optional: Downsample for better performance if point cloud is very large
# pcd = pcd.voxel_down_sample(voxel_size=0.01)

o3d.visualization.draw_geometries(
    [pcd],
    window_name="Point Cloud Visualization",
    width=1024,
    height=768,
    left=50,
    top=50,
)
