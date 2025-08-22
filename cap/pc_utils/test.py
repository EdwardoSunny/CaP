import numpy as np
import pyrealsense2 as rs
import open3d as o3d

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Enable streams
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

# Create point cloud object
pc = rs.pointcloud()

# Warm up camera
print("Warming up camera...")
for _ in range(30):
    pipeline.wait_for_frames()

print("Capturing frame...")

try:
    # Capture a single frame
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()

    if not depth_frame or not color_frame:
        print("Failed to capture frames")
        exit()

    # Generate point cloud
    pc.map_to(color_frame)
    points = pc.calculate(depth_frame)

    # Get vertices and texture coordinates
    vtx = np.asanyarray(points.get_vertices())
    tex = np.asanyarray(points.get_texture_coordinates())

    # Convert to numpy arrays
    vertices = np.array([[v[0], v[1], v[2]] for v in vtx])

    # Get color data
    color_image = np.asanyarray(color_frame.get_data())

    # Map colors to points
    colors = []
    for t in tex:
        x = int(t[0] * color_image.shape[1])
        y = int(t[1] * color_image.shape[0])
        x = np.clip(x, 0, color_image.shape[1] - 1)
        y = np.clip(y, 0, color_image.shape[0] - 1)
        # Convert BGR to RGB and normalize
        colors.append(
            [
                color_image[y, x, 2] / 255.0,
                color_image[y, x, 1] / 255.0,
                color_image[y, x, 0] / 255.0,
            ]
        )

    colors = np.array(colors)

    # Filter out invalid points (where z = 0)
    valid_indices = vertices[:, 2] != 0
    vertices = vertices[valid_indices]
    colors = colors[valid_indices]

    print(f"Point cloud has {len(vertices)} valid points")

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Visualize
    print("Displaying point cloud. Close the window to exit.")
    o3d.visualization.draw_geometries(
        [pcd], window_name="RealSense Point Cloud", width=1280, height=720
    )

finally:
    # Cleanup
    pipeline.stop()
    print("Done.")
