import numpy as np
import pyrealsense2 as rs

# Get connected camera serials
ctx = rs.context()
devices = ctx.query_devices()
serials = [dev.get_info(rs.camera_info.serial_number) for dev in devices]

# Create pipelines for each camera
pipelines = []
for serial in serials:  # Use all connected cameras
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)
    pipelines.append(pipeline)

# Create point cloud object
pc = rs.pointcloud()

# Warm up cameras
for _ in range(30):
    for pipeline in pipelines:
        pipeline.wait_for_frames()

# Capture and save point clouds
for i, pipeline in enumerate(pipelines):
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    
    if depth_frame and color_frame:
        # Generate point cloud
        pc.map_to(color_frame)
        points = pc.calculate(depth_frame)
        
        # Save as PLY file
        points.export_to_ply(f"pointcloud_{i}.ply", color_frame)
        print(f"Saved: pointcloud_{i}.ply")

# Cleanup
for pipeline in pipelines:
    pipeline.stop()
