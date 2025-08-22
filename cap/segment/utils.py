import numpy as np
import pyrealsense2 as rs
from PIL import Image
import concurrent.futures
from openai import OpenAI
import io
import base64
import json
import numpy as np
import pyrealsense2 as rs
import open3d as o3d
import os
import time
import threading


client = OpenAI()

# Global lock for camera access
_camera_lock = threading.Lock()

def strip_markdown_code_fences(text):
    return text.strip().removeprefix('```').removesuffix('```').strip()

def call_vlm(frames, prompt, model="gpt-4.1-mini", temperature=0.8):
    """Call VLM API with the given frames and prompt"""
    content = [{"type": "text", "text": prompt}]
    
    # Convert each frame to base64 and add to content
    for frame in frames:
        image = Image.fromarray(frame)
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}})
    
    # Call the VLM API
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
        temperature=temperature
    )
    
    return response.choices[0].message.content

def capture_rs_rgb(serial_no=None):
    """Capture RGB image from RealSense and save to disk"""
    with _camera_lock:
        pipeline = None
        try:
            # Wait a bit to ensure camera is not busy
            time.sleep(0.05)
            
            # Check if device is available
            ctx = rs.context()
            devices = ctx.query_devices()
            if len(devices) == 0:
                print("No RealSense devices found")
                return None
            
            # Find the specific device if serial provided
            if serial_no:
                device_found = False
                for dev in devices:
                    if dev.get_info(rs.camera_info.serial_number) == serial_no:
                        device_found = True
                        break
                if not device_found:
                    print(f"Device with serial {serial_no} not found")
                    return None
            
            pipeline = rs.pipeline()
            config = rs.config()
            
            # Enable specific camera if serial provided
            if serial_no:
                config.enable_device(serial_no)
            
            # Configure color stream
            config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
            
            # Start pipeline
            profile = pipeline.start(config)
            
            # Get the color sensor and configure exposure/white balance
            device = profile.get_device()
            color_sensor = device.first_color_sensor()
            
            # Set exposure and gain
            color_sensor.set_option(rs.option.enable_auto_exposure, 0)  # Disable auto exposure
            color_sensor.set_option(rs.option.exposure, 120)  # Set exposure (microseconds)
            color_sensor.set_option(rs.option.gain, 0)  # Set gain
            
            # Set white balance
            color_sensor.set_option(rs.option.enable_auto_white_balance, 0)  # Disable auto white balance
            color_sensor.set_option(rs.option.white_balance, 3500)  # Set white balance (Kelvin)
            
            # Wait for a few frames to stabilize
            for _ in range(5):
                pipeline.wait_for_frames()
            
            # Capture frame
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            
            if not color_frame:
                print("Failed to capture color frame")
                return None
            
            # Convert to numpy array (BGR format)
            bgr_image = np.asanyarray(color_frame.get_data())
            
            # Convert BGR to RGB for saving
            rgb_image = bgr_image[..., ::-1]
            
            return rgb_image
            
        except Exception as e:
            print(f"Error in capture_rs_rgb: {e}")
            return None
        finally:
            if pipeline:
                try:
                    pipeline.stop()
                except:
                    pass
                time.sleep(0.1)  # Brief delay to ensure cleanup

def get_obs_objects(camera_serials, model):
    obs_prompt = """
    name all objects in the scene that you see in the images. For each object, just name it as concisely as possible while describe the object while still containing good descriptive adjectives, for instance:
    - corn on the cob should --> yellow corn
    - digital camera with wrist strap --> pink digital camera
    - blue plastic reusable cup --> blue cup 
    If its a named object, use the generic name:
    - Keurig machine --> coffee machine
    - iPhone 14 Pro Max --> smartphone
    - Sony Camera --> camera
    Output your list of objects as a directly parseable JSON array. Here's an example of the output format:
    ["corn", "digital camera", "cup"]
    Do not output anything else, no explanations, no additional text, just the JSON array that should be directly parseable.

    Always ignore the robot if its visible in the scene. Never name it as one of the objects. But you need to name every other object that you see in the scene.
    """
    
    def process_camera(serial):
        try:
            image = capture_rs_rgb(serial)
            if image is None:
                return serial, []
            response = call_vlm([image], obs_prompt, model, temperature=0.0)
            response = strip_markdown_code_fences(response)
            return serial, json.loads(response)
        except Exception as e:
            print(f"Error processing camera {serial}: {e}")
            return serial, []
    
    # Process cameras sequentially to avoid resource conflicts
    cam_obs_objs = {}
    for serial in camera_serials:
        serial_result, data = process_camera(serial)
        cam_obs_objs[serial_result] = data
        time.sleep(0.2)  # Brief delay between cameras
    
    return cam_obs_objs

def capture_rs_pc(camera_serial, calib_path, max_depth=2.0, min_depth=0.1, max_points=100000, 
                  calib_units="mm", point_cloud_units="m", icp_path=None, warmup_frames=10):
    """
    Capture and return a transformed point cloud from a specific RealSense camera.
    
    Args:
        camera_serial (str): Serial number of the camera
        calib_path (str): Path to calibration transforms file
        max_depth (float): Maximum depth threshold for filtering
        min_depth (float): Minimum depth threshold for filtering
        max_points (int): Maximum number of points to return (downsampling)
        calib_units (str): Units used in calibration data ("mm", "m", "cm", "inch")
        point_cloud_units (str): Units for output point cloud ("mm", "m", "cm", "inch")
        icp_path (str): Optional path to ICP refinement transforms
        warmup_frames (int): Number of frames to skip for camera warmup
        
    Returns:
        tuple: (points_3d, colors) - transformed 3D points and corresponding colors
               Returns (None, None) if capture fails
    """
    
    with _camera_lock:
        pipeline = None
        try:
            # Wait a bit to ensure camera is not busy
            time.sleep(0.05)
            
            # Load calibration data
            if not os.path.exists(calib_path):
                raise FileNotFoundError(f"Calibration file {calib_path} not found!")
            
            transforms = np.load(calib_path, allow_pickle=True).item()
            
            if camera_serial not in transforms:
                raise ValueError(f"No calibration data for camera {camera_serial}")
            
            # Unit conversion setup
            unit_factors = {
                'mm': 0.001,  # mm to meters
                'm': 1.0,     # meters to meters
                'cm': 0.01,   # cm to meters
                'inch': 0.0254, # inches to meters
            }
            
            if calib_units not in unit_factors or point_cloud_units not in unit_factors:
                raise ValueError(f"Unsupported units. Supported: {list(unit_factors.keys())}")
            
            unit_scale = unit_factors[calib_units] / unit_factors[point_cloud_units]
            
            # Load and convert calibration transform
            tcr = transforms[camera_serial]["tcr"].copy()
            tcr[:3, 3] *= unit_scale  # Scale translation components
            
            # Load ICP transforms if available
            icp_tf = None
            if icp_path and os.path.exists(icp_path):
                icp_transforms = np.load(icp_path, allow_pickle=True).item()
                if camera_serial in icp_transforms:
                    icp_tf = icp_transforms[camera_serial].copy()
                    icp_tf[:3, 3] *= unit_scale  # Scale translation components
            
            # Check if device is available
            ctx = rs.context()
            devices = ctx.query_devices()
            device_found = False
            for dev in devices:
                if dev.get_info(rs.camera_info.serial_number) == camera_serial:
                    device_found = True
                    break
            if not device_found:
                print(f"Camera {camera_serial} not found")
                return None, None
            
            # Initialize pipeline
            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_device(camera_serial)
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            
            profile = pipeline.start(config)
            pc = rs.pointcloud()
            
            # Warmup
            for _ in range(warmup_frames):
                pipeline.wait_for_frames()
            
            # Add stabilization delay
            time.sleep(0.1)
            
            # Capture frames
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            
            if not depth_frame or not color_frame:
                return None, None
            
            # Create point cloud using RealSense
            pc.map_to(color_frame)
            points = pc.calculate(depth_frame)
            
            # Extract points and colors
            vtx = np.asanyarray(points.get_vertices())
            tex = np.asanyarray(points.get_texture_coordinates())
            
            points_3d = np.column_stack((vtx['f0'], vtx['f1'], vtx['f2']))
            
            # Get colors
            color_image = np.asanyarray(color_frame.get_data())
            h, w = color_image.shape[:2]
            u = np.clip((tex['f0'] * w).astype(int), 0, w-1)
            v = np.clip((tex['f1'] * h).astype(int), 0, h-1)
            
            colors = color_image[v, u] / 255.0  # Normalize to [0,1]
            colors = colors[:, [2, 1, 0]]  # BGR to RGB
            
            # Filter valid points
            valid_mask = (
                (points_3d[:, 2] > min_depth) & 
                (points_3d[:, 2] < max_depth) & 
                ~np.isnan(points_3d).any(axis=1) &
                ~np.isinf(points_3d).any(axis=1)
            )
            
            valid_points = points_3d[valid_mask]
            valid_colors = colors[valid_mask]
            
            # Downsample if too many points
            if len(valid_points) > max_points:
                indices = np.random.choice(len(valid_points), max_points, replace=False)
                valid_points = valid_points[indices]
                valid_colors = valid_colors[indices]
            
            # Transform to robot coordinates
            # Convert to homogeneous coordinates
            ones = np.ones((valid_points.shape[0], 1))
            points_homo = np.hstack([valid_points, ones])
            
            # Apply main calibration transform (camera to robot)
            points_robot = (tcr @ points_homo.T).T[:, :3]
            
            # Apply ICP refinement if available
            if icp_tf is not None:
                ones = np.ones((points_robot.shape[0], 1))
                points_homo = np.hstack([points_robot, ones])
                points_robot = (icp_tf @ points_homo.T).T[:, :3]
            
            return points_robot, valid_colors
            
        except Exception as e:
            print(f"Error capturing from camera {camera_serial}: {e}")
            return None, None
            
        finally:
            if pipeline:
                try:
                    pipeline.stop()
                except:
                    pass
                time.sleep(0.1)  # Brief delay to ensure cleanup
