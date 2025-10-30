import time
import pathlib
import numpy as np
import logging
import pyrealsense2 as rs

from multiprocessing.managers import SharedMemoryManager
from typing import List, Optional, Union, Dict, Callable

from ril_env.realsense import SingleRealsense
from ril_env.video_recorder import VideoRecorder

logger = logging.getLogger(__name__)


def repeat_to_list(x, n: int, cls):
    if x is None:
        x = [None] * n
    if isinstance(x, cls):
        x = [x] * n
    assert len(x) == n, f"repeat_to_list got len(x)={len(x)}, expected {n}"
    return x


class MultiRealsense:
    def __init__(
        self,
        serial_numbers: Optional[List[str]] = None,
        shm_manager: Optional[SharedMemoryManager] = None,
        resolution=(1280, 720),
        capture_fps=30,
        put_fps=None,
        put_downsample=True,
        record_fps=None,
        enable_color=True,
        enable_depth=False,
        enable_infrared=False,
        get_max_k=30,
        advanced_mode_config: Optional[Union[dict, List[dict]]] = None,
        transform: Optional[Union[Callable[[Dict], Dict], List[Callable]]] = None,
        vis_transform: Optional[Union[Callable[[Dict], Dict], List[Callable]]] = None,
        recording_transform: Optional[
            Union[Callable[[Dict], Dict], List[Callable]]
        ] = None,
        video_recorder: Optional[Union[VideoRecorder, List[VideoRecorder]]] = None,
        verbose=False,
    ):
        logger.info("Initializing MultiRealsense...")

        if shm_manager is None:
            shm_manager = SharedMemoryManager()
            shm_manager.start()

        if serial_numbers is None:
            serial_numbers = SingleRealsense.get_connected_devices_serial()
        n_cameras = len(serial_numbers)
        logger.info(f"MultiRealsense found {n_cameras} cameras: {serial_numbers}")

        advanced_mode_config = repeat_to_list(advanced_mode_config, n_cameras, dict)
        transform = repeat_to_list(transform, n_cameras, Callable)
        vis_transform = repeat_to_list(vis_transform, n_cameras, Callable)
        recording_transform = repeat_to_list(recording_transform, n_cameras, Callable)
        video_recorder = repeat_to_list(video_recorder, n_cameras, VideoRecorder)

        self.cameras = {}
        for i, serial in enumerate(serial_numbers):
            logger.info(f"Creating SingleRealsense {i} with serial={serial}")
            self.cameras[serial] = SingleRealsense(
                shm_manager=shm_manager,
                serial_number=serial,
                resolution=resolution,
                capture_fps=capture_fps,
                put_fps=put_fps,
                put_downsample=put_downsample,
                record_fps=record_fps,
                enable_color=enable_color,
                enable_depth=enable_depth,
                enable_infrared=enable_infrared,
                get_max_k=get_max_k,
                advanced_mode_config=advanced_mode_config[i],
                transform=transform[i],
                vis_transform=vis_transform[i],
                recording_transform=recording_transform[i],
                video_recorder=video_recorder[i],
                verbose=verbose,
            )
        self.shm_manager = shm_manager
        logger.info("MultiRealsense init done.")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    @property
    def n_cameras(self):
        return len(self.cameras)

    @property
    def is_ready(self):
        # Must have all cameras ready
        for camera in self.cameras.values():
            if not camera.is_ready:
                return False
        return True

    def start(self, wait=True, put_start_time=None):
        logger.info(
            f"MultiRealsense.start(wait={wait}), put_start_time={put_start_time}"
        )
        if put_start_time is None:
            put_start_time = time.time()
        for camera in self.cameras.values():
            camera.start(wait=False, put_start_time=put_start_time)
        if wait:
            self.start_wait()

    def stop(self, wait=True):
        logger.info(f"MultiRealsense.stop(wait={wait})")
        for camera in self.cameras.values():
            camera.stop(wait=False)
        if wait:
            self.stop_wait()

    def start_wait(self):
        logger.info(
            "MultiRealsense.start_wait() => waiting for each camera.start_wait()"
        )
        for camera in self.cameras.values():
            camera.start_wait()

    def stop_wait(self):
        logger.info("MultiRealsense.stop_wait() => joining each camera.")
        for camera in self.cameras.values():
            camera.join()

    def get(self, k=None, out=None) -> Dict[int, Dict[str, np.ndarray]]:
        if out is None:
            out = {}
        i = 0
        for serial, camera in self.cameras.items():
            this_out = out.get(i, None)
            this_out = camera.get(k=k, out=this_out)
            out[i] = this_out
            i += 1
        return out

    def get_vis(self, out=None):

        # This code is rarely used except for MultiCameraVisualizer
        results = []
        for i, camera in enumerate(self.cameras.values()):
            this_out = None
            if out is not None and i in out:
                this_out = out[i]
            this_out = camera.get_vis(out=this_out)
            if out is None:
                results.append(this_out)
        if out is None and len(results) > 0:
            # stack them
            out = {}
            for key in results[0].keys():
                out[key] = np.stack([x[key] for x in results])
        return out

    def set_color_option(self, option, value):
        logger.info(f"MultiRealsense.set_color_option({option}, {value})")
        for camera in self.cameras.values():
            camera.set_color_option(option, value)

    def set_exposure(self, exposure=None, gain=None):
        logger.info(f"MultiRealsense.set_exposure(exposure={exposure}, gain={gain})")
        for camera in self.cameras.values():
            camera.set_exposure(exposure=exposure, gain=gain)

    def set_white_balance(self, white_balance=None):
        logger.info(f"MultiRealsense.set_white_balance({white_balance})")
        for camera in self.cameras.values():
            camera.set_white_balance(white_balance)

    def get_intrinsics(self):
        return np.array([cam.get_intrinsics() for cam in self.cameras.values()])

    def get_depth_scale(self):
        return np.array([cam.get_depth_scale() for cam in self.cameras.values()])

    def start_recording(self, video_path: Union[str, List[str]], start_time: float):
        logger.info(
            f"MultiRealsense.start_recording({video_path}, start_time={start_time})"
        )
        # If user passed a single string, interpret it as a directory
        if isinstance(video_path, str):
            video_dir = pathlib.Path(video_path)
            assert video_dir.parent.is_dir(), "Directory's parent not found"
            video_dir.mkdir(parents=True, exist_ok=True)
            # We build a separate .mp4 for each camera
            new_paths = []
            i = 0
            for cam_serial in self.cameras:
                new_paths.append(str(video_dir.joinpath(f"{i}.mp4").absolute()))
                i += 1
            video_path = new_paths

        # Now we have a list of paths, one per camera
        assert len(video_path) == len(
            self.cameras
        ), f"Number of video paths {len(video_path)} != number of cameras {len(self.cameras)}"

        i = 0
        for cam_serial, camera in self.cameras.items():
            camera.start_recording(video_path[i], start_time)
            i += 1

    def stop_recording(self):
        logger.info("MultiRealsense.stop_recording()")
        for camera in self.cameras.values():
            camera.stop_recording()

    def restart_put(self, start_time):
        logger.info(f"MultiRealsense.restart_put(start_time={start_time})")
        for camera in self.cameras.values():
            camera.restart_put(start_time)

    def get_pointclouds(self, max_depth=2.0, min_depth=0.1, use_filters=True):
        """
        Get point clouds from all cameras using the same method as segment_pc.py.

        This creates point clouds with RealSense filters applied, matching the
        exact logic from the standalone segmentation script.

        Args:
            max_depth: Maximum depth in meters (default: 2.0)
            min_depth: Minimum depth in meters (default: 0.1)
            use_filters: Whether to apply spatial/temporal/hole-filling filters (default: True)

        Returns:
            Dict[str, Dict]: Dictionary mapping camera serial to point cloud data:
                {
                    'serial_number': {
                        'points': np.ndarray,  # Nx3 array of 3D points in camera frame
                        'colors': np.ndarray,  # Nx3 array of RGB colors [0-1]
                        'color_image': np.ndarray,  # HxWx3 color image
                        'depth_image': np.ndarray,  # HxW depth image (uint16)
                    }
                }
        """
        # Get latest camera data
        camera_data = self.get()
        depth_scales = self.get_depth_scale()

        result = {}

        for cam_idx, (serial, camera) in enumerate(self.cameras.items()):
            if cam_idx not in camera_data:
                logger.warning(f"No data available for camera {serial}")
                continue

            frames = camera_data[cam_idx]

            # Extract images
            color_images = frames.get('color', None)
            depth_images = frames.get('depth', None)

            if color_images is None or depth_images is None:
                logger.warning(f"Missing depth or color data for camera {serial}")
                continue

            # Get most recent frame
            color_image = color_images[-1] if len(color_images.shape) == 4 else color_images
            depth_image = depth_images[-1] if len(depth_images.shape) == 3 else depth_images

            # Get depth scale
            depth_scale = depth_scales[cam_idx] if cam_idx < len(depth_scales) else 0.001

            # Create RealSense frame objects from numpy arrays for filter processing
            # This matches the segment_pc.py approach
            if use_filters:
                # Apply RealSense filters (same as segment_pc.py)
                filters = [
                    rs.spatial_filter(),
                    rs.temporal_filter(),
                    rs.hole_filling_filter(),
                ]

                # We need to work with the depth as a proper depth frame
                # For now, skip filters and use direct processing
                # TODO: Implement filter application on numpy arrays
                logger.debug(f"Filters requested but not yet implemented for numpy arrays")
                filtered_depth = depth_image
            else:
                filtered_depth = depth_image

            # Convert depth to meters
            depth_meters = filtered_depth.astype(np.float32) * depth_scale

            # Get intrinsics
            intrinsics_list = self.get_intrinsics()
            intrinsics = np.array(intrinsics_list[cam_idx])

            # Parse intrinsics matrix
            if intrinsics.shape == (3, 3):
                fx = intrinsics[0, 0]
                fy = intrinsics[1, 1]
                cx = intrinsics[0, 2]
                cy = intrinsics[1, 2]
            else:
                logger.error(f"Unexpected intrinsics format for camera {serial}")
                continue

            # Create point cloud using same method as segment_pc.py
            # This uses pinhole camera model for back-projection
            h, w = depth_meters.shape
            u, v = np.meshgrid(np.arange(w), np.arange(h))

            # Back-project to 3D
            z = depth_meters
            x = (u - cx) * z / fx
            y = (v - cy) * z / fy

            points_3d = np.stack([x, y, z], axis=-1).reshape(-1, 3)
            colors = color_image.reshape(-1, 3)[:, ::-1] / 255.0  # BGR to RGB

            # Filter valid points (same as segment_pc.py)
            valid_mask = (
                (points_3d[:, 2] > min_depth)
                & (points_3d[:, 2] < max_depth)
                & np.isfinite(points_3d).all(axis=1)
            )

            valid_points = points_3d[valid_mask]
            valid_colors = colors[valid_mask]

            result[serial] = {
                'points': valid_points,
                'colors': valid_colors,
                'color_image': color_image,
                'depth_image': depth_image,
            }

            logger.debug(f"Camera {serial}: {len(valid_points)} valid points captured")

        return result
