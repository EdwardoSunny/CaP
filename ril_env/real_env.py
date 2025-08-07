import logging
import time
import math
import numpy as np
import shutil
import pathlib

from multiprocessing.managers import SharedMemoryManager
from ril_env.xarm_controller import XArmConfig, XArmController
from ril_env.replay_buffer import ReplayBuffer
from ril_env.realsense import SingleRealsense
from ril_env.cv2_util import get_image_transform, optimal_row_cols
from ril_env.video_recorder import VideoRecorder
from ril_env.multi_realsense import MultiRealsense
from ril_env.multi_camera_visualizer import MultiCameraVisualizer
from ril_env.timestamp_accumulator import (
    TimestampActionAccumulator,
    TimestampObsAccumulator,
)
from typing import Tuple, List, Optional, Dict, Union

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_OBS_KEY_MAP = {
    # Robot - using the correct key names that match XArmController.get_all_state()
    "TCPPose": "robot_eef_pose",
    "TCPSpeed": "robot_eef_pose_vel",
    "JointAngles": "robot_joint",
    "JointSpeeds": "robot_joint_vel",
    # Additional keys if they exist
    "Grasp": "robot_gripper",
    "robot_receive_timestamp": "robot_timestamp",
    # Camera stuff,
    "Camera_0": "camera_0",
    "Camera_1": "camera_1",
    "Camera_2": "camera_2",
    # Timestamps
    "step_idx": "step_idx",
    "timestamp": "timestamp",
}


class RealEnv:
    def __init__(
        self,
        output_dir: Union[pathlib.Path, str] = "./recordings/",
        xarm_config: Optional[XArmConfig] = None,
        frequency: int = 30,
        num_obs_steps: int = 2,
        obs_image_resolution: Tuple[int, int] = (640, 480),
        max_obs_buffer_size: int = 30,
        camera_serial_numbers: Optional[List[int]] = None,
        obs_key_map: Dict = DEFAULT_OBS_KEY_MAP,
        obs_float32: bool = False,
        init_joints: bool = False,
        video_capture_fps: int = 30,
        video_capture_resolution: Tuple[int, int] = (1280, 720),
        record_raw_video: bool = True,
        thread_per_video: int = 3,
        video_crf: int = 3,
        enable_multi_cam_vis: bool = False,
        multi_cam_vis_resolution: Tuple[int, int] = (1280, 720),
        shm_manager: Optional[SharedMemoryManager] = None,
    ):
        logger.info("[RealEnv] Initializing environment.")

        output_dir = pathlib.Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        video_dir = output_dir.joinpath("videos")
        video_dir.mkdir(parents=True, exist_ok=True)

        zarr_path = str(output_dir.joinpath("replay_buffer.zarr").absolute())
        replay_buffer = ReplayBuffer.create_from_path(zarr_path=zarr_path, mode="a")

        logger.info(f"[RealEnv] Output directory: {output_dir}")
        logger.info(f"[RealEnv] Video directory: {video_dir}")
        logger.info(f"[RealEnv] Replay buffer path: {zarr_path}")

        if xarm_config is None:
            xarm_config = XArmConfig()

        assert (
            frequency <= video_capture_fps
        ), "Cannot run frequency faster than video capture."

        if shm_manager is None:
            shm_manager = SharedMemoryManager()
            shm_manager.start()
            logger.info("[RealEnv] Started local SharedMemoryManager")
        self.shm_manager = shm_manager

        if camera_serial_numbers is None:
            camera_serial_numbers = SingleRealsense.get_connected_devices_serial()

        logger.info(f"[RealEnv] Camera serial numbers: {camera_serial_numbers}")

        color_tf = get_image_transform(
            input_res=video_capture_resolution,
            output_res=obs_image_resolution,
            bgr_to_rgb=True,
        )
        color_transform = color_tf

        if obs_float32:

            def float_transform(img):
                return color_tf(img).astype(np.float32) / 255.0

            color_transform = float_transform

        def transform(data):
            if "color" in data:
                data["color"] = color_transform(data["color"])
            return data

        # Multi-cam visual transformation
        rw, rh, col, row = optimal_row_cols(
            n_cameras=len(camera_serial_numbers),
            in_wh_ratio=obs_image_resolution[0] / obs_image_resolution[1],
            max_resolution=multi_cam_vis_resolution,
        )
        vis_color_transform = get_image_transform(
            input_res=video_capture_resolution,
            output_res=(rw, rh),
            bgr_to_rgb=False,
        )

        def vis_transform(data):
            if "color" in data:
                data["color"] = vis_color_transform(data["color"])
            return data

        recording_transform = None
        recording_fps = video_capture_fps
        recording_pix_fmt = "bgr24"
        if not record_raw_video:
            recording_transform = transform
            recording_fps = frequency
            recording_pix_fmt = "rgb24"

        video_recorder = VideoRecorder.create_h264(
            fps=recording_fps,
            codec="h264",
            input_pix_fmt=recording_pix_fmt,
            crf=video_crf,
            thread_type="FRAME",
            thread_count=thread_per_video,
        )
        logger.info(f"[RealEnv] Recording FPS: {recording_fps}")

        realsense = MultiRealsense(
            serial_numbers=camera_serial_numbers,
            shm_manager=shm_manager,
            resolution=video_capture_resolution,
            record_fps=recording_fps,
            capture_fps=video_capture_fps,
            put_fps=video_capture_fps,
            put_downsample=True,
            enable_color=True,
            enable_depth=False,
            enable_infrared=False,
            get_max_k=max_obs_buffer_size,
            transform=transform,
            vis_transform=vis_transform,
            recording_transform=recording_transform,
            video_recorder=video_recorder,
            verbose=False,
        )

        multi_cam_vis = None
        if enable_multi_cam_vis:
            multi_cam_vis = MultiCameraVisualizer(
                realsense=realsense,
                row=row,
                col=col,
                window_name="Multi Cam Vis",
                rgb_to_bgr=False,
            )

        robot = XArmController(
            shm_manager=shm_manager,
            xarm_config=xarm_config,
        )

        self.realsense = realsense
        self.robot = robot
        self.xarm_config = xarm_config
        self.multi_cam_vis = multi_cam_vis
        self.video_capture_fps = video_capture_fps
        self.frequency = frequency
        self.num_obs_steps = num_obs_steps
        self.max_obs_buffer_size = max_obs_buffer_size
        self.obs_key_map = obs_key_map

        self.output_dir = output_dir
        self.video_dir = video_dir
        self.replay_buffer = replay_buffer

        self.last_realsense_data = None

        self.obs_accumulator = None
        self.action_accumulator = None
        self.stage_accumulator = None

        self.start_time = None

    # Start-stop API
    @property
    def is_ready(self):
        return self.realsense.is_ready and self.robot.is_ready

    def start(self, wait=True):
        if wait:
            self.realsense.start(wait=True)
            self.robot.start(wait=True)
        else:
            self.realsense.start(wait=False)
            self.robot.start(wait=False)
        if self.multi_cam_vis is not None:
            self.multi_cam_vis.start(wait=False)

    def stop(self, wait=True):
        self.end_episode()
        if wait:
            self.realsense.stop(wait=True)
            self.robot.stop(wait=True)
        else:
            self.realsense.stop(wait=False)
            self.robot.stop(wait=False)
        if self.multi_cam_vis is not None:
            self.multi_cam_vis.stop(wait=False)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # Async env API
    def get_obs(self) -> Dict:
        "observation dict"
        assert self.is_ready

        k = math.ceil(self.num_obs_steps * (self.video_capture_fps / self.frequency))
        self.last_realsense_data = self.realsense.get(k=k, out=self.last_realsense_data)

        # Running at 50 hz
        last_robot_data = self.robot.get_all_state()

        dt = 1 / self.frequency
        last_timestamp = np.max(
            [x["timestamp"][-1] for x in self.last_realsense_data.values()]
        )
        obs_align_timestamps = last_timestamp - (
            np.arange(self.num_obs_steps)[::-1] * dt
        )

        camera_obs = dict()
        for camera_idx, value in self.last_realsense_data.items():
            this_timestamps = value["timestamp"]
            this_idxs = list()
            for t in obs_align_timestamps:
                is_before_idxs = np.nonzero(this_timestamps < t)[0]
                this_idx = 0
                if len(is_before_idxs) > 0:
                    this_idx = is_before_idxs[-1]
                this_idxs.append(this_idx)

            camera_obs[f"camera_{camera_idx}"] = value["color"][this_idxs]

        robot_timestamps = last_robot_data["robot_receive_timestamp"]
        this_timestamps = robot_timestamps
        this_idxs = list()
        for t in obs_align_timestamps:
            is_before_idxs = np.nonzero(this_timestamps < t)[0]
            this_idx = 0
            if len(is_before_idxs) > 0:
                this_idx = is_before_idxs[-1]
            this_idxs.append(this_idx)

        robot_obs_raw = dict()
        for k, v in last_robot_data.items():
            if k in self.obs_key_map:
                robot_obs_raw[self.obs_key_map[k]] = v

        robot_obs = dict()
        for k, v in robot_obs_raw.items():
            robot_obs[k] = v[this_idxs]

        # Accumulate observations
        if self.obs_accumulator is not None:
            # Create a complete observation that includes both robot and camera
            complete_obs = dict()
            
            # Add robot observations with original keys
            for k, v in last_robot_data.items():
                if k in self.obs_key_map:
                    complete_obs[k] = v
            
            # Add camera observations
            """
            for camera_idx, data in self.last_realsense_data.items():
                camera_key = f"Camera_{camera_idx}"
                complete_obs[camera_key] = data["color"]
            """
            # Create absolute timestamps for accumulation
            now = time.time()
            relative_timestamps = robot_timestamps
            absolute_timestamps = np.array([now + (t - relative_timestamps[0]) for t in relative_timestamps])
            
            # Put data in the accumulator
            self.obs_accumulator.put(complete_obs, absolute_timestamps)

        obs_data = dict(camera_obs)
        obs_data.update(robot_obs)
        obs_data["timestamp"] = obs_align_timestamps
        return obs_data

    def exec_actions(
        self,
        actions: np.ndarray,
        timestamps: np.ndarray,
        stages: Optional[np.ndarray] = None,
    ):
        assert self.is_ready
        if not isinstance(actions, np.ndarray):
            actions = np.array(actions)
        if not isinstance(timestamps, np.ndarray):
            timestamps = np.array(timestamps)
        if stages is None:
            stages = np.zeros_like(timestamps, dtype=np.int64)
        elif not isinstance(stages, np.ndarray):
            stages = np.array(stages, dtype=np.int64)

        receive_time = time.time()
        is_new = timestamps > receive_time
        new_actions = actions[is_new]
        new_timestamps = timestamps[is_new]
        new_stages = stages[is_new]
        
        # Execute robot actions
        for i in range(len(new_actions)):
            new_action = new_actions[i]
            pose = new_action[:6]
            grasp = new_action[-1]
            self.robot.step(pose, grasp)

        # Record actions if recording
        if self.action_accumulator is not None:
            # Use the same timestamps for both robot and recordings
            self.action_accumulator.put(
                new_actions,
                new_timestamps,
            )

        if self.stage_accumulator is not None:
            self.stage_accumulator.put(
                new_stages,
                new_timestamps,
            )

    def get_robot_state(self):
        return self.robot.get_state()

    def start_episode(self, start_time=None):
        if start_time is None:
            start_time = time.time()
        
        # Use absolute timestamps for everything
        self.start_time = start_time

        assert self.is_ready

        episode_id = self.replay_buffer.n_episodes
        this_video_dir = self.video_dir.joinpath(str(episode_id))
        this_video_dir.mkdir(parents=True, exist_ok=True)
        n_cameras = self.realsense.n_cameras
        video_paths = list()
        for i in range(n_cameras):
            video_paths.append(str(this_video_dir.joinpath(f"{i}.mp4").absolute()))

        # Use absolute timestamps for recordings
        self.realsense.restart_put(start_time=start_time)
        self.realsense.start_recording(video_path=video_paths, start_time=start_time)

        # Initialize accumulators with absolute timestamps
        self.obs_accumulator = TimestampObsAccumulator(
            start_time=start_time,
            dt=1 / self.frequency,
        )
        self.action_accumulator = TimestampActionAccumulator(
            start_time=start_time,
            dt=1 / self.frequency,
        )
        self.stage_accumulator = TimestampActionAccumulator(
            start_time=start_time,
            dt=1 / self.frequency,
        )
        print(f"Episode {episode_id} started!")

    def end_episode(self):
        assert self.is_ready

        self.realsense.stop_recording()

        if self.obs_accumulator is not None:
            assert self.action_accumulator is not None
            assert self.stage_accumulator is not None

            obs_data = self.obs_accumulator.data
            obs_timestamps = self.obs_accumulator.timestamps

            actions = self.action_accumulator.actions
            action_timestamps = self.action_accumulator.timestamps
            stages = self.stage_accumulator.actions
            n_steps = min(len(obs_timestamps), len(action_timestamps))
            if n_steps > 0:
                episode = dict()
                episode["timestamp"] = obs_timestamps[:n_steps]
                episode["action"] = actions[:n_steps]
                episode["stage"] = stages[:n_steps]
                for key, value in obs_data.items():
                    episode[key] = value[:n_steps]
                self.replay_buffer.add_episode(episode, compressors="disk")
                episode_id = self.replay_buffer.n_episodes - 1
                print(f"Episode {episode_id} saved!")

            self.obs_accumulator = None
            self.action_accumulator = None
            self.stage_accumulator = None

    def drop_episode(self):
        self.end_episode()
        self.replay_buffer.drop_episode()
        episode_id = self.replay_buffer.n_episodes
        this_video_dir = self.video_dir.joinpath(str(episode_id))
        if this_video_dir.exists():
            shutil.rmtree(str(this_video_dir))
        print(f"Episode {episode_id} dropped!")
