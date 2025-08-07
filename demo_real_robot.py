import time
import traceback
import cv2
import numpy as np
import scipy.spatial.transform as st
import logging
import pathlib

from multiprocessing.managers import SharedMemoryManager

from ril_env.precise_sleep import precise_wait
from ril_env.xarm_controller import XArmConfig, XArm
from ril_env.real_env import RealEnv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

"""
TODO:
- Actually check that we are recording things
- Fix the camera visualization
- Fix the spacemouse-robot weird movement problem
- Clean up the code, add some documentation
- Demonstrations!
"""


def main(
    output="./recordings/",
    vis_camera_idx=0,
    init_joints=True,  # Not used ATM
    frequency=30,  # Cannot increase frequency
    command_latency=0.01,
    record_res=(1280, 720),
):
    dt = 1.0 / frequency
    output_dir = pathlib.Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    xarm_config = XArmConfig()

    with SharedMemoryManager() as shm_manager:
        with RealEnv(
                output_dir=output_dir,
                xarm_config=xarm_config,
                frequency=frequency,
                num_obs_steps=2,
                obs_image_resolution=record_res,
                max_obs_buffer_size=30,
                obs_float32=True,
                init_joints=init_joints,
                video_capture_fps=30,
                video_capture_resolution=record_res,
                record_raw_video=True,
                thread_per_video=3,
                video_crf=21,
                enable_multi_cam_vis=False,  # Totally broken RN
                multi_cam_vis_resolution=(1280, 720),
                shm_manager=shm_manager,
            ) as env:
                logger.info("Configuring camera settings...")
                env.realsense.set_exposure(exposure=120, gain=0)
                env.realsense.set_white_balance(white_balance=5900)

                time.sleep(1)
                logger.info("System initialized")

                state = env.get_robot_state()

                print(state)
                target_pose = np.array(state["TCPPose"], dtype=np.float32)
                logger.info(f"Initial pose: {target_pose}")

                exit()

if __name__ == "__main__":
    main()
