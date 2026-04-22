"""
Minimal end-to-end test for the hardcoded `open` trajectory.

Bootstraps the XArm + a thin motion shim, loads configs/hardcoded_traj/open.txt
via HardcodedActionAdapter, then dispatches a pre-written JSON plan containing
just {"OPEN": [...]} through the VH json dispatcher.

No perception, no grasp model, no LLM — just motion + gripper.
"""

import logging
import pathlib
import time
from multiprocessing.managers import SharedMemoryManager

import numpy as np

from cap.lmp.json_dispatcher import run_json_plan
from cap.lmp.utils import load_config
from cap.lmp.waypoint_action_adapter import HardcodedActionAdapter
from cap.motion.xarm_motion import XArmMotionController
from ril_env.real_env import RealEnv
from ril_env.xarm_controller import XArmConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MotionEnv:
    """Tiny LMPWrapper stand-in — only the methods exec_hardcoded_traj uses."""

    def __init__(self, motion: XArmMotionController):
        self.motion = motion

    def goto_pos(self, pose, duration: float = 3.0):
        pose = list(np.array(pose, dtype=np.float32))
        return self.motion.move_to_pose(
            target_position=pose[:3],
            target_orientation=pose[3:],
            duration=duration,
        )

    def open_gripper(self):
        return self.motion.open_gripper()

    def close_gripper(self):
        return self.motion.close_gripper()


def main(plan_path: str = "test_plans/open_dishwasher.json",
         traj_dir: str = "configs/hardcoded_traj",
         action_map_path: str = "configs/action_map.yaml"):
    xarm_config = XArmConfig()

    with SharedMemoryManager() as shm:
        with RealEnv(
            output_dir=pathlib.Path("./recordings"),
            xarm_config=xarm_config,
            frequency=30,
            num_obs_steps=2,
            obs_image_resolution=(1280, 720),
            max_obs_buffer_size=30,
            obs_float32=True,
            init_joints=True,
            video_capture_fps=30,
            video_capture_resolution=(1280, 720),
            record_raw_video=False,
            thread_per_video=3,
            video_crf=21,
            enable_multi_cam_vis=False,
            multi_cam_vis_resolution=(1280, 720),
            enable_cameras=False,   # pure motion test
            shm_manager=shm,
        ) as env:
            time.sleep(1)

            motion = XArmMotionController(env, xarm_config)
            shim = MotionEnv(motion)
            adapter = HardcodedActionAdapter(shim, traj_dir)
            action_map = load_config(action_map_path)

            plan_str = pathlib.Path(plan_path).read_text()
            logger.info("=" * 60)
            logger.info(f"Plan: {plan_str.strip()}")
            logger.info("=" * 60)

            results = run_json_plan(plan_str, adapter, action_map)

            logger.info("=" * 60)
            logger.info("Results:")
            for r in results:
                logger.info(f"  {r}")
            logger.info("=" * 60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", default="test_plans/open_dishwasher.json")
    parser.add_argument("--traj-dir", default="configs/hardcoded_traj")
    parser.add_argument("--action-map", default="configs/action_map.yaml")
    args = parser.parse_args()
    main(plan_path=args.plan, traj_dir=args.traj_dir, action_map_path=args.action_map)
