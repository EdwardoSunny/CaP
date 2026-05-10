"""
test_full_plan.py — Dispatch a pre-written JSON action plan through the full
CaP stack.

Same bootstrapping as main.py (perception + grasp + motion + LMPWrapper +
HardcodedActionAdapter), but skips the LLM: reads a plan straight from a
file and hands it to run_json_plan. Use this to exercise real `grab`
(perception-driven) plus scripted `open`/`put_in` without involving any
language model.
"""

import argparse
import logging
import os
import pathlib
import time
from multiprocessing.managers import SharedMemoryManager

# Visualizations disabled. To re-enable: uncomment below or run with
# `ENABLE_GRASP_VIZ=1 uv run python test_full_plan.py ...`.
# os.environ["ENABLE_GRASP_VIZ"] = "1"

from cap.lmp.json_dispatcher import run_json_plan
from cap.lmp.lmp_wrapper import setup_LMP
from cap.lmp.utils import load_config
from ril_env.real_env import RealEnv
from ril_env.xarm_controller import XArmConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _infer_traj_dir(plan_path: str) -> str:
    """Auto-pick a per-task hardcoded-traj directory based on the plan file's
    parent folder name. e.g. test_plans/dishwasher/foo.json →
    configs/hardcoded_traj/dishwasher (if it exists).
    """
    task_name = pathlib.Path(plan_path).parent.name
    candidate = pathlib.Path("configs/hardcoded_traj") / task_name
    if candidate.is_dir():
        return str(candidate)
    return "configs/hardcoded_traj"


def _read_camera_serials(traj_dir: str):
    """Optional override of perception camera_serials, read from the task's
    overrides.yaml. Returns a list[str] or None if not set."""
    import yaml
    path = pathlib.Path(traj_dir) / "overrides.yaml"
    if not path.is_file():
        return None
    with path.open() as f:
        data = yaml.safe_load(f) or {}
    serials = data.get("camera_serials")
    if serials is None:
        return None
    return [str(s) for s in serials]


def main(
    plan_path: str,
    grasp_strategy: str = "hardcode",
    hardcoded_traj_dir: str = None,
    record_res: tuple = (1280, 720),
    frequency: int = 30,
):
    if hardcoded_traj_dir is None:
        hardcoded_traj_dir = _infer_traj_dir(plan_path)
        logger.info(f"Auto-selected traj dir: {hardcoded_traj_dir}")
    camera_serials = _read_camera_serials(hardcoded_traj_dir)
    if camera_serials is not None:
        logger.info(f"Per-task camera_serials override: {camera_serials}")
    xarm_config = XArmConfig()
    output_dir = pathlib.Path("./recordings")
    output_dir.mkdir(parents=True, exist_ok=True)

    with SharedMemoryManager() as shm_manager:
        with RealEnv(
            output_dir=output_dir,
            xarm_config=xarm_config,
            frequency=frequency,
            num_obs_steps=2,
            obs_image_resolution=record_res,
            max_obs_buffer_size=30,
            obs_float32=True,
            init_joints=True,
            video_capture_fps=30,
            video_capture_resolution=record_res,
            record_raw_video=True,
            thread_per_video=3,
            video_crf=21,
            enable_multi_cam_vis=False,
            multi_cam_vis_resolution=(1280, 720),
            enable_cameras=False,   # perception module owns its own cameras
            shm_manager=shm_manager,
        ) as env:
            logger.info("System initialized")
            time.sleep(1)

            config = load_config("configs/real_config.yaml")
            lmp_tabletop_ui, _ = setup_LMP(
                config, env, xarm_config,
                grasp_strategy=grasp_strategy,
                plan_mode="json",
                hardcoded_traj_dir=hardcoded_traj_dir,
                camera_serials=camera_serials,
            )

            # setup_LMP wired the adapter + action_map onto the LMP's cfg.
            adapter = lmp_tabletop_ui._cfg["_json_adapter"]
            action_map = lmp_tabletop_ui._cfg["_action_map"]

            plan_str = pathlib.Path(plan_path).read_text()

            logger.info("=" * 60)
            logger.info(f"Dispatching plan: {plan_path}")
            logger.info(f"  {plan_str.strip()}")
            logger.info("=" * 60)

            results = run_json_plan(plan_str, adapter, action_map)

            logger.info("=" * 60)
            logger.info("Results:")
            for r in results:
                logger.info(f"  {r}")
            logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True,
                        help="Path to JSON plan file (repo VH format).")
    parser.add_argument("--grasp", default="hardcode",
                        choices=["hardcode", "graspgen"],
                        help="Grasp strategy. 'hardcode' avoids GraspGen checkpoint load.")
    parser.add_argument("--traj-dir", default=None,
                        help="Directory with hardcoded trajectory .txt files. "
                             "Default: auto-pick configs/hardcoded_traj/<task>/ from plan path.")
    args = parser.parse_args()
    main(
        plan_path=args.plan,
        grasp_strategy=args.grasp,
        hardcoded_traj_dir=args.traj_dir,
    )
