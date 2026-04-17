"""
LMP Wrapper — thin API surface for LLM-generated robot commands.

Delegates to:
- cap.motion.XArmMotionController   — robot motion
- cap.grasp.GraspGenStrategy         — grasp generation (swappable)
- cap.perception.SAMMolmoPerception  — object segmentation (Molmo + SAM2)
- cap.pipeline.GraspPipeline         — orchestration
- cap.viz.GraspVisualizer            — visualization
"""

import os
import time
import numpy as np
import scipy.spatial.transform as st
import logging
from typing import Optional
from pathlib import Path

import shapely

from cap.lmp.lmp import LMP, LMPFGen
from cap.grasp.alignment import compute_object_yaw

# Modular components
from cap.motion.base import MotionController
from cap.motion.xarm_motion import XArmMotionController
from cap.grasp.base import GraspStrategy, GraspResult
from cap.grasp.graspgen_strategy import GraspGenStrategy
from cap.grasp.hardcode_strategy import HardcodeStrategy
from cap.perception.base import PerceptionModule
from cap.perception.sam_clip_perception import SAMMolmoPerception
from cap.pipeline import GraspPipeline
from cap.viz.visualization import GraspVisualizer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LMPWrapper — the only class the LMP layer sees
# ---------------------------------------------------------------------------

class LMPWrapper:
    """
    Provides the callable API surface for LLM-generated Python code.

    All heavy lifting is delegated to injected strategy objects.
    The wrapper handles:
    - LMP environment state (known objects, workspace, etc.)
    - Detection caching
    - Unit conversions (GraspResult is in meters; robot commands are in mm/deg)
    - Glue between pipeline results and motion commands
    """

    def __init__(
        self,
        motion: MotionController,
        pipeline: GraspPipeline,
        visualizer: Optional[GraspVisualizer] = None,
        known_objects: Optional[list] = None,
    ):
        """
        Args:
            motion: Robot motion controller (e.g. XArmMotionController)
            pipeline: Grasp pipeline (perception + grasp strategy + viz)
            visualizer: Optional visualizer for after-execution comparison
            known_objects: Initial list of known object names
        """
        self.motion = motion
        self.pipeline = pipeline
        self.visualizer = visualizer

        # Detection cache
        self._detection_cache = {}

        # After-visualization data
        self._last_viz_data = None

        # LMP environment state
        self._setup_lmp_environment(known_objects)

        logger.info(f"Robot Primitives initialized. Current pose: {self.motion.get_robot_pose()}")

    # ------------------------------------------------------------------
    # LMP environment setup
    # ------------------------------------------------------------------

    def _setup_lmp_environment(self, known_objects=None):
        """Setup LMP-specific environment variables and object tracking."""
        self.known_objects = known_objects or [
            "bread", "red ball", "white bottle", "red cup", "tissue box", "screwdriver", "shelf", "rubix cube",
        ]
        self._last_viz_data = None

        # Global position offset applied to every detected object position.
        # Units: millimeters. [dx, dy, dz] in robot base frame.
        # Adjust these to compensate for systematic perception errors.
        #   +X = further from robot, -X = closer to robot
        #   +Y = robot's left,      -Y = robot's right
        #   +Z = up,                 -Z = down (deeper grasp)
        self.detection_offset_mm = np.array([0.0, 0.0, 30.0])

        self.workspace_bounds = {
            "x_min": -0.3, "x_max": 0.3,
            "y_min": -0.8, "y_max": -0.2,
            "z_table": 0.0,
        }

        self.corner_positions = {
            "top left corner": (-0.25, -0.25, 0),
            "top right corner": (0.25, -0.25, 0),
            "bottom left corner": (-0.25, -0.75, 0),
            "bottom right corner": (0.25, -0.75, 0),
        }

        self.side_positions = {
            "top side": (0, -0.25, 0),
            "bottom side": (0, -0.75, 0),
            "left side": (-0.25, -0.5, 0),
            "right side": (0.25, -0.5, 0),
        }

        self.colors = {
            "red": (1.0, 0.0, 0.0, 1.0),
            "green": (0.0, 1.0, 0.0, 1.0),
            "blue": (0.0, 0.0, 1.0, 1.0),
            "yellow": (1.0, 1.0, 0.0, 1.0),
            "orange": (1.0, 0.5, 0.0, 1.0),
            "purple": (0.5, 0.0, 0.5, 1.0),
            "pink": (1.0, 0.75, 0.8, 1.0),
            "cyan": (0.0, 1.0, 1.0, 1.0),
            "brown": (0.6, 0.3, 0.1, 1.0),
            "gray": (0.5, 0.5, 0.5, 1.0),
        }

    # ------------------------------------------------------------------
    # Core detection (delegates to pipeline)
    # ------------------------------------------------------------------

    def _get_eef_transform(self) -> Optional[np.ndarray]:
        """Get current EEF as 4x4 transform in meters."""
        try:
            pose = self.motion.get_robot_pose()
            pos_m = pose[:3] / 1000.0
            rot = st.Rotation.from_euler("xyz", pose[3:], degrees=True)
            T = np.eye(4)
            T[:3, :3] = rot.as_matrix()
            T[:3, 3] = pos_m
            return T
        except Exception as e:
            logger.warning(f"Could not get EEF transform: {e}")
            return None

    def detect_object_location(self, prompt):
        """
        Run the full perception + grasp pipeline for an object.

        Returns:
            np.array [x, y, z, roll, pitch, yaw] in mm/degrees, or None.
        """
        eef_transform = self._get_eef_transform()

        grasp_result = self.pipeline.detect_and_grasp(
            text_prompt=prompt,
            current_eef_pose=eef_transform,
        )

        if not grasp_result.success:
            logger.warning(f"No grasp found for '{prompt}'")
            return None

        # Convert to robot command format
        grasp_pose = GraspPipeline.grasp_to_pose_6dof(grasp_result)

        if grasp_pose is None:
            return None

        # Apply global detection offset
        grasp_pose[:3] += self.detection_offset_mm

        # Override yaw with object-aligned yaw so the gripper fingers
        # close across the object's narrow side.
        if grasp_result.object_points is not None and len(grasp_result.object_points) >= 3:
            yaw_deg = compute_object_yaw(grasp_result.object_points)
            grasp_pose[5] = yaw_deg
            logger.info(f"Object alignment: yaw set to {yaw_deg:.1f} deg")

        position_mm = grasp_pose[:3]
        orientation_deg = grasp_pose[3:]

        logger.info(f"Best grasp pose for robot (after offset {self.detection_offset_mm.tolist()} mm):")
        logger.info(f"  Position (mm): [{position_mm[0]:.1f}, {position_mm[1]:.1f}, {position_mm[2]:.1f}]")
        logger.info(f"  Orientation (deg): [{orientation_deg[0]:.1f}, {orientation_deg[1]:.1f}, {orientation_deg[2]:.1f}]")
        logger.info(f"  Score: {grasp_result.best_score:.3f}")

        # Store data for after-visualization
        self._last_viz_data = {
            "full_points": grasp_result.scene_points,
            "full_colors": grasp_result.scene_colors,
            "seg_points": grasp_result.object_points,
            "seg_colors": grasp_result.object_colors,
            "best_grasp": grasp_result.best_grasp,
            "prompt": prompt,
        }

        return grasp_pose

    def get_object_center(self, prompt, top_percentile=10, use_cache=True):
        """
        Get grasp pose for an object (with caching).

        Returns:
            np.array [x, y, z, roll, pitch, yaw] in mm/degrees, or None.
        """
        if use_cache and prompt in self._detection_cache:
            logger.info(f"Using cached grasp pose for '{prompt}'")
            return self._detection_cache[prompt]["full_pose"]

        grasp_pose = self.detect_object_location(prompt)
        if grasp_pose is None:
            logger.warning(f"No grasp found for object: {prompt}")
            return None

        self._detection_cache[prompt] = {"full_pose": grasp_pose}
        return grasp_pose

    def clear_detection_cache(self):
        """Clear the detection cache."""
        self._detection_cache.clear()
        logger.info("Detection cache cleared")

    # ------------------------------------------------------------------
    # Robot state queries (delegate to motion controller)
    # ------------------------------------------------------------------

    def get_robot_pos(self):
        """Return robot EEF [x, y, z] in mm."""
        return self.motion.get_robot_pos()

    def get_robot_pose(self):
        """Return [x, y, z, roll, pitch, yaw] in mm/degrees."""
        return self.motion.get_robot_pose()

    def get_robot_xy(self):
        """Return [x, y] in mm."""
        return self.motion.get_robot_xy()

    # ------------------------------------------------------------------
    # Motion commands (delegate to motion controller)
    # ------------------------------------------------------------------

    def goto_pos(self, position_xyz_or_pose, duration=3.0, stage_val=0):
        """
        Move to position [x,y,z] or full pose [x,y,z,r,p,y].
        Triggers after-visualization when reaching a grasp target.
        """
        current_pose = self.get_robot_pose()

        if len(position_xyz_or_pose) == 6:
            target_position = list(np.array(position_xyz_or_pose[:3], dtype=np.float32))
            target_orientation = list(np.array(position_xyz_or_pose[3:], dtype=np.float32))
            logger.info(f"goto_pos() full 6DOF: pos={target_position}, ori={target_orientation}")
        else:
            target_position = list(np.array(position_xyz_or_pose, dtype=np.float32))
            target_orientation = [180.0, 0.0, 0.0]
            logger.info(f"goto_pos() position only: {target_position}, default ori={target_orientation}")

        result = self.motion.move_to_pose(
            target_position=target_position,
            target_orientation=target_orientation,
            duration=duration,
            stage_val=stage_val,
        )

        # After-visualization: disabled — set ENABLE_GRASP_VIZ=1 to re-enable
        if self._last_viz_data is not None and self.visualizer is not None and os.environ.get("ENABLE_GRASP_VIZ"):
            d = self._last_viz_data
            if d["best_grasp"] is not None:
                grasp_pos_mm = d["best_grasp"][:3, 3] * 1000.0
                target_pos_mm = np.array(target_position[:3], dtype=np.float64)
                if np.linalg.norm(grasp_pos_mm - target_pos_mm) < 5.0:
                    eef_transform = self._get_eef_transform()
                    try:
                        self.visualizer.visualize_after_execution(
                            full_points=d["full_points"],
                            full_colors=d["full_colors"],
                            seg_points=d["seg_points"],
                            seg_colors=d["seg_colors"],
                            best_grasp=d["best_grasp"],
                            eef_transform=eef_transform,
                            prompt=d["prompt"],
                        )
                    except Exception as e:
                        logger.warning(f"After-visualization failed: {e}")
                    self._last_viz_data = None

        return result

    def goto_xy(self, position_xy, duration=2.0, stage_val=0):
        """Move to [x, y] keeping current z."""
        current_pos = self.get_robot_pos()
        target_xyz = np.concatenate([position_xy, [current_pos[2]]])
        return self.goto_pos(target_xyz, duration, stage_val)

    def goto_pose(self, position_xyz, orientation_rpy, duration=3.0, stage_val=0):
        """Move to position + orientation."""
        return self.motion.move_to_pose(
            target_position=list(position_xyz),
            target_orientation=list(orientation_rpy),
            duration=duration,
            stage_val=stage_val,
        )

    def move_relative(self, delta_xyz, delta_rpy=None, duration=2.0, stage_val=0):
        """Move relative to current pose."""
        return self.motion.move_relative(
            delta_position=delta_xyz,
            delta_orientation=delta_rpy,
            duration=duration,
            stage_val=stage_val,
        )

    # def move_up(self, distance=0.05, duration=1.0, stage_val=0):
    def move_up(self, distance=2.00, duration=1.0, stage_val=0):
        """Move robot up by specified distance."""
        return self.move_relative([0, 0, distance], duration=duration, stage_val=stage_val)

    def move_down(self, distance=2.00, duration=1.0, stage_val=0):
        """Move robot down by specified distance."""
        return self.move_relative([0, 0, -distance], duration=duration, stage_val=stage_val)

    # ------------------------------------------------------------------
    # Gripper commands
    # ------------------------------------------------------------------

    def open_gripper(self, stage_val=0):
        return self.motion.open_gripper(stage_val)

    def close_gripper(self, stage_val=0):
        return self.motion.close_gripper(stage_val)

    def set_gripper(self, grasp_value, stage_val=0):
        return self.motion.set_gripper(grasp_value, stage_val)

    def is_gripper_open(self):
        return self.motion._current_grasp < 0.5

    def is_gripper_closed(self):
        return self.motion._current_grasp >= 0.5

    def get_gripper_state(self):
        return self.motion._current_grasp

    # ------------------------------------------------------------------
    # Compound motions
    # ------------------------------------------------------------------

    def pick_place(
        self, pick_pos, place_pos,
        approach_offset=100.0, place_offset=50.0,
        stage_val=0,
    ):
        """Execute pick and place operation.

        Args:
            pick_pos: Pick target — [x,y,z], [x,y,z,r,p,y], or [x,y] (mm/deg).
            place_pos: Place target — same formats as pick_pos.
            approach_offset: How far above the target (mm) to approach from.
            place_offset: How far above the place target (mm) to release.
            stage_val: Stage value for action recording.
        """
        try:
            pick_pos_xyz = np.array(pick_pos[:3], dtype=np.float64)
            place_pos_xyz = np.array(place_pos[:3], dtype=np.float64)

            # Approach points: directly above pick/place targets
            approach_pick = np.array([pick_pos_xyz[0], pick_pos_xyz[1], pick_pos_xyz[2] + approach_offset])
            approach_place = np.array([place_pos_xyz[0], place_pos_xyz[1], place_pos_xyz[2] + approach_offset])

            logger.info(f"Pick and place: {pick_pos_xyz} -> {place_pos_xyz}")
            logger.info(f"  Approach offset: {approach_offset} mm, Place offset: {place_offset} mm")

            self.goto_pos(approach_pick, duration=3.0, stage_val=stage_val)
            self.open_gripper(stage_val)
            time.sleep(0.5)
            self.goto_pos(pick_pos, duration=2.0, stage_val=stage_val)
            self.close_gripper(stage_val)
            time.sleep(1.0)
            self.goto_pos(approach_pick, duration=2.0, stage_val=stage_val)
            self.goto_pos(approach_place, duration=3.0, stage_val=stage_val)
            # Place above the target to avoid pressing into the table
            place_above = np.array([place_pos_xyz[0], place_pos_xyz[1], place_pos_xyz[2] + place_offset])
            self.goto_pos(place_above, duration=2.0, stage_val=stage_val)
            self.open_gripper(stage_val)
            time.sleep(0.5)
            self.goto_pos(approach_place, duration=2.0, stage_val=stage_val)

            logger.info("Pick and place completed successfully")
            return True

        except Exception as e:
            logger.error(f"Pick and place failed: {e}")
            return False

    def follow_traj(self, trajectory, duration_per_point=1.0, stage_val=0):
        """Follow a trajectory of positions."""
        for i, pos in enumerate(trajectory):
            logger.info(f"Following trajectory point {i+1}/{len(trajectory)}: {pos}")
            if len(pos) == 2:
                self.goto_xy(pos, duration=duration_per_point, stage_val=stage_val)
            else:
                self.goto_pos(pos, duration=duration_per_point, stage_val=stage_val)

    def wait(self, duration):
        time.sleep(duration)

    # ------------------------------------------------------------------
    # LMP-required object functions
    # ------------------------------------------------------------------

    def get_obj_names(self):
        return self.known_objects.copy()

    def is_obj_visible(self, obj_name):
        return obj_name in self.known_objects

    def get_obj_pos(self, obj_name, use_segmentation=True):
        """
        Get object pose. Returns [x, y, z, roll, pitch, yaw] in mm/degrees.
        Checks predefined positions first, then uses segmentation.
        """
        obj_name_clean = obj_name.replace("the", "").replace("_", " ").strip()

        if obj_name_clean in self.corner_positions:
            pos = list(self.corner_positions[obj_name_clean])
            return pos + [179.0, 0.0, 0.0]
        elif obj_name_clean in self.side_positions:
            pos = list(self.side_positions[obj_name_clean])
            return pos + [179.0, 0.0, 0.0]

        if use_segmentation:
            logger.info(f"Using segmentation to locate: {obj_name_clean}")
            grasp_pose = self.get_object_center(obj_name_clean)
            if grasp_pose is not None:
                return grasp_pose.tolist()
            else:
                raise RuntimeError(
                    f"Object '{obj_name_clean}' could not be detected. "
                    "Segmentation returned no valid grasp pose."
                )

        raise RuntimeError(
            f"Object '{obj_name_clean}' not found. "
            "Not a predefined position and segmentation is disabled."
        )

    def get_bbox(self, obj_name):
        pos = self.get_obj_pos(obj_name)
        size = 0.02
        return (pos[0] - size, pos[1] - size, pos[0] + size, pos[1] + size)

    def get_color(self, obj_name):
        for color_name, rgb in self.colors.items():
            if color_name in obj_name.lower():
                return rgb
        return (0.5, 0.5, 0.5, 1.0)

    def denormalize_xy(self, pos_normalized):
        x_range = self.workspace_bounds["x_max"] - self.workspace_bounds["x_min"]
        y_range = self.workspace_bounds["y_max"] - self.workspace_bounds["y_min"]
        x = pos_normalized[0] * x_range + self.workspace_bounds["x_min"]
        y = pos_normalized[1] * y_range + self.workspace_bounds["y_min"]
        return np.array([x, y])

    def get_corner_name(self, pos):
        pos_2d = np.array(pos[:2])
        positions_2d = np.array([[p[0], p[1]] for p in self.corner_positions.values()])
        distances = np.linalg.norm(positions_2d - pos_2d, axis=1)
        return list(self.corner_positions.keys())[int(np.argmin(distances))]

    def get_side_name(self, pos):
        pos_2d = np.array(pos[:2])
        positions_2d = np.array([[p[0], p[1]] for p in self.side_positions.values()])
        distances = np.linalg.norm(positions_2d - pos_2d, axis=1)
        return list(self.side_positions.keys())[int(np.argmin(distances))]

    def put_first_on_second(self, obj1, obj2):
        try:
            pick_pos = self.get_obj_pos(obj1) if isinstance(obj1, str) else np.array(obj1)
            place_pos = self.get_obj_pos(obj2) if isinstance(obj2, str) else np.array(obj2)
            logger.info(f"put_first_on_second: {obj1} -> {obj2}")
            return self.pick_place(pick_pos, place_pos)
        except Exception as e:
            logger.error(f"Error in put_first_on_second: {e}")
            return False

    # ------------------------------------------------------------------
    # Object management
    # ------------------------------------------------------------------

    def add_object(self, obj_name, position=None):
        if obj_name not in self.known_objects:
            self.known_objects.append(obj_name)

    def remove_object(self, obj_name):
        if obj_name in self.known_objects:
            self.known_objects.remove(obj_name)

    def update_object_list(self, object_list):
        self.known_objects = object_list.copy()

    def clear_objects(self):
        self.known_objects.clear()

    # ------------------------------------------------------------------
    # Workspace utilities
    # ------------------------------------------------------------------

    def get_workspace_bounds(self):
        return self.workspace_bounds.copy()

    def set_workspace_bounds(self, bounds):
        self.workspace_bounds.update(bounds)

    def get_corner_positions(self):
        return self.corner_positions.copy()

    def get_side_positions(self):
        return self.side_positions.copy()


# ---------------------------------------------------------------------------
# Factory function — wires everything together
# ---------------------------------------------------------------------------

def setup_LMP(config, env, xarm_config, grasp_strategy="graspgen", model=None, vllm_host=None, max_tokens=None, context_window=None, plan_mode="python"):
    """
    Setup the full LMP system.

    Args:
        config: Configuration dict from real_config.yaml
        env: RealEnv instance
        xarm_config: XArmConfig instance
        grasp_strategy: Which grasp strategy to use.
                        "graspgen" — NVIDIA GraspGen (default)
                        Or pass a GraspStrategy instance directly.
        model: Override LLM model for all LMPs. If it contains a "/" (e.g.
               "Qwen/Qwen3.5-27B-FP8"), vllm_host is used as the base URL
               and chat_completions mode is enabled automatically.
        vllm_host: vLLM server address, e.g. "edward@scai4.cs.ucla.edu:8000".
                   Used when model contains "/".

    Returns:
        (lmp_tabletop_ui, LMP_env)
    """
    # --- Override model in all LMP configs if provided ---
    if model is not None:
        is_local = "/" in model
        if is_local and not vllm_host:
            raise ValueError(f"Model '{model}' looks like a local model (contains '/') but no vllm_host was provided.")
        for lmp_name in config["lmp_config"]["lmps"]:
            lmp_cfg = config["lmp_config"]["lmps"][lmp_name]
            lmp_cfg["model"] = model
            if is_local:
                lmp_cfg["base_url"] = f"http://{vllm_host}/v1"
                lmp_cfg["api_key"] = "not-needed"
                lmp_cfg["api_mode"] = "chat_completions"
            if max_tokens is not None:
                lmp_cfg["max_tokens"] = max_tokens
            if context_window is not None:
                lmp_cfg["context_window"] = context_window
    project_root = Path(__file__).parent.parent.parent

    # --- 1. Motion controller ---
    motion = XArmMotionController(env, xarm_config)
    logger.info("=" * 60)
    logger.info("Init Success")
    logger.info("=" * 60)

    # --- 2. Perception (Molmo + SAM2) ---
    perception = SAMMolmoPerception(
        camera_serials=["327122079374", "317222072157"],
        calib_file=str(project_root / "transforms" / "transforms.npy"),
        sam2_checkpoint=str(project_root / "ckpt" / "sam2.1_hiera_large.pt"),
        sam2_config="sam2.1/sam2.1_hiera_l",
    )

    # --- 3. Grasp strategy ---
    if isinstance(grasp_strategy, GraspStrategy):
        strategy = grasp_strategy
    elif grasp_strategy == "graspgen":
        gripper_config = str(
            project_root / "GraspGen" / "GraspGenModels" / "checkpoints"
            / "graspgen_robotiq_2f_140.yml"
        )
        strategy = GraspGenStrategy(gripper_config)
    elif grasp_strategy == "hardcode":
        strategy = HardcodeStrategy()
    else:
        raise ValueError(f"Unknown grasp strategy: {grasp_strategy}")

    # --- 4. Visualizer ---
    visualizer = GraspVisualizer()

    # --- 5. Pipeline ---
    pipeline = GraspPipeline(
        perception=perception,
        grasp_strategy=strategy,
        visualizer=visualizer,
    )

    # --- 6. LMP wrapper ---
    LMP_env = LMPWrapper(
        motion=motion,
        pipeline=pipeline,
        visualizer=visualizer,
    )

    # --- 7. Build variable_vars for LLM-generated code ---
    fixed_vars = {
        "np": np,
        "time": __import__("time"),
    }
    fixed_vars.update(
        {name: getattr(shapely.geometry, name) for name in shapely.geometry.__all__}
    )
    fixed_vars.update(
        {name: getattr(shapely.affinity, name) for name in shapely.affinity.__all__}
    )

    variable_vars = {
        # Core robot functions
        "get_robot_pos": LMP_env.get_robot_pos,
        "get_robot_xy": LMP_env.get_robot_xy,
        "goto_pos": LMP_env.goto_pos,
        "goto_xy": LMP_env.goto_xy,
        "move_relative": LMP_env.move_relative,
        "move_up": LMP_env.move_up,
        "move_down": LMP_env.move_down,
        "pick_place": LMP_env.pick_place,
        "follow_traj": LMP_env.follow_traj,
        "wait": LMP_env.wait,
        # Gripper
        "open_gripper": LMP_env.open_gripper,
        "close_gripper": LMP_env.close_gripper,
        "set_gripper": LMP_env.set_gripper,
        "is_gripper_open": LMP_env.is_gripper_open,
        "is_gripper_closed": LMP_env.is_gripper_closed,
        # Object queries
        "get_obj_pos": LMP_env.get_obj_pos,
        "get_obj_names": LMP_env.get_obj_names,
        "is_obj_visible": LMP_env.is_obj_visible,
        "put_first_on_second": LMP_env.put_first_on_second,
        "denormalize_xy": LMP_env.denormalize_xy,
        "get_corner_name": LMP_env.get_corner_name,
        "get_side_name": LMP_env.get_side_name,
        "get_bbox": LMP_env.get_bbox,
        "get_color": LMP_env.get_color,
        "get_object_center": LMP_env.get_object_center,
        "clear_detection_cache": LMP_env.clear_detection_cache,
        # Utility
        "say": lambda msg: print(f"robot says: {msg}"),
    }

    # Sub-LMPs
    lmp_fgen = LMPFGen(config["lmp_config"]["lmps"]["fgen"], fixed_vars, variable_vars)

    variable_vars.update(
        {
            k: LMP(
                k, config["lmp_config"]["lmps"][k], lmp_fgen, fixed_vars, variable_vars
            )
            for k in [
                "parse_obj_name",
                "parse_position",
                "parse_question",
                "transform_shape_pts",
            ]
        }
    )

    print(fixed_vars)
    print("=" * 40)
    print(variable_vars)

    if plan_mode == "json":
        from cap.lmp.vh_action_adapter import VirtualHomeActionAdapter
        from cap.lmp.utils import load_config

        tabletop_cfg = config["lmp_config"]["lmps"]["tabletop_ui_json"]
        action_map_fname = tabletop_cfg.get("action_map_fname", "action_map")
        action_map_path = project_root / "configs" / f"{action_map_fname}.yaml"
        action_map = load_config(str(action_map_path))
        adapter = VirtualHomeActionAdapter(LMP_env)
        tabletop_cfg["_json_adapter"] = adapter
        tabletop_cfg["_action_map"] = action_map

        lmp_tabletop_ui = LMP(
            "tabletop_ui_json",
            tabletop_cfg,
            lmp_fgen,
            fixed_vars,
            variable_vars,
        )
    elif plan_mode == "python":
        lmp_tabletop_ui = LMP(
            "tabletop_ui",
            config["lmp_config"]["lmps"]["tabletop_ui"],
            lmp_fgen,
            fixed_vars,
            variable_vars,
        )
    else:
        raise ValueError(f"Unknown plan_mode: {plan_mode!r} (expected 'python' or 'json')")

    return lmp_tabletop_ui, LMP_env
