"""
Hand-written trajectory adapter for VirtualHome JSON dispatch.

Each trajectory is a plain text file in a directory. The filename (without
.txt) IS the adapter method name — e.g. ``put_in.txt`` overrides the
``put_in`` method, which VH action PUTIN maps to via configs/action_map.yaml.

Trajectory file format (one step per line):

    # comments start with '#'
    x, y, z, rx, ry, rz              ← move to absolute pose (mm, degrees)
    x, y, z, rx, ry, rz, duration    ← same, custom duration in seconds
    OPEN                             ← open gripper
    CLOSE                            ← close gripper
    WAIT 0.5                         ← sleep this many seconds
    END                              ← stop parsing (rest of file ignored)

Example put_in.txt:

    # Drop an object into the dishwasher
    393.2, 0.0, 250.0, 180.0, 0.0, 0.0, 3.0
    393.2, 0.0, 100.0, 180.0, 0.0, 0.0, 2.0
    OPEN
    WAIT 0.5
    393.2, 0.0, 250.0, 180.0, 0.0, 0.0, 2.0
    END

Wiring: pass a ``hardcoded_traj_dir`` into ``setup_LMP`` so the JSON
dispatcher uses ``HardcodedActionAdapter`` instead of the bare
``VirtualHomeActionAdapter``. Actions with no matching .txt file fall back
to the parent's real implementation.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import yaml

from cap.lmp.lmp_wrapper import LMPWrapper
from cap.lmp.vh_action_adapter import VirtualHomeActionAdapter

logger = logging.getLogger(__name__)

OVERRIDES_FILENAME = "overrides.yaml"

DEFAULT_MOVE_DURATION = 3.0


def _parse_traj(path: Path) -> list[dict]:
    """Parse a trajectory file into a list of step dicts."""
    steps: list[dict] = []
    with path.open() as f:
        for lineno, raw in enumerate(f, 1):
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue

            upper = line.upper()
            if upper == "END":
                break
            if upper == "OPEN":
                steps.append({"type": "open"})
                continue
            if upper == "CLOSE":
                steps.append({"type": "close"})
                continue
            if upper == "HOME":
                steps.append({"type": "home"})
                continue
            if upper.startswith("WAIT"):
                tokens = line.split()
                if len(tokens) != 2:
                    raise ValueError(
                        f"{path}:{lineno}: WAIT needs one duration, got {line!r}"
                    )
                steps.append({"type": "wait", "duration": float(tokens[1])})
                continue

            # Otherwise: comma-separated floats (6 pose values, optional 7th = duration)
            tokens = [t.strip() for t in line.split(",") if t.strip()]
            if len(tokens) not in (6, 7):
                raise ValueError(
                    f"{path}:{lineno}: expected 6 or 7 comma-separated floats "
                    f"(x,y,z,rx,ry,rz[,duration]) or a keyword (OPEN/CLOSE/WAIT/END); "
                    f"got {line!r}"
                )
            try:
                vals = [float(t) for t in tokens]
            except ValueError as e:
                raise ValueError(f"{path}:{lineno}: bad float in {line!r} ({e})")

            pose = vals[:6]
            duration = vals[6] if len(vals) == 7 else DEFAULT_MOVE_DURATION
            steps.append({"type": "move", "pose": pose, "duration": duration})

    return steps


def exec_hardcoded_traj(env: LMPWrapper, path: str | Path) -> None:
    """Execute the trajectory at *path* through LMPWrapper primitives."""
    path = Path(path)
    steps = _parse_traj(path)
    logger.info(f"[HC] Executing {path.name} ({len(steps)} steps)")
    for i, step in enumerate(steps, 1):
        t = step["type"]
        if t == "move":
            pose = step["pose"]
            dur = step["duration"]
            logger.info(
                f"  {i}. move → ({pose[0]:.1f},{pose[1]:.1f},{pose[2]:.1f}) "
                f"rpy=({pose[3]:.1f},{pose[4]:.1f},{pose[5]:.1f}) dt={dur}s"
            )
            env.goto_pos(pose, duration=dur)
        elif t == "open":
            logger.info(f"  {i}. open_gripper")
            env.open_gripper()
        elif t == "close":
            logger.info(f"  {i}. close_gripper")
            env.close_gripper()
        elif t == "wait":
            logger.info(f"  {i}. wait {step['duration']}s")
            time.sleep(step["duration"])
        elif t == "home":
            # Fire the joint-mode HOME command (same as home.py / _go_home).
            try:
                robot = env.motion.env.robot
            except AttributeError:
                logger.warning(f"  {i}. home — no env.robot accessible; skipping")
                continue
            logger.info(f"  {i}. home (joint-mode set_servo_angle)")
            robot.home()
            time.sleep(6.0)


class HardcodedActionAdapter(VirtualHomeActionAdapter):
    """
    Drop-in replacement for VirtualHomeActionAdapter.

    Scans ``traj_dir`` for ``*.txt`` files. For each file, the stem
    (filename minus extension) becomes a method name on this adapter whose
    body just calls ``exec_hardcoded_traj`` on that file. Runtime args are
    logged and ignored — each trajectory is fully scripted.

    Actions without a matching .txt file fall through to the parent class's
    real implementation.
    """

    def __init__(
        self,
        env: LMPWrapper,
        traj_dir: str | Path,
        approach_offset_mm: float = 100.0,
    ):
        super().__init__(env, approach_offset_mm=approach_offset_mm)
        # Per-instance copies — start empty, then merge whatever the task's
        # overrides.yaml supplies. This keeps tasks fully isolated.
        self.GRAB_Y_OFFSETS_MM = {}
        self.GRAB_Z_OFFSETS_MM = {}
        self.GRAB_POSES_MM_DEG = {}
        self.POST_GRAB_LIFT_MM = {}
        self.POST_GRAB_BACK_OUT_MM = {}
        self.POST_GRAB_HOME = set()
        self.POST_GRAB_TRANSIT_POSE_MM_DEG = {}
        self.TOP_DOWN_GRAB = set()
        self.PLACE_POSES_MM_DEG = {}
        self.PLACE_Z_EXTRA_MM = {}
        self.SOAK_POSES_MM_DEG = {}
        self.POUR_ACTIONS = {}
        self.PERCEPTION_OFFSETS_MM = {}
        # Per-(action, target) hardcoded trajectory file paths (see
        # VirtualHomeActionAdapter.{OPEN,CLOSE,PUT_IN,PUT_ON}_TRAJ_FILES).
        self.OPEN_TRAJ_FILES = {}
        self.CLOSE_TRAJ_FILES = {}
        self.GRAB_TRAJ_FILES = {}
        self.PUT_IN_TRAJ_FILES = {}
        self.PUT_ON_TRAJ_FILES = {}
        # Set of trajectory file paths that are referenced by action_traj_files;
        # we skip these in the global setattr loop (they're not method overrides).
        self._per_target_traj_paths: set[Path] = set()
        self._go_home_after_place = False

        self._traj_dir = Path(traj_dir)
        if not self._traj_dir.is_dir():
            logger.warning(
                f"[HardcodedActionAdapter] {self._traj_dir} is not a directory; "
                "no overrides installed."
            )
            return

        self._load_overrides()

        installed: list[str] = []
        for txt_path in sorted(self._traj_dir.glob("*.txt")):
            method_name = txt_path.stem
            # *_motion.txt files are loaded explicitly by spray/soak/clean
            # adapter methods (perception + approach + motion). Don't
            # install them as standalone method overrides.
            if method_name.endswith("_motion"):
                continue
            # Files referenced by action_traj_files (per-target trajectories)
            # are dispatched at runtime by the adapter methods, not installed
            # as global method overrides.
            if txt_path.resolve() in self._per_target_traj_paths:
                continue
            _parse_traj(txt_path)  # validate at load time; surfaces format errors early
            setattr(self, method_name, self._make_player(method_name, txt_path))
            installed.append(method_name)

        if installed:
            logger.info(
                f"[HardcodedActionAdapter] Installed {len(installed)} trajectory "
                f"override(s) from {self._traj_dir}: {installed}"
            )

    def _load_overrides(self) -> None:
        """Load task-specific GRAB / PLACE overrides from overrides.yaml in
        the traj_dir, if present. The YAML is optional and may contain any
        subset of: grab_y_offsets_mm, place_poses_mm_deg, place_z_extra_mm.
        """
        path = self._traj_dir / OVERRIDES_FILENAME
        if not path.is_file():
            return
        with path.open() as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise ValueError(f"{path} must be a YAML mapping at top level")

        self.GRAB_Y_OFFSETS_MM.update(
            {str(k).strip().lower(): float(v)
             for k, v in (data.get("grab_y_offsets_mm") or {}).items()}
        )
        self.GRAB_Z_OFFSETS_MM.update(
            {str(k).strip().lower(): float(v)
             for k, v in (data.get("grab_z_offsets_mm") or {}).items()}
        )
        self.GRAB_POSES_MM_DEG.update(
            {str(k).strip().lower(): [float(x) for x in v]
             for k, v in (data.get("grab_poses_mm_deg") or {}).items()}
        )
        self.POST_GRAB_LIFT_MM.update(
            {str(k).strip().lower(): float(v)
             for k, v in (data.get("post_grab_lift_mm") or {}).items()}
        )
        self.POST_GRAB_BACK_OUT_MM.update(
            {str(k).strip().lower(): float(v)
             for k, v in (data.get("post_grab_back_out_mm") or {}).items()}
        )
        for k in (data.get("post_grab_home") or []):
            self.POST_GRAB_HOME.add(str(k).strip().lower())
        self.POST_GRAB_TRANSIT_POSE_MM_DEG.update(
            {str(k).strip().lower(): [float(x) for x in v]
             for k, v in (data.get("post_grab_transit_pose_mm_deg") or {}).items()}
        )
        for k in (data.get("top_down_grab") or []):
            self.TOP_DOWN_GRAB.add(str(k).strip().lower())
        # Each value can be either a 6-list (any destination) or a dict of
        # {destination: 6-list} for per-destination poses.
        place_poses_raw = data.get("place_poses_mm_deg") or {}
        for k, v in place_poses_raw.items():
            key = str(k).strip().lower()
            if isinstance(v, dict):
                self.PLACE_POSES_MM_DEG[key] = {
                    str(dk).strip().lower(): [float(x) for x in dv]
                    for dk, dv in v.items()
                }
            else:
                self.PLACE_POSES_MM_DEG[key] = [float(x) for x in v]
        self.PLACE_Z_EXTRA_MM.update(
            {str(k).strip().lower(): float(v)
             for k, v in (data.get("place_z_extra_mm") or {}).items()}
        )
        self.PERCEPTION_OFFSETS_MM.update(
            {str(k).strip().lower(): [float(x) for x in v]
             for k, v in (data.get("perception_offsets_mm") or {}).items()}
        )
        for k, v in (data.get("soak_poses_mm_deg") or {}).items():
            key = str(k).strip().lower()
            self.SOAK_POSES_MM_DEG[key] = {
                "above": [float(x) for x in v["above"]],
                "dip": [float(x) for x in v["dip"]],
                "cycles": int(v.get("cycles", 2)),
            }
        for obj_k, dest_map in (data.get("pour_actions") or {}).items():
            obj_key = str(obj_k).strip().lower()
            self.POUR_ACTIONS[obj_key] = {}
            for dest_k, cfg in dest_map.items():
                dest_key = str(dest_k).strip().lower()
                after = []
                for step in (cfg.get("after_pour") or []):
                    after.append({
                        "pose": [float(x) for x in step["pose"]],
                        "duration": float(step.get("duration", 3.0)),
                        "release": bool(step.get("release", False)),
                    })
                self.POUR_ACTIONS[obj_key][dest_key] = {
                    "tilt_pitch_deg": float(cfg.get("tilt_pitch_deg", 30.0)),
                    "cycles": int(cfg.get("cycles", 1)),
                    "hold_s": float(cfg.get("hold_s", 0.5)),
                    "height_offset_mm": float(cfg.get("height_offset_mm", 152.0)),
                    "leg_s": float(cfg.get("leg_s", 0.6)),
                    "after_pour": after,
                }
        # Per-(action, target) hardcoded trajectory file mapping. Schema:
        #   action_traj_files:
        #     open:  {target_name: filename.txt, ...}
        #     close: {target_name: filename.txt, ...}
        #     put_in:
        #       obj_name: {dest_name: filename.txt, ...}
        #     put_on:
        #       obj_name: {dest_name: filename.txt, ...}
        # File names are resolved relative to the traj_dir.
        atf = data.get("action_traj_files") or {}
        if not isinstance(atf, dict):
            raise ValueError(f"{path}: action_traj_files must be a mapping")
        for action_key, dest_attr in (
            ("open", self.OPEN_TRAJ_FILES),
            ("close", self.CLOSE_TRAJ_FILES),
            ("grab", self.GRAB_TRAJ_FILES),
        ):
            for target_name, fname in (atf.get(action_key) or {}).items():
                fpath = (self._traj_dir / str(fname)).resolve()
                _parse_traj(fpath)  # validate at load time
                dest_attr[str(target_name).strip().lower()] = str(fpath)
                self._per_target_traj_paths.add(fpath)
        for action_key, dest_attr in (
            ("put_in", self.PUT_IN_TRAJ_FILES),
            ("put_on", self.PUT_ON_TRAJ_FILES),
        ):
            for obj_name, dest_map in (atf.get(action_key) or {}).items():
                if not isinstance(dest_map, dict):
                    raise ValueError(
                        f"{path}: action_traj_files.{action_key}.{obj_name} "
                        f"must be a mapping of dest→filename"
                    )
                obj_key = str(obj_name).strip().lower()
                inner: dict[str, str] = {}
                for dest_name, fname in dest_map.items():
                    fpath = (self._traj_dir / str(fname)).resolve()
                    _parse_traj(fpath)
                    inner[str(dest_name).strip().lower()] = str(fpath)
                    self._per_target_traj_paths.add(fpath)
                dest_attr[obj_key] = inner
        # Flag — when True, after a hardcoded place, the run loop's HOME
        # command fires (joint-mode set_servo_angle to xarm_config.home_pos,
        # same as home.py).
        self._go_home_after_place = bool(data.get("go_home_after_place", False))

        logger.info(
            f"[HardcodedActionAdapter] Loaded overrides from {path.name}: "
            f"grab_y_offsets={list(self.GRAB_Y_OFFSETS_MM)}, "
            f"grab_z_offsets={list(self.GRAB_Z_OFFSETS_MM)}, "
            f"grab_poses={list(self.GRAB_POSES_MM_DEG)}, "
            f"post_grab_lift={list(self.POST_GRAB_LIFT_MM)}, "
            f"post_grab_transit={list(self.POST_GRAB_TRANSIT_POSE_MM_DEG)}, "
            f"top_down_grab={sorted(self.TOP_DOWN_GRAB)}, "
            f"place_poses={list(self.PLACE_POSES_MM_DEG)}, "
            f"place_z_extra={list(self.PLACE_Z_EXTRA_MM)}, "
            f"open_trajs={list(self.OPEN_TRAJ_FILES)}, "
            f"close_trajs={list(self.CLOSE_TRAJ_FILES)}, "
            f"grab_trajs={list(self.GRAB_TRAJ_FILES)}, "
            f"put_in_trajs={ {k: list(v) for k, v in self.PUT_IN_TRAJ_FILES.items()} }, "
            f"put_on_trajs={ {k: list(v) for k, v in self.PUT_ON_TRAJ_FILES.items()} }"
        )

    def _make_player(self, method_name: str, path: Path):
        env = self._env

        def play(*args: Any, **kwargs: Any):
            if args or kwargs:
                logger.info(
                    f"[HC] {method_name}({args}) — args ignored (hardcoded trajectory)"
                )
            exec_hardcoded_traj(env, path)

        play.__name__ = method_name
        return play
