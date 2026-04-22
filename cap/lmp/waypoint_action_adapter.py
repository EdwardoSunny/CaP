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

from cap.lmp.lmp_wrapper import LMPWrapper
from cap.lmp.vh_action_adapter import VirtualHomeActionAdapter

logger = logging.getLogger(__name__)

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
        self._traj_dir = Path(traj_dir)
        if not self._traj_dir.is_dir():
            logger.warning(
                f"[HardcodedActionAdapter] {self._traj_dir} is not a directory; "
                "no overrides installed."
            )
            return

        installed: list[str] = []
        for txt_path in sorted(self._traj_dir.glob("*.txt")):
            method_name = txt_path.stem
            _parse_traj(txt_path)  # validate at load time; surfaces format errors early
            setattr(self, method_name, self._make_player(method_name, txt_path))
            installed.append(method_name)

        if installed:
            logger.info(
                f"[HardcodedActionAdapter] Installed {len(installed)} trajectory "
                f"override(s) from {self._traj_dir}: {installed}"
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
