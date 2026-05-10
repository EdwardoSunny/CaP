"""
VirtualHome-style action adapter for CaP.

Bridges VirtualHome AS-style JSON plans (WALK, GRAB, PUTIN, ...) to the
real-robot primitives exposed by LMPWrapper. Each method below takes the
positional args from the JSON entry and composes LMPWrapper calls.

Object integer IDs from the VH schema are accepted but ignored — CaP
resolves objects by name via perception.

Actions with no real-robot analog (OPEN/CLOSE container, SWITCHON, etc.)
are implemented as no-op logs so a VH plan runs without crashing; edit
these when a real behavior is available.
"""

from __future__ import annotations

import logging
import math
import time
from typing import Optional

import numpy as np

from cap.lmp.lmp_wrapper import LMPWrapper

logger = logging.getLogger(__name__)


class VirtualHomeActionAdapter:
    # Class-level defaults are empty. Per-task overrides live in
    # configs/hardcoded_traj/<task>/overrides.yaml and are loaded by
    # HardcodedActionAdapter at construction time as instance attributes.
    GRAB_Y_OFFSETS_MM: dict = {}
    GRAB_Z_OFFSETS_MM: dict = {}     # Added on top of perception's +30 mm; negative descends deeper
    GRAB_POSES_MM_DEG: dict = {}     # Absolute hardcoded grab pose (bypasses perception)
    POST_GRAB_LIFT_MM: dict = {}     # Per-object +Z lift (mm) after a hardcoded grab
    POST_GRAB_BACK_OUT_MM: dict = {} # Per-object X delta (mm, negative = retreat away)
                                     # applied AFTER the lift, same orientation/Y/Z
    POST_GRAB_HOME: set = set()      # Object names that trigger joint-mode HOME after the back-out
    POST_GRAB_TRANSIT_POSE_MM_DEG: dict = {}  # Per-object staging pose after the lift
    # Object names that should use a forced top-down [180, 0, 0] orientation
    # for the perception-driven grab — overrides any object-aligned yaw.
    TOP_DOWN_GRAB: set = set()
    PLACE_POSES_MM_DEG: dict = {}
    PLACE_Z_EXTRA_MM: dict = {}
    # Per-target hardcoded SOAK trajectory: dict of {above: [6dof], dip: [6dof], cycles: int}.
    # When set, perception is skipped and SOAK just plays above → (dip → above) × cycles.
    SOAK_POSES_MM_DEG: dict = {}
    # Per (obj, container) PUTIN/PUTON override that runs a tilt-pour over the
    # perceived target instead of dropping (gripper stays closed during pour),
    # then plays a custom return sequence with an optional release step.
    # Schema: {obj_lower: {container_lower: {tilt_pitch_deg, cycles, hold_s,
    #                                        after_pour: [{pose:[6], release?}]}}}
    POUR_ACTIONS: dict = {}
    # Per-(action, target) hardcoded trajectory file paths.
    # When a method (open / close / put_in / put_on) fires with a target name
    # that matches one of these maps, that trajectory file is played instead
    # of the default stub or perception path.
    OPEN_TRAJ_FILES: dict = {}     # target_lower → str(path)
    CLOSE_TRAJ_FILES: dict = {}    # target_lower → str(path)
    GRAB_TRAJ_FILES: dict = {}     # target_lower → str(path)
    PUT_IN_TRAJ_FILES: dict = {}   # obj_lower → {dest_lower → str(path)}
    PUT_ON_TRAJ_FILES: dict = {}   # obj_lower → {dest_lower → str(path)}
    # Per-target [dx, dy, dz] correction applied to perceived object positions
    # used by soak / clean / spray. Compensates for systematic perception bias
    # on specific objects.
    PERCEPTION_OFFSETS_MM: dict = {}

    # Motion parameters for compound actions (spray/soak/clean). Tune here.
    # SOAK: approach high above the cup to clear its rim AND clear any rag
    # hanging from the gripper above other objects on the table, then descend
    # to a target height above the perceived cup centroid.
    SOAK_APPROACH_HEIGHT_MM: float = 400.0   # ~16" above perceived cup — high transit
    SOAK_DIP_HEIGHT_MM: float = 100.0        # ~4" above perceived — deeper dip (~12" down from approach)
    # CLEAN: approach high above the target to keep the held rag clear of
    # other tabletop objects, then descend to wipe height for the circular
    # motion, then retreat back up before the next action.
    CLEAN_APPROACH_HEIGHT_MM: float = 250.0  # ~10" transit height
    CLEAN_HOVER_OFFSET_MM: float = 50.0      # ~2" — actual wipe height above target (was 3", now 1" lower)
    CLEAN_CIRCLE_RADIUS_MM: float = 50.0     # ~2 inches
    CLEAN_CIRCLE_STEPS: int = 8              # waypoints around the circle
    SPRAY_HEIGHT_OFFSET_MM: float = 152.0     # 6 inches above target
    SPRAY_HOLD_S: float = 0.5             # brief settle above target before pouring
    SPRAY_PITCH_AMPLITUDE_DEG: float = 10.0   # ± oscillation around CURRENT (held) pitch
    SPRAY_PITCH_OSCILLATIONS: int = 3         # number of forward+back cycles
    SPRAY_PITCH_LEG_S: float = 0.6            # duration per pitch leg
    POST_PLACE_RETREAT_HEIGHT_MM: float = 200.0   # ~8" above release pose; clears the just-placed object

    def __init__(self, env: LMPWrapper, approach_offset_mm: float = 100.0):
        self._env = env
        self._approach_offset_mm = float(approach_offset_mm)
        # Set by HardcodedActionAdapter subclass; used by spray/soak/clean
        # to find their <method>_motion.txt files.
        self._traj_dir = None

    # ------------------------------------------------------------------
    # Motion
    # ------------------------------------------------------------------

    def walk(self, name: str, id: Optional[str] = None):
        pose = self._env.get_obj_pos(name)
        pos = np.array(pose[:3], dtype=np.float64)
        above = pos + np.array([0.0, 0.0, self._approach_offset_mm])
        logger.info(f"[VH] WALK → {name}")
        self._env.goto_pos(above.tolist())

    def grab(self, name: str, id: Optional[str] = None):
        # Per-target hardcoded multi-step trajectory wins over everything.
        if self._try_per_target_traj("grab", name):
            return
        # Hardcoded absolute grab pose: just go to the pose and close gripper.
        # No approach-from-above; the pose itself is where the gripper closes.
        key = name.strip().lower()
        hardcoded = self.GRAB_POSES_MM_DEG.get(key)
        if hardcoded is not None:
            lift = float(self.POST_GRAB_LIFT_MM.get(key, 0.0))
            back_out = float(self.POST_GRAB_BACK_OUT_MM.get(key, 0.0))
            do_home = key in self.POST_GRAB_HOME
            transit = self.POST_GRAB_TRANSIT_POSE_MM_DEG.get(key)
            notes = ["goto hardcoded pose, close gripper"]
            if lift:
                notes.append(f"lift {lift:.1f} mm")
            if back_out:
                notes.append(f"back out X{back_out:+.1f} mm")
            if do_home:
                notes.append("HOME")
            if transit is not None:
                notes.append(f"transit to {list(transit)}")
            logger.info(f"[VH] GRAB → {name}  ({'; '.join(notes)})")

            # 1. goto pick pose
            self._env.goto_pos(list(hardcoded), duration=3.0)
            # 2. close gripper
            self._env.close_gripper()
            # 3. optional Z lift (same XY/orientation)
            cur = list(hardcoded)
            if lift:
                cur[2] += lift
                self._env.goto_pos(cur, duration=2.0)
            # 4. optional back-out in X (same Y/Z/orientation)
            if back_out:
                cur[0] += back_out
                self._env.goto_pos(cur, duration=2.0)
            # 5. optional joint-mode HOME
            if do_home:
                try:
                    robot = self._env.motion.env.robot
                    logger.info("  [VH] firing joint-mode HOME after grab")
                    robot.home()
                    time.sleep(6.0)
                except AttributeError:
                    logger.warning("  [VH] no env.robot; HOME skipped")
            # 6. optional staging pose (only if no HOME — they conflict)
            if transit is not None and not do_home:
                self._env.goto_pos(list(transit), duration=3.0)
            return
        # Otherwise: perception + optional Y/Z offsets
        pose = list(self._env.get_obj_pos(name))
        y_offset = self.GRAB_Y_OFFSETS_MM.get(key, 0.0)
        z_offset = self.GRAB_Z_OFFSETS_MM.get(key, 0.0)
        if y_offset:
            pose[1] += y_offset
        if z_offset:
            pose[2] += z_offset
        # Force top-down orientation if this object is in TOP_DOWN_GRAB —
        # ignores object-aligned yaw / IK rotation guesses.
        force_top_down = key in self.TOP_DOWN_GRAB
        if force_top_down:
            pose = list(pose[:3]) + [180.0, 0.0, 0.0]
        notes = []
        if y_offset:
            notes.append(f"y {y_offset:+.1f} mm")
        if z_offset:
            notes.append(f"z {z_offset:+.1f} mm")
        if force_top_down:
            notes.append("top-down orientation")
        suffix = f"  ({', '.join(notes)})" if notes else ""
        logger.info(f"[VH] GRAB → {name}{suffix}")
        self._env.pick(pose, approach_offset=self._approach_offset_mm)

    def put_on(self, obj_name: str, obj_id: Optional[str],
               surface_name: str, surface_id: Optional[str] = None):
        # Assumes a preceding GRAB already placed the object in the gripper.
        # Home-return only fires after a HARDCODED place (typically the stool
        # drop-off poses); perception-path placements don't trigger it, which
        # keeps mid-plan PUT actions (e.g. baseline's PUTIN rag in cup) from
        # bouncing the arm back to home before the next action.
        if self._try_per_target_put_traj("put_on", obj_name, surface_name):
            return
        if self._try_pour_action(obj_name, surface_name, "PUTON"):
            return
        if self._try_hardcoded_place(obj_name, dest_label=surface_name, vh_action="PUTON"):
            self._go_home()
            return
        pose = self._env.get_obj_pos(surface_name)
        place = self._apply_perception_offset(surface_name, pose[:3]).tolist()
        place_offset = 50.0 + self.PLACE_Z_EXTRA_MM.get(obj_name.strip().lower(), 0.0)
        logger.info(f"[VH] PUTON {obj_name} → {surface_name}  (release {place_offset:.1f} mm above)")
        self._env.place(place, place_offset=place_offset)

    def put_in(self, obj_name: str, obj_id: Optional[str],
               container_name: str, container_id: Optional[str] = None):
        if self._try_per_target_put_traj("put_in", obj_name, container_name):
            return
        if self._try_pour_action(obj_name, container_name, "PUTIN"):
            return
        if self._try_hardcoded_place(obj_name, dest_label=container_name, vh_action="PUTIN"):
            self._go_home()
            return
        pose = self._env.get_obj_pos(container_name)
        place = self._apply_perception_offset(container_name, pose[:3]).tolist()
        place_offset = 50.0 + self.PLACE_Z_EXTRA_MM.get(obj_name.strip().lower(), 0.0)
        logger.info(f"[VH] PUTIN {obj_name} → {container_name}  (release {place_offset:.1f} mm above)")
        self._env.place(place, place_offset=place_offset)

    def _try_pour_action(self, obj_name: str, dest_name: str, vh_action: str) -> bool:
        """If POUR_ACTIONS has an entry for (obj_name, dest_name), run a
        tilt-pour over the perceived destination then play the after_pour
        sequence. Returns True if it fired, False if no entry."""
        obj_key = obj_name.strip().lower()
        dest_key = dest_name.strip().lower()
        per_obj = self.POUR_ACTIONS.get(obj_key)
        if not per_obj:
            return False
        cfg = per_obj.get(dest_key)
        if not cfg:
            return False

        # Resolve perceived destination pose
        target_pose = self._env.get_obj_pos(dest_name)
        target_pos = self._apply_perception_offset(dest_name, target_pose[:3])

        # Current EEF orientation (preserves grab orientation through transit)
        try:
            current = self._env.get_robot_pose()
        except AttributeError:
            current = self._env.motion.get_robot_pose()
        roll, pitch, yaw = float(current[3]), float(current[4]), float(current[5])

        tilt_deg = float(cfg.get("tilt_pitch_deg", 30.0))
        cycles = int(cfg.get("cycles", 1))
        hold_s = float(cfg.get("hold_s", 0.5))
        height_offset = float(cfg.get("height_offset_mm", self.SPRAY_HEIGHT_OFFSET_MM))
        leg_s = float(cfg.get("leg_s", self.SPRAY_PITCH_LEG_S))

        above_xyz = (target_pos + np.array([0.0, 0.0, height_offset])).tolist()
        above_pose = above_xyz + [roll, pitch, yaw]

        logger.info(
            f"[VH] {vh_action} {obj_name} → {dest_name}  POUR "
            f"(hover {height_offset:.0f} mm above, oscillate pitch ±{tilt_deg:.1f}° × {cycles})"
        )
        # Hover above target
        self._env.goto_pos(above_pose, duration=2.0)
        time.sleep(hold_s)
        # Tilt-pour: pitch +tilt → -tilt, repeated
        for c in range(cycles):
            logger.info(f"  pour cycle {c+1}/{cycles}: pitch → {pitch + tilt_deg:.1f}°")
            self._env.goto_pos(above_xyz + [roll, pitch + tilt_deg, yaw], duration=leg_s)
            logger.info(f"  pour cycle {c+1}/{cycles}: pitch → {pitch - tilt_deg:.1f}°")
            self._env.goto_pos(above_xyz + [roll, pitch - tilt_deg, yaw], duration=leg_s)
        # Return to upright above target
        self._env.goto_pos(above_pose, duration=leg_s)

        # Play the after_pour sequence (waypoints + optional release at any step)
        for i, step in enumerate(cfg.get("after_pour", []) or [], 1):
            pose = step["pose"]
            duration = float(step.get("duration", 3.0))
            do_release = bool(step.get("release", False))
            logger.info(
                f"  after_pour {i}: goto {pose}"
                + (" + release" if do_release else "")
            )
            self._env.goto_pos(list(pose), duration=duration)
            if do_release:
                self._env.open_gripper()
                time.sleep(0.5)
        return True

    def _go_home(self):
        """Send the same HOME command home.py uses: joint-mode set_servo_angle
        to xarm_config.home_pos. Sleeps long enough for the joint motion to
        complete before returning. No-op if go_home_after_place is disabled."""
        if not getattr(self, "_go_home_after_place", False):
            return
        try:
            robot = self._env.motion.env.robot
        except AttributeError:
            logger.warning("[VH] _go_home: no env.robot accessible; skipping")
            return
        logger.info("[VH] Returning to home (joint-mode set_servo_angle)")
        robot.home()
        # set_servo_angle(wait=True) inside the run loop blocks while the
        # joint motion completes; we need to wait long enough here for that
        # to happen before any subsequent action fires. ~6 s is generous.
        time.sleep(6.0)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _apply_perception_offset(self, target_name: str, xyz) -> np.ndarray:
        """Apply per-object [dx, dy, dz] correction to a perceived xyz."""
        pos = np.array(xyz[:3], dtype=np.float64)
        offset = self.PERCEPTION_OFFSETS_MM.get(target_name.strip().lower())
        if offset is not None:
            pos = pos + np.array(offset, dtype=np.float64)
            logger.info(
                f"  [perception offset] {target_name!r}: "
                f"dx={offset[0]:+.1f} dy={offset[1]:+.1f} dz={offset[2]:+.1f} mm"
            )
        return pos

    def _try_hardcoded_place(self, obj_name: str, dest_label: str, vh_action: str) -> bool:
        """If (obj_name, dest_label) has an entry in PLACE_POSES_MM_DEG, run
        a 6-DOF approach → release → retreat at that pose and return True.

        PLACE_POSES_MM_DEG values can be either:
          * a 6-element list  → applies to any destination (legacy / wildcard)
          * a dict of {dest_label: 6-list}  → per-destination poses
        """
        entry = self.PLACE_POSES_MM_DEG.get(obj_name.strip().lower())
        if entry is None:
            return False

        if isinstance(entry, dict):
            target_pose = entry.get(dest_label.strip().lower())
            if target_pose is None:
                # Object has hardcoded poses but not for this destination →
                # fall through to perception.
                return False
        else:
            target_pose = entry  # legacy: single list applies to any dest

        target_pose = list(target_pose)
        approach_pose = list(target_pose)
        approach_pose[2] += self._approach_offset_mm
        retreat_pose = list(target_pose)
        retreat_pose[2] += self.POST_PLACE_RETREAT_HEIGHT_MM
        logger.info(
            f"[VH] {vh_action} {obj_name} → {dest_label!r}  hardcoded pose {target_pose} "
            f"(retreat {self.POST_PLACE_RETREAT_HEIGHT_MM:.0f} mm above)"
        )
        self._env.goto_pos(approach_pose, duration=3.0)
        self._env.goto_pos(target_pose, duration=2.0)
        self._env.open_gripper()
        time.sleep(0.5)
        self._env.goto_pos(retreat_pose, duration=2.0)
        return True

    # ------------------------------------------------------------------
    # Stubs — no tabletop analog yet, kept so plans don't crash
    # ------------------------------------------------------------------

    def open(self, name: str, id: Optional[str] = None):
        if self._try_per_target_traj("open", name):
            return
        logger.info(f"[VH] OPEN {name} (noop — no manipulator analog)")

    def close(self, name: str, id: Optional[str] = None):
        if self._try_per_target_traj("close", name):
            return
        logger.info(f"[VH] CLOSE {name} (noop — no manipulator analog)")

    def _try_per_target_traj(self, action: str, target: str) -> bool:
        """Look up a per-target hardcoded trajectory file for OPEN/CLOSE/GRAB.
        Returns True if it ran a trajectory."""
        target_key = str(target).strip().lower()
        target_map = (
            self.OPEN_TRAJ_FILES if action == "open"
            else self.CLOSE_TRAJ_FILES if action == "close"
            else self.GRAB_TRAJ_FILES if action == "grab"
            else None
        )
        if target_map is None:
            return False
        traj = target_map.get(target_key)
        if traj is None:
            return False
        from pathlib import Path as _P
        if not _P(traj).is_file():
            logger.warning(f"[VH] {action.upper()} {target}: traj file {traj} not found; skipping")
            return False
        from cap.lmp.waypoint_action_adapter import exec_hardcoded_traj
        logger.info(f"[VH] {action.upper()} {target} → hardcoded {traj}")
        exec_hardcoded_traj(self._env, traj)
        return True

    def _try_per_target_put_traj(self, action: str, obj_name: str, dest_name: str) -> bool:
        """Look up a per-(obj, dest) hardcoded trajectory file for PUTIN/PUTON.
        Returns True if it ran a trajectory."""
        obj_key = str(obj_name).strip().lower()
        dest_key = str(dest_name).strip().lower()
        target_map = (
            self.PUT_IN_TRAJ_FILES if action == "put_in"
            else self.PUT_ON_TRAJ_FILES if action == "put_on"
            else None
        )
        if target_map is None:
            return False
        traj = (target_map.get(obj_key) or {}).get(dest_key)
        if traj is None:
            return False
        from pathlib import Path as _P
        if not _P(traj).is_file():
            logger.warning(
                f"[VH] {action.upper()} {obj_name}→{dest_name}: traj file {traj} not found; skipping"
            )
            return False
        from cap.lmp.waypoint_action_adapter import exec_hardcoded_traj
        logger.info(f"[VH] {action.upper()} {obj_name} → {dest_name}  hardcoded {traj}")
        exec_hardcoded_traj(self._env, traj)
        return True

    def switch_on(self, name: str, id: Optional[str] = None):
        logger.info(f"[VH] SWITCHON {name} (noop)")

    def switch_off(self, name: str, id: Optional[str] = None):
        logger.info(f"[VH] SWITCHOFF {name} (noop)")

    def look_at(self, name: str, id: Optional[str] = None):
        logger.info(f"[VH] LOOKAT {name} (noop)")

    def touch(self, name: str, id: Optional[str] = None):
        pose = self._env.get_obj_pos(name)
        pos = np.array(pose[:3], dtype=np.float64)
        above = pos + np.array([0.0, 0.0, self._approach_offset_mm])
        logger.info(f"[VH] TOUCH {name}")
        self._env.goto_pos(above.tolist())
        self._env.goto_pos(pose)
        self._env.goto_pos(above.tolist())

    # ------------------------------------------------------------------
    # Compound actions: perception → approach → hardcoded motion
    # ------------------------------------------------------------------

    def spray(self, target_name: str, target_id: Optional[str] = None):
        """Apply held spray bottle to target. Perception locates the target;
        EEF hovers SPRAY_HEIGHT_OFFSET_MM above it (preserving whatever
        orientation the arm has on arrival), then oscillates pitch
        ±SPRAY_PITCH_AMPLITUDE_DEG around that current pitch as a delta —
        no absolute target pose, just gentle wiggles."""
        pose = self._env.get_obj_pos(target_name)
        target_pos = self._apply_perception_offset(target_name, pose[:3])

        # Current EEF orientation — keep the bottle's held angle as we transit.
        try:
            current = self._env.get_robot_pose()
        except AttributeError:
            current = self._env.motion.get_robot_pose()
        roll, pitch, yaw = float(current[3]), float(current[4]), float(current[5])

        # Hover SPRAY_HEIGHT_OFFSET_MM above target — no descent.
        above_xyz = (target_pos + np.array([0.0, 0.0, self.SPRAY_HEIGHT_OFFSET_MM])).tolist()
        above_pose = above_xyz + [roll, pitch, yaw]

        amp = self.SPRAY_PITCH_AMPLITUDE_DEG
        n = self.SPRAY_PITCH_OSCILLATIONS
        leg = self.SPRAY_PITCH_LEG_S
        logger.info(
            f"[VH] SPRAY → {target_name}  "
            f"(hover {self.SPRAY_HEIGHT_OFFSET_MM:.1f} mm above target, "
            f"oscillate pitch ±{amp:.1f}° around current {pitch:.1f}° × {n} cycles)"
        )
        self._env.goto_pos(above_pose, duration=2.0)
        time.sleep(self.SPRAY_HOLD_S)

        for cycle in range(n):
            pitch_fwd = pitch + amp
            pitch_bwd = pitch - amp
            logger.info(f"  spray cycle {cycle+1}/{n}: pitch → {pitch_fwd:.1f}°")
            self._env.goto_pos(above_xyz + [roll, pitch_fwd, yaw], duration=leg)
            logger.info(f"  spray cycle {cycle+1}/{n}: pitch → {pitch_bwd:.1f}°")
            self._env.goto_pos(above_xyz + [roll, pitch_bwd, yaw], duration=leg)
        # Return to neutral
        logger.info(f"  spray return: pitch → {pitch:.1f}°")
        self._env.goto_pos(above_pose, duration=leg)

    def soak(self, target_name: str, target_id: Optional[str] = None):
        """Dip held object in a container (water source). If a hardcoded
        SOAK_POSES_MM_DEG entry exists for target_name, use it (skips
        perception entirely). Otherwise fall back to the perception path:
        approach high above the perceived cup, descend to SOAK_DIP_HEIGHT_MM
        above the perceived centroid, hold, ascend."""
        hardcoded = self.SOAK_POSES_MM_DEG.get(target_name.strip().lower())
        if hardcoded is not None:
            above = list(hardcoded["above"])
            dip = list(hardcoded["dip"])
            cycles = int(hardcoded.get("cycles", 2))
            logger.info(
                f"[VH] SOAK → {target_name}  (hardcoded above={above}, dip={dip}, "
                f"cycles={cycles})"
            )
            # Approach above the cup
            self._env.goto_pos(above, duration=3.0)
            # cycles × (dip → up to above)
            for i in range(cycles):
                logger.info(f"  SOAK cycle {i+1}/{cycles}: dip")
                self._env.goto_pos(dip, duration=2.0)
                logger.info(f"  SOAK cycle {i+1}/{cycles}: ascend")
                self._env.goto_pos(above, duration=2.0)
            return

        pose = self._env.get_obj_pos(target_name)
        pos = self._apply_perception_offset(target_name, pose[:3])
        above = pos + np.array([0.0, 0.0, self.SOAK_APPROACH_HEIGHT_MM])
        dip = pos + np.array([0.0, 0.0, self.SOAK_DIP_HEIGHT_MM])
        logger.info(
            f"[VH] SOAK → {target_name}  "
            f"(approach z={above[2]:.1f}, dip z={dip[2]:.1f}, "
            f"perceived z={pos[2]:.1f})"
        )
        self._env.goto_pos(above.tolist(), duration=2.0)
        self._env.goto_pos(dip.tolist(), duration=2.0)
        time.sleep(0.5)
        self._env.goto_pos(above.tolist(), duration=2.0)

    def clean(self, target_name: str, target_id: Optional[str] = None):
        """Wipe target with held object. High approach (clears other objects
        on the table while transiting), descend to wipe height, trace a
        circle, then retreat back to high approach pose."""
        pose = self._env.get_obj_pos(target_name)
        pos = self._apply_perception_offset(target_name, pose[:3])
        approach = pos.copy()
        approach[2] += self.CLEAN_APPROACH_HEIGHT_MM
        center = pos.copy()
        center[2] += self.CLEAN_HOVER_OFFSET_MM
        radius = self.CLEAN_CIRCLE_RADIUS_MM
        n = self.CLEAN_CIRCLE_STEPS
        logger.info(
            f"[VH] CLEAN → {target_name}  "
            f"(approach z={approach[2]:.1f}, wipe z={center[2]:.1f}, "
            f"circle r={radius:.1f} mm, {n} waypoints)"
        )
        # High approach above target — keeps held rag clear of other objects
        self._env.goto_pos(approach.tolist(), duration=2.0)
        # Descend to wipe height
        self._env.goto_pos(center.tolist(), duration=2.0)
        # Trace circle
        for i in range(n + 1):
            angle = 2 * math.pi * i / n
            wp = center.copy()
            wp[0] += radius * math.cos(angle)
            wp[1] += radius * math.sin(angle)
            self._env.goto_pos(wp.tolist(), duration=0.6)
        self._env.goto_pos(center.tolist(), duration=1.0)
        # Retreat back to high approach pose
        self._env.goto_pos(approach.tolist(), duration=2.0)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def say(self, message: str):
        print(f"robot says: {message}")
