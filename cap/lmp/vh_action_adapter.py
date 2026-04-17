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
from typing import Optional

import numpy as np

from cap.lmp.lmp_wrapper import LMPWrapper

logger = logging.getLogger(__name__)


class VirtualHomeActionAdapter:
    def __init__(self, env: LMPWrapper, approach_offset_mm: float = 100.0):
        self._env = env
        self._approach_offset_mm = float(approach_offset_mm)

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
        pose = self._env.get_obj_pos(name)
        pos = np.array(pose[:3], dtype=np.float64)
        above = pos + np.array([0.0, 0.0, self._approach_offset_mm])
        logger.info(f"[VH] GRAB → {name}")
        self._env.goto_pos(above.tolist())
        self._env.open_gripper()
        self._env.goto_pos(pose)
        self._env.close_gripper()
        self._env.goto_pos(above.tolist())

    def put_on(self, obj_name: str, obj_id: Optional[str],
               surface_name: str, surface_id: Optional[str] = None):
        pick = self._env.get_obj_pos(obj_name)
        place = self._env.get_obj_pos(surface_name)
        logger.info(f"[VH] PUTON {obj_name} → {surface_name}")
        self._env.pick_place(pick, place)

    def put_in(self, obj_name: str, obj_id: Optional[str],
               container_name: str, container_id: Optional[str] = None):
        pick = self._env.get_obj_pos(obj_name)
        place = self._env.get_obj_pos(container_name)
        logger.info(f"[VH] PUTIN {obj_name} → {container_name}")
        self._env.pick_place(pick, place)

    # ------------------------------------------------------------------
    # Stubs — no tabletop analog yet, kept so plans don't crash
    # ------------------------------------------------------------------

    def open(self, name: str, id: Optional[str] = None):
        logger.info(f"[VH] OPEN {name} (noop — no manipulator analog)")

    def close(self, name: str, id: Optional[str] = None):
        logger.info(f"[VH] CLOSE {name} (noop — no manipulator analog)")

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
    # Utility
    # ------------------------------------------------------------------

    def say(self, message: str):
        print(f"robot says: {message}")
