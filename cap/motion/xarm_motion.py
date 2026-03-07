"""Concrete xArm motion controller extracted from lmp_wrapper.py."""

import logging
import time
from typing import List, Optional

import numpy as np
from scipy.spatial.transform import Rotation as st_Rotation
from scipy.spatial.transform import Slerp as st_Slerp

from cap.motion.base import MotionController

logger = logging.getLogger(__name__)


class XArmMotionController(MotionController):
    """Motion controller for the xArm robot.

    Implements the :class:`MotionController` interface using the same
    interpolation and timing logic that was previously embedded in
    ``LMPWrapper``.
    """

    def __init__(self, env, xarm_config, frequency=30, command_latency=0.01):
        self.env = env
        self._xarm_config = xarm_config
        self._frequency = frequency
        self._dt = 1.0 / frequency
        self._current_grasp = 0.0

    # ------------------------------------------------------------------
    # Abstract method implementations
    # ------------------------------------------------------------------

    def get_robot_pos(self) -> np.ndarray:
        """Return robot end-effector xyz position in robot base frame."""
        state = self.env.get_robot_state()
        current_pose = np.array(state["TCPPose"], dtype=np.float32)
        return current_pose[:3]

    def get_robot_pose(self) -> np.ndarray:
        """Return full robot pose [x, y, z, roll, pitch, yaw]."""
        state = self.env.get_robot_state()
        return np.array(state["TCPPose"], dtype=np.float32)

    def move_to_pose(
        self,
        target_position: List[float],
        target_orientation: List[float],
        duration: float = 3.0,
        stage_val: int = 0,
    ) -> bool:
        """Internal move to pose function using teleop script logic."""

        print("MOVING TO", target_position)
        try:
            target_pose = np.array(
                target_position + target_orientation, dtype=np.float64
            )

            # Get current pose from robot
            state = self.env.get_robot_state()
            start_pose = np.array(state["TCPPose"], dtype=np.float64)

            logger.info(f"move_to_pose():")
            logger.info(f"   Start: {start_pose[:3].tolist()}")
            logger.info(f"   Target: {target_pose[:3].tolist()}")
            logger.info(f"   Start orientation: [{start_pose[3]:.1f}, {start_pose[4]:.1f}, {start_pose[5]:.1f}]")
            logger.info(f"   Target orientation: [{target_pose[3]:.1f}, {target_pose[4]:.1f}, {target_pose[5]:.1f}]")

            # Calculate interpolation steps
            interpolation_steps = int(duration * self._frequency)

            logger.debug(
                f"Moving from {start_pose} to {target_pose} over {duration}s ({interpolation_steps} steps)"
            )

            t_start = time.monotonic()

            for iter_idx in range(interpolation_steps):
                # Same timing logic as teleop script
                t_cycle_end = t_start + (iter_idx + 1) * self._dt
                t_command_target = t_cycle_end + self._dt

                # Pump obs - same as teleop script
                obs = self.env.get_obs()

                # Interpolation with proper rotation handling
                t = (iter_idx + 1) / interpolation_steps

                # Linear interpolation for position
                interpolated_position = start_pose[:3] + t * (target_pose[:3] - start_pose[:3])

                # SLERP for orientation to avoid gimbal lock and angle wrapping
                start_rot = st_Rotation.from_euler('xyz', start_pose[3:], degrees=True)
                target_rot = st_Rotation.from_euler('xyz', target_pose[3:], degrees=True)
                interpolated_rot = st_Slerp([0, 1], st_Rotation.concatenate([start_rot, target_rot]))
                interpolated_orientation = interpolated_rot(t).as_euler('xyz', degrees=True)

                # Combine position and orientation
                interpolated_pose = np.concatenate([interpolated_position, interpolated_orientation])

                # DEBUG: Print first and last commands
                if iter_idx == 0:
                    logger.info(f"\nDEBUG: First interpolated command (t={t:.3f}):")
                    logger.info(f"   Pose: {interpolated_pose.tolist()}")
                    logger.info(f"   Orientation: [{interpolated_pose[3]:.1f}, {interpolated_pose[4]:.1f}, {interpolated_pose[5]:.1f}]")
                elif iter_idx == interpolation_steps - 1:
                    logger.info(f"\nDEBUG: Last interpolated command (t={t:.3f}):")
                    logger.info(f"   Pose: {interpolated_pose.tolist()}")
                    logger.info(f"   Orientation: [{interpolated_pose[3]:.1f}, {interpolated_pose[4]:.1f}, {interpolated_pose[5]:.1f}]")
                    logger.info(f"   Should match target: [{target_pose[3]:.1f}, {target_pose[4]:.1f}, {target_pose[5]:.1f}]")

                # Create action with current grasp state - same format as teleop
                action = np.concatenate([interpolated_pose, [self._current_grasp]])

                # Execute with same timing logic as teleop
                exec_timestamp = t_command_target - time.monotonic() + time.time()
                self.env.exec_actions(
                    actions=[action],
                    timestamps=[exec_timestamp],
                    stages=[stage_val],
                )

                # Wait for cycle end - same as teleop
                self._precise_wait(t_cycle_end)

            # Update current pose
            self._current_pose = target_pose.copy()
            return True

        except Exception as e:
            logger.error(f"Error during movement: {e}")
            return False

    def move_relative(
        self,
        delta_position: List[float],
        delta_orientation: Optional[List[float]] = None,
        duration: float = 1.0,
        stage_val: int = 0,
    ) -> bool:
        """Internal relative movement function using teleop script logic."""
        if delta_orientation is None:
            delta_orientation = [0.0, 0.0, 0.0]
        try:
            # Get current pose
            state = self.env.get_robot_state()
            current_pose = np.array(state["TCPPose"], dtype=np.float32)

            # Apply gains - same as teleop script
            dpos = (
                np.array(delta_position, dtype=np.float32)
                * self._xarm_config.position_gain
            )
            drot = (
                np.array(delta_orientation, dtype=np.float32)
                * self._xarm_config.orientation_gain
            )

            # Same rotation logic as teleop script
            curr_rot = st_Rotation.from_euler("xyz", current_pose[3:], degrees=True)
            delta_rot = st_Rotation.from_euler("xyz", drot, degrees=True)
            final_rot = delta_rot * curr_rot

            # Calculate target pose
            target_position = current_pose[:3] + dpos
            target_orientation = final_rot.as_euler("xyz", degrees=True)

            return self.move_to_pose(
                target_position=target_position.tolist(),
                target_orientation=target_orientation.tolist(),
                duration=duration,
                stage_val=stage_val,
            )

        except Exception as e:
            logger.error(f"Error during relative movement: {e}")
            return False

    def set_gripper(self, grasp_value: float, stage_val: int = 0, settle_time: float = 1.5) -> bool:
        """Set gripper opening.

        Sends the gripper command and holds position while the gripper
        physically closes/opens. The Robotiq gripper takes ~1-2s to fully
        close, so we keep sending hold-position commands with the new grasp
        value for settle_time seconds.

        Args:
            grasp_value: 0.0 = open, 1.0 = closed
            stage_val: Stage value for actions
            settle_time: Time in seconds to hold position while gripper moves (default 1.5s)
        """
        try:
            # Update grasp state
            self._current_grasp = float(grasp_value)

            # Get current pose
            state = self.env.get_robot_state()
            current_pose = np.array(state["TCPPose"], dtype=np.float32)

            action = np.concatenate([current_pose, [self._current_grasp]])

            # Send hold-position + gripper command for the full settle duration.
            # This ensures the gripper has time to physically close/open while
            # the robot stays in place.
            steps = int(settle_time * self._frequency)
            t_start = time.monotonic()

            for iter_idx in range(steps):
                t_cycle_end = t_start + (iter_idx + 1) * self._dt
                t_command_target = t_cycle_end + self._dt

                exec_timestamp = t_command_target - time.monotonic() + time.time()
                self.env.exec_actions(
                    actions=[action],
                    timestamps=[exec_timestamp],
                    stages=[stage_val],
                )

                self._precise_wait(t_cycle_end)

            logger.info(f"Gripper {'closed' if grasp_value >= 0.5 else 'opened'} (value={self._current_grasp}, held {settle_time:.1f}s)")
            return True

        except Exception as e:
            logger.error(f"Error setting gripper: {e}")
            return False

    def hold_position(self, duration: float, stage_val: int = 0):
        """Hold current position for specified duration.

        Args:
            duration: Duration to hold position in seconds
            stage_val: Stage value for actions
        """
        steps = int(duration * self._frequency)
        t_start = time.monotonic()

        # Get current pose
        current_pose = self.get_robot_pose()

        for iter_idx in range(steps):
            # Same timing logic as teleop script
            t_cycle_end = t_start + (iter_idx + 1) * self._dt
            t_command_target = t_cycle_end + self._dt

            # Pump obs
            obs = self.env.get_obs()

            # Send current pose - same as teleop when no significant movement
            action = np.concatenate([current_pose, [self._current_grasp]])
            exec_timestamp = t_command_target - time.monotonic() + time.time()

            self.env.exec_actions(
                actions=[action],
                timestamps=[exec_timestamp],
                stages=[stage_val],
            )

            self._precise_wait(t_cycle_end)

    def align_gripper_yaw(self, target_yaw_deg: float, duration: float = 1.0, stage_val: int = 0) -> bool:
        """Rotate only the gripper yaw (wrist / joint 7) to target_yaw_deg.

        Reads the current TCP pose, keeps position and roll/pitch unchanged,
        and moves only the yaw component. Because position and the other two
        orientation axes stay identical, the xArm IK resolves this as a pure
        joint-7 rotation.

        Args:
            target_yaw_deg: Desired TCP yaw in degrees.
            duration: Time to complete the rotation.
            stage_val: Stage value for action recording.

        Returns:
            True if the motion succeeded.
        """
        current_pose = self.get_robot_pose()  # [x, y, z, roll, pitch, yaw]
        target_orientation = [current_pose[3], current_pose[4], target_yaw_deg]
        logger.info(
            f"align_gripper_yaw: rotating yaw from {current_pose[5]:.1f} to "
            f"{target_yaw_deg:.1f} deg (delta {target_yaw_deg - current_pose[5]:.1f} deg)"
        )
        return self.move_to_pose(
            target_position=current_pose[:3].tolist(),
            target_orientation=target_orientation,
            duration=duration,
            stage_val=stage_val,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _precise_wait(self, target_time):
        """Precise wait function - same as teleop script."""
        try:
            # Try to use the precise_wait from teleop script if available
            from ril_env.precise_sleep import precise_wait

            precise_wait(target_time)
        except ImportError:
            # Fallback to regular sleep
            wait_time = target_time - time.monotonic()
            if wait_time > 0:
                time.sleep(wait_time)
