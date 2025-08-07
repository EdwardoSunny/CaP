import time
import numpy as np
import scipy.spatial.transform as st
import logging
from typing import List, Optional, Union, Tuple
import shapely
from cap.lmp.lmp import LMP, LMPFGen
from ril_env.precise_sleep import precise_wait
from ril_env.xarm_controller import XArmConfig, XArm
from ril_env.real_env import RealEnv

logger = logging.getLogger(__name__)

class LMPWrapper:
    def __init__(self, env, xarm_config, frequency=30, command_latency=0.01):
        """
        Initialize robot primitives using teleop script's exact components.
        
        Args:
            env: RealEnv instance from teleop script
            xarm_config: XArmConfig instance from teleop script
            frequency: Control frequency in Hz
            command_latency: Command latency in seconds
        """
        self.env = env
        self._xarm_config = xarm_config
        self._frequency = frequency
        self._command_latency = command_latency
        self._dt = 1.0 / frequency
        self._current_grasp = 0.0

        # Get initial robot state - same as teleop script
        state = self.env.get_robot_state()

        self._current_pose = np.array(state["TCPPose"], dtype=np.float32)
        
        # LMP-specific additions
        self._setup_lmp_environment()
        
        logger.info(f"Robot Primitives initialized. Current pose: {self._current_pose}")
    
    def _setup_lmp_environment(self):
        """Setup LMP-specific environment variables and object tracking."""
        # Mock object tracking (replace with actual computer vision)
        self.known_objects = []
        
        # Workspace bounds for your robot (adjust these to match your setup)
        self.workspace_bounds = {
            'x_min': -0.3, 'x_max': 0.3,
            'y_min': -0.8, 'y_max': -0.2,
            'z_table': 0.0
        }
        
        # Predefined positions for LMP commands
        self.corner_positions = {
            'top left corner': (-0.25, -0.25, 0),
            'top right corner': (0.25, -0.25, 0),
            'bottom left corner': (-0.25, -0.75, 0),
            'bottom right corner': (0.25, -0.75, 0),
        }
        
        self.side_positions = {
            'top side': (0, -0.25, 0),
            'bottom side': (0, -0.75, 0),
            'left side': (-0.25, -0.5, 0),
            'right side': (0.25, -0.5, 0),
        }
        
        # Color definitions for object recognition
        self.colors = {
            'red': (1.0, 0.0, 0.0, 1.0),
            'green': (0.0, 1.0, 0.0, 1.0),
            'blue': (0.0, 0.0, 1.0, 1.0),
            'yellow': (1.0, 1.0, 0.0, 1.0),
            'orange': (1.0, 0.5, 0.0, 1.0),
            'purple': (0.5, 0.0, 0.5, 1.0),
            'pink': (1.0, 0.75, 0.8, 1.0),
            'cyan': (0.0, 1.0, 1.0, 1.0),
            'brown': (0.6, 0.3, 0.1, 1.0),
            'gray': (0.5, 0.5, 0.5, 1.0),
        }
    
    def get_robot_pos(self):
        """Return robot end-effector xyz position in robot base frame."""
        state = self.env.get_robot_state()
        current_pose = np.array(state["TCPPose"], dtype=np.float32)
        return current_pose[:3]
    
    def get_robot_pose(self):
        """Return full robot pose [x, y, z, roll, pitch, yaw]."""
        state = self.env.get_robot_state()
        return np.array(state["TCPPose"], dtype=np.float32)
    
    def get_robot_xy(self):
        """Return robot end-effector xy position in robot base frame."""
        return self.get_robot_pos()[:2]
    
    def goto_pos(self, position_xyz, duration=3.0, stage_val=0):
        """
        Move the robot end-effector to the desired xyz position.
        
        Args:
            position_xyz: Target [x, y, z] position
            duration: Time to complete movement
            stage_val: Stage value for action
        """
        current_pose = self.get_robot_pose()
        target_position = np.array(position_xyz, dtype=np.float32)
        target_orientation = current_pose[3:].tolist()  # Keep current orientation
        
        return self._move_to_pose(
            target_position=target_position.tolist(),
            target_orientation=target_orientation,
            duration=duration,
            stage_val=stage_val
        )
    
    def goto_xy(self, position_xy, duration=2.0, stage_val=0):
        """
        Move robot end-effector to desired xy position while maintaining same z.
        
        Args:
            position_xy: Target [x, y] position
            duration: Time to complete movement
            stage_val: Stage value for action
        """
        current_pos = self.get_robot_pos()
        target_xyz = np.concatenate([position_xy, [current_pos[2]]])
        return self.goto_pos(target_xyz, duration, stage_val)
    
    def goto_pose(self, position_xyz, orientation_rpy, duration=3.0, stage_val=0):
        """
        Move robot to specific pose (position + orientation).
        
        Args:
            position_xyz: Target [x, y, z] position
            orientation_rpy: Target [roll, pitch, yaw] orientation in degrees
            duration: Time to complete movement
            stage_val: Stage value for action
        """
        return self._move_to_pose(
            target_position=position_xyz,
            target_orientation=orientation_rpy,
            duration=duration,
            stage_val=stage_val
        )
    
    def move_relative(self, delta_xyz, delta_rpy=None, duration=2.0, stage_val=0):
        """
        Move robot relative to current pose.
        
        Args:
            delta_xyz: Relative [dx, dy, dz] movement
            delta_rpy: Relative [droll, dpitch, dyaw] rotation (optional)
            duration: Time to complete movement
            stage_val: Stage value for action
        """
        if delta_rpy is None:
            delta_rpy = [0.0, 0.0, 0.0]
            
        return self._move_relative(
            delta_position=delta_xyz,
            delta_orientation=delta_rpy,
            duration=duration,
            stage_val=stage_val
        )
    
    # def move_up(self, distance=0.05, duration=1.0, stage_val=0):
    def move_up(self, distance=2.00, duration=1.0, stage_val=0):
        """Move robot up by specified distance."""
        return self.move_relative([0, 0, distance], duration=duration, stage_val=stage_val)
    
    def move_down(self, distance=2.00, duration=1.0, stage_val=0):
        """Move robot down by specified distance."""
        return self.move_relative([0, 0, -distance], duration=duration, stage_val=stage_val)
    
    def open_gripper(self, stage_val=0):
        """Open the gripper."""
        return self._set_gripper(0.0, stage_val)
    
    def close_gripper(self, stage_val=0):
        """Close the gripper."""
        return self._set_gripper(1.0, stage_val)
    
    def set_gripper(self, grasp_value, stage_val=0):
        """Set gripper to specific value."""
        return self._set_gripper(grasp_value, stage_val)
    
    def pick_place(self, pick_pos, place_pos, pick_height=0.15, place_height=0.15, 
                   approach_height=0.25, stage_val=0):
        """
        Execute pick and place operation.
        
        Args:
            pick_pos: [x, y] or [x, y, z] pick position
            place_pos: [x, y] or [x, y, z] place position
            pick_height: Z height for picking (if pick_pos is 2D)
            place_height: Z height for placing (if place_pos is 2D)
            approach_height: Z height for approach movements
            stage_val: Stage value for actions
        """
        try:
            # Convert to 3D positions if needed
            if len(pick_pos) == 2:
                pick_pos_xyz = np.array([pick_pos[0], pick_pos[1], pick_height])
            else:
                pick_pos_xyz = np.array(pick_pos)
                
            if len(place_pos) == 2:
                place_pos_xyz = np.array([place_pos[0], place_pos[1], place_height])
            else:
                place_pos_xyz = np.array(place_pos)
            
            approach_pick = np.array([pick_pos_xyz[0], pick_pos_xyz[1], approach_height])
            approach_place = np.array([place_pos_xyz[0], place_pos_xyz[1], approach_height])
            
            logger.info(f"Pick and place: {pick_pos_xyz} -> {place_pos_xyz}")
            
            # Move to approach pick position
            self.goto_pos(approach_pick, duration=3.0, stage_val=stage_val)
            
            # Open gripper
            self.open_gripper(stage_val)
            time.sleep(0.5)
            
            # Move down to pick
            self.goto_pos(pick_pos_xyz, duration=2.0, stage_val=stage_val)
            
            # Close gripper
            self.close_gripper(stage_val)
            time.sleep(1.0)
            
            # Move up to approach height
            self.goto_pos(approach_pick, duration=2.0, stage_val=stage_val)
            
            # Move to approach place position
            self.goto_pos(approach_place, duration=3.0, stage_val=stage_val)
            
            # Move down to place
            self.goto_pos(place_pos_xyz, duration=2.0, stage_val=stage_val)
            
            # Open gripper
            self.open_gripper(stage_val)
            time.sleep(0.5)
            
            # Move up
            self.goto_pos(approach_place, duration=2.0, stage_val=stage_val)
            
            logger.info("Pick and place completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Pick and place failed: {e}")
            return False
    
    def follow_traj(self, trajectory, duration_per_point=1.0, stage_val=0):
        """
        Follow a trajectory of positions.
        
        Args:
            trajectory: List of [x, y] or [x, y, z] positions
            duration_per_point: Time to spend moving to each point
            stage_val: Stage value for actions
        """
        for i, pos in enumerate(trajectory):
            logger.info(f"Following trajectory point {i+1}/{len(trajectory)}: {pos}")
            if len(pos) == 2:
                self.goto_xy(pos, duration=duration_per_point, stage_val=stage_val)
            else:
                self.goto_pos(pos, duration=duration_per_point, stage_val=stage_val)
    
    def hold_position(self, duration, stage_val=0):
        """
        Hold current position for specified duration.
        
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
    
    def wait(self, duration):
        """Simple wait function."""
        time.sleep(duration)
    
    def is_gripper_open(self):
        """Check if gripper is open."""
        return self._current_grasp < 0.5
    
    def is_gripper_closed(self):
        """Check if gripper is closed."""
        return self._current_grasp >= 0.5
    
    def get_gripper_state(self):
        """Get current gripper state."""
        return self._current_grasp
    
    # ========== LMP-Required Functions ==========
    
    def get_obj_names(self):
        """Return list of known object names."""
        return self.known_objects.copy()
    
    def is_obj_visible(self, obj_name):
        """Check if object is visible/known."""
        return obj_name in self.known_objects
    
    def get_obj_pos(self, obj_name):
        """
        Get object position. You'll need to implement actual object detection.
        For now, returns mock positions or corner/side positions.
        
        Args:
            obj_name: Name of object or position reference
            
        Returns:
            np.array: [x, y, z] position
        """
        obj_name = obj_name.replace('the', '').replace('_', ' ').strip()
        
        if obj_name in self.corner_positions:
            return list(self.corner_positions[obj_name])
        elif obj_name in self.side_positions:
            return list(self.side_positions[obj_name])
        else:
            # TODO: Replace with actual object detection/tracking
            logger.warning(f"Mock position returned for {obj_name}")
            return [0.0, -0.5, 0.0]
    
    def get_bbox(self, obj_name):
        """
        Get object bounding box.
        
        Args:
            obj_name: Name of object
            
        Returns:
            tuple: (min_x, min_y, max_x, max_y) bounding box
        """
        pos = self.get_obj_pos(obj_name)
        # Return simple bounding box around position (adjust size as needed)
        size = 0.02  # 2cm box
        return (pos[0]-size, pos[1]-size, pos[0]+size, pos[1]+size)
    
    def get_color(self, obj_name):
        """
        Extract color from object name.
        
        Args:
            obj_name: Name of object
            
        Returns:
            tuple: (r, g, b, a) color values
        """
        for color_name, rgb in self.colors.items():
            if color_name in obj_name.lower():
                return rgb
        return (0.5, 0.5, 0.5, 1.0)  # Default gray
    
    def denormalize_xy(self, pos_normalized):
        """
        Convert normalized coordinates [0,1] to workspace coordinates.
        
        Args:
            pos_normalized: [x, y] in range [0, 1]
            
        Returns:
            np.array: [x, y] in workspace coordinates
        """
        x_range = self.workspace_bounds['x_max'] - self.workspace_bounds['x_min']
        y_range = self.workspace_bounds['y_max'] - self.workspace_bounds['y_min']
        
        x = pos_normalized[0] * x_range + self.workspace_bounds['x_min']
        y = pos_normalized[1] * y_range + self.workspace_bounds['y_min']
        
        return np.array([x, y])
    
    def get_corner_name(self, pos):
        """
        Get the name of the closest corner to a position.
        
        Args:
            pos: [x, y] or [x, y, z] position
            
        Returns:
            str: Name of closest corner
        """
        pos_2d = np.array(pos[:2])
        corner_positions_2d = np.array([[p[0], p[1]] for p in self.corner_positions.values()])
        distances = np.linalg.norm(corner_positions_2d - pos_2d, axis=1)
        closest_idx = np.argmin(distances)
        corner_names = list(self.corner_positions.keys())
        return corner_names[closest_idx]
    
    def get_side_name(self, pos):
        """
        Get the name of the closest side to a position.
        
        Args:
            pos: [x, y] or [x, y, z] position
            
        Returns:
            str: Name of closest side
        """
        pos_2d = np.array(pos[:2])
        side_positions_2d = np.array([[p[0], p[1]] for p in self.side_positions.values()])
        distances = np.linalg.norm(side_positions_2d - pos_2d, axis=1)
        closest_idx = np.argmin(distances)
        side_names = list(self.side_positions.keys())
        return side_names[closest_idx]
    
    def put_first_on_second(self, obj1, obj2):
        """
        Put first object on second object or position.
        This is the core LMP function that maps to your pick_place.
        
        Args:
            obj1: Object name or position to pick
            obj2: Object name or position to place on
            
        Returns:
            bool: Success status
        """
        try:
            # Get pick position
            if isinstance(obj1, str):
                pick_pos = self.get_obj_pos(obj1)[:2]  # Get x,y only
            else:
                pick_pos = np.array(obj1)[:2]
            
            # Get place position  
            if isinstance(obj2, str):
                place_pos = self.get_obj_pos(obj2)[:2]  # Get x,y only
            else:
                place_pos = np.array(obj2)[:2]
            
            logger.info(f"Executing put_first_on_second: {obj1} -> {obj2}")
            logger.info(f"Pick position: {pick_pos}, Place position: {place_pos}")
            
            # Execute pick and place
            return self.pick_place(pick_pos, place_pos)
            
        except Exception as e:
            logger.error(f"Error in put_first_on_second: {e}")
            return False
    
    # ========== Object Management Functions ==========
    
    def add_object(self, obj_name, position=None):
        """
        Add an object to the known objects list.
        
        Args:
            obj_name: Name of object to add
            position: Optional position (for future use with object tracking)
        """
        if obj_name not in self.known_objects:
            self.known_objects.append(obj_name)
            logger.info(f"Added object: {obj_name}")
    
    def remove_object(self, obj_name):
        """
        Remove an object from the known objects list.
        
        Args:
            obj_name: Name of object to remove
        """
        if obj_name in self.known_objects:
            self.known_objects.remove(obj_name)
            logger.info(f"Removed object: {obj_name}")
    
    def update_object_list(self, object_list):
        """
        Update the list of known objects.
        
        Args:
            object_list: List of object names
        """
        self.known_objects = object_list.copy()
        logger.info(f"Updated object list: {self.known_objects}")
    
    def clear_objects(self):
        """Clear all known objects."""
        self.known_objects.clear()
        logger.info("Cleared all objects")
    
    # ========== Utility Functions ==========
    
    def get_workspace_bounds(self):
        """Get workspace boundaries."""
        return self.workspace_bounds.copy()
    
    def set_workspace_bounds(self, bounds):
        """
        Set workspace boundaries.
        
        Args:
            bounds: Dictionary with keys 'x_min', 'x_max', 'y_min', 'y_max', 'z_table'
        """
        self.workspace_bounds.update(bounds)
        logger.info(f"Updated workspace bounds: {self.workspace_bounds}")
    
    def get_corner_positions(self):
        """Get all corner positions."""
        return self.corner_positions.copy()
    
    def get_side_positions(self):
        """Get all side positions."""
        return self.side_positions.copy()
    
    # ========== Private Methods (unchanged) ==========
    
    def _move_to_pose(self, target_position, target_orientation, duration=3.0, stage_val=0):
        """Internal move to pose function using teleop script logic."""

        print("MOVING TO", target_position)
        try:
            target_pose = np.array(target_position + target_orientation, dtype=np.float32)
            
            # Get current pose - same as teleop script
            state = self.env.get_robot_state()
            start_pose = np.array(state["TCPPose"], dtype=np.float32)
            
            # Calculate interpolation steps
            interpolation_steps = int(duration * self._frequency)
            
            logger.debug(f"Moving from {start_pose} to {target_pose} over {duration}s ({interpolation_steps} steps)")
            
            t_start = time.monotonic()
            
            for iter_idx in range(interpolation_steps):
                # Same timing logic as teleop script
                t_cycle_end = t_start + (iter_idx + 1) * self._dt
                t_command_target = t_cycle_end + self._dt
                
                # Pump obs - same as teleop script
                obs = self.env.get_obs()
                
                # Linear interpolation between start and target
                t = (iter_idx + 1) / interpolation_steps
                interpolated_pose = start_pose + t * (target_pose - start_pose)
                
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
    
    def _move_relative(self, delta_position, delta_orientation, duration=1.0, stage_val=0):
        """Internal relative movement function using teleop script logic."""
        try:
            # Get current pose
            state = self.env.get_robot_state()
            current_pose = np.array(state["TCPPose"], dtype=np.float32)
            
            # Apply gains - same as teleop script
            dpos = np.array(delta_position, dtype=np.float32) * self._xarm_config.position_gain
            drot = np.array(delta_orientation, dtype=np.float32) * self._xarm_config.orientation_gain
            
            # Same rotation logic as teleop script
            curr_rot = st.Rotation.from_euler("xyz", current_pose[3:], degrees=True)
            delta_rot = st.Rotation.from_euler("xyz", drot, degrees=True)
            final_rot = delta_rot * curr_rot
            
            # Calculate target pose
            target_position = current_pose[:3] + dpos
            target_orientation = final_rot.as_euler("xyz", degrees=True)
            
            return self._move_to_pose(
                target_position=target_position.tolist(),
                target_orientation=target_orientation.tolist(),
                duration=duration,
                stage_val=stage_val
            )
            
        except Exception as e:
            logger.error(f"Error during relative movement: {e}")
            return False
    
    def _set_gripper(self, grasp_value, stage_val=0):
        """Internal gripper control function using teleop script logic."""
        try:
            # Update grasp state
            self._current_grasp = float(grasp_value)
            
            # Get current pose
            state = self.env.get_robot_state()
            current_pose = np.array(state["TCPPose"], dtype=np.float32)
            
            # Send command with current pose - same as teleop script
            action = np.concatenate([current_pose, [self._current_grasp]])
            
            # Use same timing logic as teleop
            t_command_target = time.monotonic() + self._dt
            exec_timestamp = t_command_target - time.monotonic() + time.time()
            
            self.env.exec_actions(
                actions=[action],
                timestamps=[exec_timestamp],
                stages=[stage_val],
            )
            
            logger.debug(f"Gripper set to: {self._current_grasp}")
            return True
            
        except Exception as e:
            logger.error(f"Error setting gripper: {e}")
            return False
    
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

def setup_LMP(config, env, xarm_config):
    """
    Setup LMP system for real robot environment using enhanced LMPWrapper.
    
    Args:
        config: Configuration dictionary containing LMP configs
        
    Returns:
        tuple: (lmp_tabletop_ui, LMP_env) - The main LMP interface and environment wrapper
    """

    LMP_env = LMPWrapper(env, xarm_config)
    
    # Creating APIs that the LMPs can interact with
    fixed_vars = {
        'np': np,
        'time': __import__('time'),
    }
    
    # Add shapely geometry functions
    fixed_vars.update({
        name: getattr(shapely.geometry, name)
        for name in shapely.geometry.__all__
    })
    fixed_vars.update({
        name: getattr(shapely.affinity, name)
        for name in shapely.affinity.__all__
    })
    
    # Add LMP environment functions (all now available in LMPWrapper)
    variable_vars = {
        # Core robot functions
        'get_robot_pos': LMP_env.get_robot_pos,
        'get_robot_xy': LMP_env.get_robot_xy,
        'goto_pos': LMP_env.goto_pos,
        'goto_xy': LMP_env.goto_xy,
        'move_relative': LMP_env.move_relative,
        'move_up': LMP_env.move_up,
        'move_down': LMP_env.move_down,
        'pick_place': LMP_env.pick_place,
        'follow_traj': LMP_env.follow_traj,
        'wait': LMP_env.wait,
        
        # Gripper functions
        'open_gripper': LMP_env.open_gripper,
        'close_gripper': LMP_env.close_gripper,
        'set_gripper': LMP_env.set_gripper,
        'is_gripper_open': LMP_env.is_gripper_open,
        'is_gripper_closed': LMP_env.is_gripper_closed,
        
        # LMP-required functions (now implemented in LMPWrapper)
        'get_obj_pos': LMP_env.get_obj_pos,
        'get_obj_names': LMP_env.get_obj_names,
        'is_obj_visible': LMP_env.is_obj_visible,
        'put_first_on_second': LMP_env.put_first_on_second,
        'denormalize_xy': LMP_env.denormalize_xy,
        'get_corner_name': LMP_env.get_corner_name,
        'get_side_name': LMP_env.get_side_name,
        'get_bbox': LMP_env.get_bbox,
        'get_color': LMP_env.get_color,
        
        # Utility functions
        'say': lambda msg: print(f'robot says: {msg}'),
    }

    # Creating the function-generating LMP
    lmp_fgen = LMPFGen(
        config["lmp_config"]['lmps']['fgen'], 
        fixed_vars, 
        variable_vars
    )

    # Creating other low-level LMPs
    variable_vars.update({
        k: LMP(k, config["lmp_config"]['lmps'][k], lmp_fgen, fixed_vars, variable_vars)
        for k in ['parse_obj_name', 'parse_position', 'parse_question', 'transform_shape_pts']
    })

    # Creating the LMP that deals with high-level language commands

    print(fixed_vars)
    print("========================================")
    print(variable_vars)

    lmp_tabletop_ui = LMP(
        'tabletop_ui', 
        config["lmp_config"]['lmps']['tabletop_ui'], 
        lmp_fgen, 
        fixed_vars, 
        variable_vars
    )

    return lmp_tabletop_ui, LMP_env
