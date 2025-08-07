import sys
import pprint
import numpy as np
import cv2
from rs_streamer import RealsenseStreamer, MarkSearch
import os

from scipy.spatial.transform import Rotation as R

from calib_utils.solver import Solver

sys.path.append(
    os.path.join(os.path.dirname(__file__), "/home/u-ril/URIL/xArm-Python-SDK")
)
from xarm import XArmAPI

from dataclasses import dataclass
from vision_utils.pc_utils import (
    deproject,
    project,
    merge_pcls,
    denoise,
)

# from vision_utils.pc_utils import *

GRIPPER_SPEED, GRIPPER_FORCE, GRIPPER_MAX_WIDTH, GRIPPER_TOLERANCE = (
    0.1,
    40,
    0.08570,
    0.01,
)

ip = "192.168.1.223"


@dataclass
class XArmConfig:
    """
    Configuration class for some (not all!) xArm7/control parameters. The important ones are here.
    You can or should change most of these to your liking, potentially with the exception of tcp_maxacc

    :config_param tcp_maxacc: TCP (Tool Center Point, i.e., end effector) maximum acceleration
    :config_param position_gain: Increasing this value makes the position gain increase
    :config_param orientation_gain: Increasing this value makes the orientation gain increase
    :config_param alpha: This is a pseudo-smoothing factor
    :config_param control_loop_rate: Self-descriptive
    :config verbose: Helpful debugging / checking print steps
    """

    tcp_maxacc: int = 5000
    position_gain: float = 10.0
    orientation_gain: float = 10.0
    alpha: float = 0.5
    control_loop_rate: int = 50
    verbose: bool = True


class XarmEnv:
    def __init__(self):

        xarm_cfg = XArmConfig()

        self.arm = XArmAPI(ip)
        self.arm.connect()

        # This may be unsafe
        self.arm.clean_error()
        self.arm.clean_warn()

        # Robot arm below:
        ret = self.arm.motion_enable(enable=True)
        if ret != 0:
            print(f"Error in motion_enable: {ret}")
            sys.exit(1)

        self.arm.set_tcp_maxacc(xarm_cfg.tcp_maxacc)
        self.arm.set_mode(0)
        self.arm.set_state(state=0)

        ret = self.arm.set_mode(1)  # This sets the mode to serve motion mode
        if ret != 0:
            print(f"Error in set_mode: {ret}")
            sys.exit(1)

        ret = self.arm.set_state(0)  # This sets the state to sport (ready) state
        if ret != 0:
            print(f"Error in set_state: {ret}")
            sys.exit(1)

        ret, state = self.arm.get_state()
        if ret != 0:
            print(f"Error getting robot state: {ret}")
            sys.exit(1)

        if state != 0:
            print(f"Robot is not ready to move. Current state: {state}")
            sys.exit(1)
        else:
            print(f"Robot is ready to move. Current state: {state}")

        self.go_home()

    def go_home(self):
        self.arm.set_mode(0)
        self.arm.set_state(state=0)
        self.arm.set_servo_angle(angle=[0, 0, 0, 70, 0, 70, 0], speed=50, wait=True)

    def pose_ee(self):
        _, initial_pose = self.arm.get_position(is_radian=False)
        current_position = np.array(initial_pose[:3])
        current_orientation = np.array(initial_pose[3:])
        # print(initial_pose)
        return current_position, current_orientation

    def move_to_ee_pose(self, position, orietation):
        # ret = self.arm.set_servo_cartesian(np.concatenate((position, orietation)), is_radian=False, speed=1, wait=True)
        ret = self.arm.set_position(
            x=position[0],
            y=position[1],
            z=position[2],
            roll=orietation[0],
            pitch=orietation[1],
            yaw=orietation[2],
            speed=200,
            is_radian=False,
            wait=True,
        )
        # print(f"Return value from set_servo_cartesian: {ret}")
        return ret


class MultiCam:
    def __init__(self, serial_nos):
        self.cameras = []
        for serial_no in serial_nos:
            self.cameras.append(RealsenseStreamer(serial_no))

        self.transforms = None

        if os.path.exists("calib/transforms.npy"):
            self.transforms = np.load("calib/transforms.npy", allow_pickle=True).item()

        if os.path.exists("calib/icp_tf.npy"):
            self.icp_tf = np.load("calib/icp_tf.npy", allow_pickle=True).item()
            # self.icp_tf = None
        else:
            self.icp_tf = None

    def robot_fingertip_pos_to_ee(self, fingertip_pos, ee_quat):
        HOME_QUAT = np.array([0.9367, 0.3474, -0.0088, -0.0433])
        FINGERTIP_OFFSET = np.array([0, 0, -0.095])
        home_euler = R.from_quat(HOME_QUAT).as_euler("zyx", degrees=True)

        ee_euler = R.from_quat(ee_quat).as_euler("zyx", degrees=True)

        offset_euler = ee_euler - home_euler

        fingertip_offset_euler = offset_euler * [1, -1, 1]
        fingertip_transf = R.from_euler("zyx", fingertip_offset_euler, degrees=True)
        fingertip_offset = fingertip_transf.as_matrix() @ FINGERTIP_OFFSET
        # fingertip_offset[2] -= 0.9*FINGERTIP_OFFSET[2]
        fingertip_offset[2] -= FINGERTIP_OFFSET[2]

        ee_pos = fingertip_pos - fingertip_offset
        return ee_pos

    def robot_ee_to_fingertip_pos(self, ee_pos, ee_quat):
        HOME_QUAT = np.array([0.9367, 0.3474, -0.0088, -0.0433])
        FINGERTIP_OFFSET = np.array([0, 0, -0.095])

        home_euler = R.from_quat(HOME_QUAT).as_euler("zyx", degrees=True)
        ee_euler = R.from_quat(ee_quat).as_euler("zyx", degrees=True)
        offset_euler = ee_euler - home_euler
        fingertip_offset_euler = offset_euler * [1, -1, 1]

        fingertip_transf = R.from_euler("zyx", fingertip_offset_euler, degrees=True)
        fingertip_offset = fingertip_transf.as_matrix() @ FINGERTIP_OFFSET
        # fingertip_offset[2] -= 0.9*FINGERTIP_OFFSET[2]
        fingertip_offset[2] -= FINGERTIP_OFFSET[2]

        fingertip_pos = np.array([ee_pos[0], ee_pos[1], ee_pos[2]]) + fingertip_offset
        return fingertip_pos

    def project_fingertip_pos(self, fingertip_pos):
        waypoints_proj = {cam.serial_no: None for cam in self.cameras}
        for cam in self.cameras:
            tcr = self.transforms[cam.serial_no]["tcr"]
            tf = np.linalg.inv(np.vstack((tcr, np.array([0, 0, 0, 1]))))[:3]
            pixel = project(fingertip_pos, cam.K, tf)
            waypoints_proj[cam.serial_no] = pixel
        return waypoints_proj

    def crop(
        self, pcd, min_bound=[0.2, -0.35, 0.10], max_bound=[0.9, 0.3, 0.5]
    ):  # what primitives were trained on
        # def crop(self, pcd, min_bound=[0.2,-0.35,0.09], max_bound=[0.9, 0.3, 0.5]):
        idxs = np.logical_and(
            np.logical_and(
                np.logical_and(pcd[:, 0] > min_bound[0], pcd[:, 0] < max_bound[0]),
                np.logical_and(pcd[:, 1] > min_bound[1], pcd[:, 1] < max_bound[1]),
            ),
            np.logical_and(pcd[:, 2] > min_bound[2], pcd[:, 2] < max_bound[2]),
        )
        return idxs

    def take_rgb(self, visualize=True):
        rgb_images = {cam.serial_no: None for cam in self.cameras}
        for idx, cam in enumerate(self.cameras):
            _, rgb_image, depth_frame, depth_img_vis = cam.capture_rgbd()
            rgb_images[cam.serial_no] = rgb_image
        return rgb_images

    def take_rgbd(self, visualize=True):
        rgb_images = {cam.serial_no: None for cam in self.cameras}
        depth_images = {cam.serial_no: None for cam in self.cameras}

        merged_points = []
        merged_colors = []

        icp_tfs = []
        cam_ids = []
        for idx, cam in enumerate(self.cameras):

            _, rgb_image, depth_frame, depth_img_vis = cam.capture_rgbd()

            rgb_images[cam.serial_no] = rgb_image

            depth_img = np.asanyarray(depth_frame.get_data())
            denoised_idxs = denoise(depth_img)

            depth_images[cam.serial_no] = depth_img

            tf = self.transforms[cam.serial_no]["tcr"]

            points_3d = deproject(depth_img, cam.K, tf)
            colors = (
                cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB).reshape(points_3d.shape)
                / 255.0
            )

            points_3d = points_3d[denoised_idxs]
            colors = colors[denoised_idxs]

            idxs = self.crop(points_3d)
            points_3d = points_3d[idxs]
            colors = colors[idxs]

            merged_points.append(points_3d)
            merged_colors.append(colors)

            if idx > 0:
                cam_ids.append(cam.serial_no)
                if self.icp_tf is not None:
                    icp_tfs.append(self.icp_tf[cam.serial_no])

        pcd_merged = merge_pcls(
            merged_points,
            merged_colors,
            tfs=icp_tfs,
            cam_ids=cam_ids,
            visualize=visualize,
        )
        return rgb_images, depth_images, pcd_merged

    def calibrate_cam(self, robot=None):
        if not os.path.exists("calib"):
            os.mkdir("calib")
            curr_calib = {}
        else:
            curr_calib = np.load("calib/transforms.npy", allow_pickle=True).item()

        self.marker_search = MarkSearch()

        self.solver = Solver()

        def gen_calib_waypoints(start_pos):
            waypoints = []
            for i in np.linspace(200, 400, 4):
                for j in np.linspace(-150, 150, 4):
                    for k in np.linspace(200, 500, 3):
                        waypoints.append(np.array([i, j, k]))
            return waypoints

        if robot is None:
            robot = XarmEnv()

        # Get ee pose
        ee_pos, ee_euler = robot.pose_ee()
        print("ee_euler", ee_euler)
        ee_euler = [-180, 0, 0]

        waypoints = gen_calib_waypoints(ee_pos)

        calib_eulers = []
        z_offsets = [0]
        for z_off in z_offsets:
            calib_euler = ee_euler + np.array([5, -5, z_off])
            calib_eulers.append(calib_euler)

        waypoints_rob = []
        waypoints_cam = {c.serial_no: [] for c in self.cameras}

        state_log = robot.move_to_ee_pose(ee_pos, calib_euler)

        itr = 0
        for waypoint in waypoints:
            print(itr, waypoint)
            itr += 1
            # print(waypoint)
            successful_waypoint = True

            intermed_waypoints = {}
            for idx, cam in enumerate(self.cameras):
                calib_euler = calib_eulers[idx]
                state_log = robot.move_to_ee_pose(
                    waypoint,
                    calib_euler,
                )

                _, rgb_image, depth_frame, depth_img = cam.capture_rgbd()
                (u, v), vis = self.marker_search.find_marker(rgb_image)
                if u is None:
                    successful_waypoint = False
                    break

                waypoint_cam = np.array(cam.deproject((u, v), depth_frame))
                print(waypoint_cam)
                waypoint_cam = 1000.0 * waypoint_cam
                print(waypoint_cam)
                intermed_waypoints[cam.serial_no] = waypoint_cam

            if successful_waypoint:
                waypoints_rob.append([waypoint[0], waypoint[1], waypoint[2]])
                for k in intermed_waypoints:
                    waypoints_cam[k].append(intermed_waypoints[k])

        pprint.pprint(waypoints_cam)
        pprint.pprint(waypoints_rob)

        transforms = {}

        waypoints_rob = np.array(waypoints_rob)

        for cam in self.cameras:
            waypoints_cam_curr = waypoints_cam[cam.serial_no]
            waypoints_cam_curr = np.array(waypoints_cam_curr)
            trc, tcr = self.solver.solve_transforms(waypoints_rob, waypoints_cam_curr)
            transforms[cam.serial_no] = {"trc": trc, "tcr": tcr}

        curr_calib.update(transforms)
        # np.save('calib/transforms.npy', transforms)
        np.save("calib/transforms.npy", curr_calib)


if __name__ == "__main__":
    # calibration
    multi_cam = MultiCam(["317422075456"])
    multi_cam.calibrate_cam()

    # Uncomment to take an image + merged point cloud
    # multi_cam = MultiCam(['317422075456'])
    # rgb_images, depth_images, pcd_merged = multi_cam.take_rgbd()
