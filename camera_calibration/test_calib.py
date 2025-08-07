import cv2
import numpy as np

from multicam import XarmEnv

from rs_streamer import RealsenseStreamer
from calib_utils.linalg_utils import transform

# from pynput import keyboard


GRIPPER_SPEED, GRIPPER_FORCE, GRIPPER_MAX_WIDTH, GRIPPER_TOLERANCE = (
    0.1,
    40,
    0.08570,
    0.01,
)


class PixelSelector:
    def __init__(self):
        pass

    def load_image(self, img, recrop=False):
        self.img = img
        if recrop:
            cropped_img = self.crop_at_point(img, 700, 300, width=400, height=300)
            self.img = cv2.resize(cropped_img, (640, 480))

    def crop_at_point(self, img, x, y, width=640, height=480):
        img = img[y : y + height, x : x + width]
        return img

    def mouse_callback(self, event, x, y, flags, param):
        cv2.imshow("pixel_selector", self.img)
        if event == cv2.EVENT_LBUTTONDBLCLK:
            self.clicks.append([x, y])
            cv2.circle(self.img, (x, y), 3, (255, 255, 0), -1)

    def run(self, img):
        self.load_image(img)
        self.clicks = []
        cv2.namedWindow("pixel_selector")
        cv2.setMouseCallback("pixel_selector", self.mouse_callback)
        while True:
            k = cv2.waitKey(20) & 0xFF
            if k == 27:
                break
        return self.clicks


def goto(robot, realsense_streamer, pixel_selector, TCR, refine=False):
    # right
    # print(TCR)
    # TCR[0,3] += 25
    # TCR[1,3] += 25

    for i in range(5):
        _, rgb_image, depth_frame, depth_img = realsense_streamer.capture_rgbd()

    pixels = pixel_selector.run(rgb_image)
    waypoint_cam = 1000.0 * np.array(
        realsense_streamer.deproject(pixels[0], depth_frame)
    )
    waypoint_rob = transform(np.array(waypoint_cam).reshape(1, 3), TCR)

    # Get waypoints in robot frame
    ee_pos_desired = np.array(waypoint_rob)[0]
    print(ee_pos_desired)
    lift_pos = ee_pos_desired + np.array([0, 0, 50])

    # Put robot in canonical orientation
    robot.go_home()
    ee_pos, ee_euler = robot.pose_ee()

    state_log = robot.move_to_ee_pose(
        ee_pos,
        ee_euler,
    )
    _, ee_euler = robot.pose_ee()
    # # #

    state_log = robot.move_to_ee_pose(
        lift_pos,
        ee_euler,
    )

    state_log = robot.move_to_ee_pose(
        ee_pos_desired,
        ee_euler,
    )

    state_log = robot.move_to_ee_pose(
        lift_pos,
        ee_euler,
    )

    # if refine:
    #    CALIB_OFFSET = teleop(robot)
    #    TCR[:,3] += CALIB_OFFSET
    # else:
    #    state_log = robot.set_ee_pose(
    #        position=ee_pos_desired, orientation=ee_quat,
    #    )

    #    state_log = robot.set_ee_pose(
    #        position=lift_pos, orientation=ee_quat,
    #    )

    return TCR


if __name__ == "__main__":
    serial_no = "317422075456"

    # Get camera, load transforms, load robot
    realsense_streamer = RealsenseStreamer(serial_no)
    transforms = np.load("calib/transforms.npy", allow_pickle=True).item()
    TCR = transforms[serial_no]["tcr"]

    robot = XarmEnv()

    # Initialize pixel selector
    pixel_selector = PixelSelector()

    # Test going to a waypoint
    TCR = goto(robot, realsense_streamer, pixel_selector, TCR, refine=True)
    transforms[serial_no]["tcr"] = TCR

    res = input("save?")
    if res == "y" or res == "Y":
        np.save("calib/transforms.npy", transforms)
        print("SAVED")
