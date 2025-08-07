import numpy as np
from calib_utils.linalg_utils import get_transform, transform_data, plot_pointclouds


class Solver:
    def __init__(self):
        pass

    def solve_transforms(self, robot_pts, cam_pts, visualize=False):
        if visualize:
            plot_pointclouds(
                [robot_pts, cam_pts]
            )  # plot robot and camera chessboard points (should be unaligned)
        TR_C = get_transform(
            "Camera", "Robot", robot_pts, cam_pts
        )  # solve for rigid transform
        TC_R = get_transform("Robot", "Camera", cam_pts, robot_pts)
        # np.save('calib/trc.npy', TR_C)
        # np.save('calib/tcr.npy', TC_R)
        cam_transformed = transform_data(
            "Camera", "Robot", cam_pts, TC_R, data_out=robot_pts
        )
        rob_transformed = transform_data(
            "Robot", "Camera", robot_pts, TR_C, data_out=cam_pts
        )
        if visualize:
            plot_pointclouds([robot_pts, cam_transformed])
            plot_pointclouds([cam_pts, rob_transformed])
        return TR_C, TC_R


if __name__ == "__main__":
    robot_pts = np.load("../calib/robot_pts.npy", allow_pickle=True)
    cam_pts = np.load("../calib/cam_pts.npy", allow_pickle=True).item()
    cam_pts = np.array(cam_pts[list(cam_pts.keys())[0]])
    print(robot_pts.shape, cam_pts.shape)
    solver = Solver()
    solver.solve_transforms(robot_pts, cam_pts)
