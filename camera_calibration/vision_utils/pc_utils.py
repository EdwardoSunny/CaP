import numpy as np
import os
import cv2
import open3d as o3d
import copy
from sklearn.neighbors import NearestNeighbors

# from simpleicp import PointCloud, SimpleICP


def deproject(depth_image, K, tf=np.eye(4), base_units=-3):
    depth_image = depth_image * (10**base_units)  # convert mm to m (TODO)

    h, w = depth_image.shape
    row_indices = np.arange(h)
    col_indices = np.arange(w)
    pixel_grid = np.meshgrid(col_indices, row_indices)
    pixels = np.c_[pixel_grid[0].flatten(), pixel_grid[1].flatten()].T

    pixels_homog = np.r_[pixels, np.ones([1, pixels.shape[1]])]
    depth_arr = np.tile(depth_image.flatten(), [3, 1])
    points_3d = depth_arr * np.linalg.inv(K).dot(pixels_homog)

    points_3d_transf = np.vstack((points_3d, np.ones([1, points_3d.shape[1]])))
    points_3d_transf = ((tf.dot(points_3d_transf)).T)[:, 0:3]

    return points_3d_transf


def deproject_pixels(pixels, depth_image, K, tf=np.eye(4), base_units=-3):
    h, w = depth_image.shape
    all_points = deproject(depth_image, K, tf, base_units)
    all_points = np.reshape(all_points, (h, w, 3))
    mask = np.zeros((h, w))
    mask[pixels[:, 1], pixels[:, 0]] = 1
    mask = mask.astype(bool)
    return all_points[mask]
    # idxs = pixels[:,0] + pixels[:,1]*h
    # return all_points[idxs]


def project(robot_point, K, TRC):
    xr, yr, zr = robot_point
    xc, yc, zc = TRC.dot(np.array([xr, yr, zr, 1]))
    u, v, depth = K.dot(np.array([xc, yc, zc]))
    u /= depth
    v /= depth
    px = np.array([int(u), int(v)])
    return px


def transform_points(tf, points_3d):
    points_3d = points_3d.T
    points_3d_transf = np.vstack((points_3d, np.ones([1, points_3d.shape[1]])))
    points_3d_transf = ((tf.dot(points_3d_transf)).T)[:, 0:3]
    return points_3d_transf


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    # print("Transformation: " + str(transformation))
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])


def rescale_pcd(pcd, scale=1.0):
    pcd_temp = copy.deepcopy(pcd)
    points = np.asarray(pcd.points)
    new_points = points * scale
    pcd_temp.points = o3d.utility.Vector3dVector(new_points)
    return pcd_temp


def align_pcds(pcds, tfs=None, cam_ids=None, visualize=False):

    target_pcd = pcds[0]

    threshold = 0.02
    trans_init = np.eye(4)
    scale = 2.0
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
        relative_fitness=0.000001, relative_rmse=0.000001, max_iteration=50
    )

    aligned_pcds = [target_pcd]

    target_pcd = rescale_pcd(target_pcd, scale=scale)

    for idx, source_pcd in enumerate(pcds[1:]):

        source_pcd = rescale_pcd(source_pcd, scale=scale)

        if tfs is None or not len(tfs):
            reg_p2p = o3d.registration.registration_icp(
                source_pcd,
                target_pcd,
                threshold,
                trans_init,
                o3d.registration.TransformationEstimationPointToPoint(),
                criteria,
            )

            tf = reg_p2p.transformation
            cam_id = cam_ids[idx]
            if os.path.exists("calib/icp_tf.npy"):
                icp_tf = np.load("calib/icp_tf.npy", allow_pickle=True).item()
            else:
                icp_tf = dict()
            icp_tf[cam_id] = tf

            print("Saving", icp_tf)
            np.save("calib/icp_tf.npy", icp_tf)
        else:
            tf = tfs[idx]

        source_pcd_transf = copy.deepcopy(source_pcd)
        source_pcd_transf.transform(tf)
        source_pcd_transf = rescale_pcd(source_pcd_transf, 1 / scale)

        aligned_pcds.append(source_pcd_transf)

    return aligned_pcds


# def align_pcds(pcds, visualize=False):
#    target_pcd = pcds[0]
#
#    aligned_pcds = [target_pcd]
#
#    scale = 1
#    target_pcd = rescale_pcd(target_pcd, scale=scale)
#
#    for source_pcd in pcds[1:]:
#
#        source_pcd = rescale_pcd(source_pcd, scale=scale)
#
#        pc_fix = PointCloud(np.asarray(target_pcd.points), columns=["x", "y", "z"])
#        pc_mov = PointCloud(np.asarray(source_pcd.points), columns=["x", "y", "z"])
#
#        # Create simpleICP object, add point clouds, and run algorithm!
#        icp = SimpleICP()
#        icp.add_point_clouds(pc_fix, pc_mov)
#        tf, X_mov_transformed, rigid_body_transformation_params, distance_residuals = icp.run(max_overlap_distance=0.15, neighbors=5, min_change=0.5)
#
#        #tf = reg_p2p.transformation
#
#        source_pcd_transf = copy.deepcopy(source_pcd)
#        source_pcd_transf.transform(tf)
#        source_pcd_transf = rescale_pcd(source_pcd_transf, 1/scale)
#
#        aligned_pcds.append(source_pcd_transf)
#
#    return aligned_pcds


def merge_pcls(pcls, colors, tfs=None, cam_ids=None, origin=[0, 0, 0], visualize=True):
    pcds = []
    for pcl, color in zip(pcls, colors):
        # Check if pcl needs to be converted into array
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcl)
        pcd.colors = o3d.utility.Vector3dVector(color)

        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.0)

        pcd = pcd.select_by_index(ind)
        pcds.append(pcd)

    aligned_pcds = align_pcds(pcds, tfs=tfs, cam_ids=cam_ids, visualize=visualize)

    if visualize:
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.02, origin=origin
        )
        aligned_pcds.append(mesh_frame)
        o3d.visualization.draw_geometries(aligned_pcds)

    pcd_combined = o3d.geometry.PointCloud()
    for pcd in pcds:
        pcd_combined += pcd

    return pcd_combined


def denoise(depth_img):
    max_val = np.amax(depth_img)
    min_val = np.amin(depth_img)
    normalized = depth_img - min_val / (max_val - min_val)
    normalized_vis = cv2.normalize(normalized, 0, 255, cv2.NORM_MINMAX)
    idxs = np.where(normalized_vis.ravel() > 0)[0]
    return idxs


def pix2pix_neighborhood(img, waypoint_proj, radius=3):
    height, width, _ = img.shape

    pixels = []
    for i in range(width):
        for j in range(height):
            pixels.append([i, j])

    pixels = np.array(pixels)

    nbrs = NearestNeighbors(radius=radius).fit(pixels)
    dists, idxs = nbrs.radius_neighbors(np.reshape(waypoint_proj, (-1, 2)))

    pixels = pixels[idxs[0]]
    return pixels


def point2point_neighborhood(source_points, target_points, thresh=0.03):
    nbrs = NearestNeighbors(n_neighbors=1).fit(target_points)
    dists, idxs = nbrs.kneighbors(source_points, return_distance=True)
    idxs = np.where(dists < thresh)[0]
    return idxs
