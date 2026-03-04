"""Grasp filtering and selection utilities."""

import numpy as np
import scipy.spatial.transform as st
import logging

logger = logging.getLogger(__name__)


def normalize_grasp_z_symmetry(grasps: np.ndarray, eef_transform: np.ndarray) -> tuple[np.ndarray, int]:
    """
    For each grasp, pick whichever of original or +180 deg around Z
    is closer to current EEF orientation.
    
    Args:
        grasps: Nx4x4 grasp poses
        eef_transform: 4x4 current EEF transform
    
    Returns:
        Tuple of (normalized grasps, number flipped)
    """
    Rz180 = np.eye(4)
    Rz180[0, 0] = -1
    Rz180[1, 1] = -1

    eef_rot_scipy = st.Rotation.from_matrix(eef_transform[:3, :3])
    n_flipped = 0
    for i in range(len(grasps)):
        rot_orig = st.Rotation.from_matrix(grasps[i, :3, :3])
        rot_flip = rot_orig * st.Rotation.from_matrix(Rz180[:3, :3])
        delta_orig = (rot_orig * eef_rot_scipy.inv()).magnitude()
        delta_flip = (rot_flip * eef_rot_scipy.inv()).magnitude()
        if delta_flip < delta_orig:
            grasps[i] = grasps[i] @ Rz180
            n_flipped += 1
    return grasps, n_flipped


def select_best_grasp(
    grasps: np.ndarray,
    scores: np.ndarray,
    eef_transform: np.ndarray | None = None,
) -> tuple[int, np.ndarray, float]:
    """
    Select the best grasp: prefer top 50% most vertical, then closest to EEF.
    
    Args:
        grasps: Nx4x4 grasp poses
        scores: N confidence scores
        eef_transform: 4x4 current EEF transform (optional)
    
    Returns:
        Tuple of (best_idx, best_grasp_4x4, best_score)
    """
    down = np.array([0.0, 0.0, -1.0])
    topdown_scores = np.array([np.dot(g[:3, 2], down) for g in grasps])

    topdown_threshold = np.median(topdown_scores)
    topdown_mask = topdown_scores >= topdown_threshold
    n_topdown = topdown_mask.sum()
    logger.info(f"  Vertical preference: keeping {n_topdown}/{len(grasps)} above median ({topdown_threshold:.3f})")

    if eef_transform is not None:
        eef_pos = eef_transform[:3, 3]
        distances = np.linalg.norm(grasps[:, :3, 3] - eef_pos, axis=1)
        masked_distances = np.where(topdown_mask, distances, np.inf)
        best_idx = int(np.argmin(masked_distances))
    else:
        masked_scores = np.where(topdown_mask, scores, -1)
        best_idx = int(np.argmax(masked_scores))

    return best_idx, grasps[best_idx], scores[best_idx]
