"""
Utility to convert raw waypoint paths to normalized B-spline control points
for use with the MPD diffusion model's SDEdit pipeline.
"""

import numpy as np
import torch

from pb_ompl.pb_ompl import fit_bspline_to_path
from torch_robotics.torch_utils.torch_utils import to_torch, to_numpy


def path_to_normalized_control_points(
    path,
    parametric_trajectory,
    dataset,
    tensor_args=None,
):
    """
    Convert a raw waypoint path to normalized B-spline control points.
    
    Args:
        path: (N, state_dim) numpy array or torch tensor of waypoints
        parametric_trajectory: the planning task's ParametricTrajectoryBspline instance
        dataset: the TrajectoryDatasetBspline instance (provides normalization)
        tensor_args: dict with 'device' and 'dtype'
        
    Returns:
        control_points_normalized: (n_learnable_control_points, state_dim) normalized tensor
    """
    if tensor_args is None:
        tensor_args = dataset.tensor_args

    path_np = to_numpy(path, dtype=np.float64)

    # Fit B-spline to the waypoint path
    bspline_params = fit_bspline_to_path(
        path_np,
        parametric_trajectory.bspline.d,
        parametric_trajectory.bspline.n_pts,
        parametric_trajectory.zero_vel_at_start_and_goal,
        parametric_trajectory.zero_acc_at_start_and_goal,
    )

    # Extract control points from the B-spline parameters (tck format: t, c, k)
    # bspline_params[1] is the control points array, shape (state_dim, n_control_points)
    control_points_np = bspline_params[1].T  # (n_control_points, state_dim)
    control_points = to_torch(control_points_np, **tensor_args)

    # Shape: (1, 1, n_control_points, state_dim) for remove_control_points_fn
    control_points_all = control_points[None, None, ...]

    # Remove outer control points (start/goal, zero vel/acc constraints)
    # Result: (1, 1, n_learnable_control_points, state_dim)
    control_points_inner = parametric_trajectory.remove_control_points_fn(control_points_all)

    # Normalize to [-1, 1] using the dataset normalizer
    control_points_normalized = dataset.normalize_control_points(control_points_inner)

    # Squeeze the batch/iter dims: (n_learnable_control_points, state_dim)
    return control_points_normalized.squeeze(0).squeeze(0)
