"""
SDEdit-specific metrics for evaluating path editing quality.

Provides:
  - discrete_frechet_distance: Fréchet distance between two paths
  - mean_l2_distance: Per-waypoint mean L2 distance between two paths
  - pairwise_frechet_diversity: Mean pairwise Fréchet distance among a set of paths
  - compute_sdedit_metrics: Batch computation of all SDEdit metrics
"""

import numpy as np
import torch

from torch_robotics.torch_utils.torch_utils import to_numpy


def _as_numpy_float64(x):
    """Convert tensor/array-like input to float64 numpy array."""
    arr = to_numpy(x) if torch.is_tensor(x) else np.asarray(x)
    return np.asarray(arr, dtype=np.float64)


def discrete_frechet_distance(path_a, path_b):
    """
    Compute the discrete Fréchet distance between two paths.

    Uses the standard dynamic programming algorithm on pairwise Euclidean distances.

    Args:
        path_a: (N, D) numpy array or torch tensor
        path_b: (M, D) numpy array or torch tensor

    Returns:
        float: the discrete Fréchet distance
    """
    P = _as_numpy_float64(path_a)
    Q = _as_numpy_float64(path_b)

    # Compute all point-to-point distances in one vectorized pass.
    # This removes millions of tiny Python-level norm calls in batch experiments.
    dist = np.linalg.norm(P[:, None, :] - Q[None, :, :], axis=-1)
    n, m = dist.shape
    ca = np.empty((n, m), dtype=np.float64)

    # DP bottom-up for efficiency
    ca[0, 0] = dist[0, 0]

    for i in range(1, n):
        prev = ca[i - 1, 0]
        d = dist[i, 0]
        ca[i, 0] = d if d > prev else prev

    for j in range(1, m):
        prev = ca[0, j - 1]
        d = dist[0, j]
        ca[0, j] = d if d > prev else prev

    for i in range(1, n):
        row = ca[i]
        prev_row = ca[i - 1]
        dist_row = dist[i]
        left = row[0]
        for j in range(1, m):
            v = prev_row[j]
            d = prev_row[j - 1]
            if d < v:
                v = d
            if left < v:
                v = left
            d = dist_row[j]
            if d > v:
                v = d
            row[j] = v
            left = v

    return float(ca[n - 1, m - 1])


def mean_l2_distance(path_a, path_b):
    """
    Per-waypoint mean L2 distance between two paths of equal length.

    If paths differ in length, the shorter is linearly interpolated to match.

    Args:
        path_a: (N, D) numpy array or torch tensor
        path_b: (M, D) numpy array or torch tensor

    Returns:
        float: mean per-waypoint Euclidean distance
    """
    P = _as_numpy_float64(path_a)
    Q = _as_numpy_float64(path_b)

    # Resample to the same length if needed
    if len(P) != len(Q):
        target_len = max(len(P), len(Q))
        P = _resample_path(P, target_len)
        Q = _resample_path(Q, target_len)

    dists = np.linalg.norm(P - Q, axis=-1)
    return float(np.mean(dists))


def pairwise_frechet_diversity(paths, max_pairs=500):
    """
    Compute mean pairwise discrete Fréchet distance among a set of paths.

    For large sets, samples a random subset of pairs to keep computation tractable.

    Args:
        paths: (B, N, D) numpy array or torch tensor — batch of paths
        max_pairs: maximum number of pairs to sample (for efficiency)

    Returns:
        float: mean pairwise Fréchet distance
    """
    P = _as_numpy_float64(paths)
    B = P.shape[0]
    if B < 2:
        return 0.0

    pairs = np.transpose(np.triu_indices(B, k=1))
    n_total_pairs = pairs.shape[0]
    if n_total_pairs > max_pairs:
        rng = np.random.default_rng(42)
        sampled_idx = rng.choice(n_total_pairs, size=max_pairs, replace=False)
        pairs = pairs[sampled_idx]

    distances = np.empty(pairs.shape[0], dtype=np.float64)
    for idx, (i, j) in enumerate(pairs):
        distances[idx] = discrete_frechet_distance(P[i], P[j])
    return float(distances.mean())


def pairwise_l2_diversity(paths):
    """
    Compute mean pairwise mean-L2 distance among a set of equal-length paths.

    More efficient than Fréchet for large batches since it vectorizes.

    Args:
        paths: (B, N, D) numpy array or torch tensor

    Returns:
        float: mean pairwise mean-L2 distance
    """
    P = _as_numpy_float64(paths)
    B = P.shape[0]
    if B < 2:
        return 0.0

    total = 0.0
    count = 0
    for i in range(B - 1):
        diffs = P[i + 1:] - P[i]
        mean_dists = np.linalg.norm(diffs, axis=-1).mean(axis=-1)
        total += float(mean_dists.sum())
        count += mean_dists.shape[0]
    return float(total / count)


def compute_sdedit_metrics(
    input_path,
    regenerated_paths,
    best_path=None,
    planning_task=None,
    robot=None,
    compute_pairwise_diversity=True,
    pairwise_frechet_max_pairs=200,
):
    """
    Compute all SDEdit-specific metrics for a single run.

    Args:
        input_path: (N, D) the original / reference path
        regenerated_paths: (B, N, D) all regenerated paths from SDEdit
        best_path: (N, D) the selected best path (optional)
        planning_task: PlanningTask instance (for collision checking)
        robot: Robot instance (for path length / smoothness via existing metrics)
        compute_pairwise_diversity: if False, skip expensive pairwise diversity
            metrics and return them as None (except B < 2, where they are 0.0).
        pairwise_frechet_max_pairs: max number of sampled pairs for pairwise
            Fréchet diversity.

    Returns:
        dict with keys:
            - success_rate: fraction of collision-free paths
            - frechet_to_input_mean: mean Fréchet distance from each output to input
            - frechet_to_input_std: std of Fréchet distances
            - l2_to_input_mean: mean per-waypoint L2 distance from each output to input
            - l2_to_input_std: std of L2 distances
            - pairwise_diversity_frechet: mean pairwise Fréchet distance among outputs
            - pairwise_diversity_l2: mean pairwise L2 distance among outputs
            - path_length_mean/std: mean/std path length (if robot provided)
            - smoothness_mean/std: mean/std smoothness (if robot provided)
            - best_frechet_to_input: Fréchet of best path to input (if best_path provided)
            - best_l2_to_input: mean L2 of best path to input (if best_path provided)
    """
    input_np = _as_numpy_float64(input_path)
    regen_np = _as_numpy_float64(regenerated_paths)

    metrics = {}
    B = regen_np.shape[0]

    # Success rate (collision-free)
    if planning_task is not None:
        try:
            # Use the planning_task's tensor_args for correct device placement
            ta = getattr(planning_task, "tensor_args", {"device": "cpu", "dtype": torch.float32})
            regen_t = torch.as_tensor(regen_np, device=ta["device"], dtype=ta["dtype"])
            fraction_valid = planning_task.compute_fraction_valid_trajs(
                regen_t, filter_joint_limits_vel_acc=False
            )
            metrics["success_rate"] = float(fraction_valid)
        except Exception:
            metrics["success_rate"] = None
    else:
        metrics["success_rate"] = None

    # Fréchet distance to input for each regenerated path
    frechet_dists = np.empty(B, dtype=np.float64)
    l2_dists = np.empty(B, dtype=np.float64)
    for i in range(B):
        frechet_dists[i] = discrete_frechet_distance(input_np, regen_np[i])
        l2_dists[i] = mean_l2_distance(input_np, regen_np[i])

    metrics["frechet_to_input_mean"] = float(frechet_dists.mean())
    metrics["frechet_to_input_std"] = float(frechet_dists.std())
    metrics["l2_to_input_mean"] = float(l2_dists.mean())
    metrics["l2_to_input_std"] = float(l2_dists.std())

    # Pairwise diversity
    if B >= 2 and compute_pairwise_diversity:
        metrics["pairwise_diversity_l2"] = pairwise_l2_diversity(regen_np)
        metrics["pairwise_diversity_frechet"] = pairwise_frechet_diversity(
            regen_np, max_pairs=pairwise_frechet_max_pairs,
        )
    elif B >= 2:
        metrics["pairwise_diversity_frechet"] = None
        metrics["pairwise_diversity_l2"] = None
    else:
        metrics["pairwise_diversity_frechet"] = 0.0
        metrics["pairwise_diversity_l2"] = 0.0

    # Best path metrics
    if best_path is not None:
        best_np = _as_numpy_float64(best_path)
        metrics["best_frechet_to_input"] = discrete_frechet_distance(input_np, best_np)
        metrics["best_l2_to_input"] = mean_l2_distance(input_np, best_np)
    else:
        metrics["best_frechet_to_input"] = None
        metrics["best_l2_to_input"] = None

    # Path length and smoothness (using existing infrastructure)
    if robot is not None:
        try:
            from torch_robotics.trajectory.metrics import compute_path_length, compute_smoothness
            regen_t = torch.as_tensor(regen_np, dtype=torch.float32)
            path_lengths = compute_path_length(regen_t, robot)
            metrics["path_length_mean"] = float(path_lengths.mean())
            metrics["path_length_std"] = float(path_lengths.std())

            smoothness = compute_smoothness(regen_t, robot)
            metrics["smoothness_mean"] = float(smoothness.mean())
            metrics["smoothness_std"] = float(smoothness.std())
        except Exception as e:
            metrics["path_length_mean"] = None
            metrics["path_length_std"] = None
            metrics["smoothness_mean"] = None
            metrics["smoothness_std"] = None
    else:
        metrics["path_length_mean"] = None
        metrics["path_length_std"] = None
        metrics["smoothness_mean"] = None
        metrics["smoothness_std"] = None

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _resample_path(path, target_len):
    """Linearly interpolate a path to have target_len waypoints."""
    N, D = path.shape
    old_t = np.linspace(0, 1, N)
    new_t = np.linspace(0, 1, target_len)
    resampled = np.zeros((target_len, D))
    for d in range(D):
        resampled[:, d] = np.interp(new_t, old_t, path[:, d])
    return resampled
