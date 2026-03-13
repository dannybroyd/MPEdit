"""
Scenario generation utilities for MPEdit experiments.

Provides:
  - generate_obstacle_scenario: Create obstacle modifications by difficulty level
  - generate_synthetic_sketch: Corrupt a valid path with Gaussian noise
  - find_closest_obstacle_point: Find the nearest point on a path to place obstacles
"""

import numpy as np
import torch

from torch_robotics.torch_utils.torch_utils import to_numpy


# ─────────────────────────────────────────────────────────────────────────────
# Obstacle scenario generation (Experiment 2)
# ─────────────────────────────────────────────────────────────────────────────

def generate_obstacle_scenario(
    env,
    reference_path,
    difficulty,
    rng=None,
    dim=None,
):
    """
    Generate obstacle modifications for a given difficulty level.

    Obstacle placement is relative to the reference path to ensure
    the difficulty matches the intended level.

    Args:
        env: environment instance (for bounds and existing obstacles)
        reference_path: (N, D) numpy array of the valid reference path
        difficulty: one of 'easy', 'medium', 'hard', 'removal'
        rng: numpy random Generator for reproducibility
        dim: environment dimensionality (auto-detected from env if None)

    Returns:
        list of modification dicts, each with keys:
            - type: 'add_sphere', 'add_box', or 'remove_sphere'
            - center, radius, sizes, index (as appropriate)
    """
    if rng is None:
        rng = np.random.default_rng()

    if dim is None:
        dim = env.dim

    path_np = to_numpy(reference_path) if torch.is_tensor(reference_path) else np.asarray(reference_path)
    # Use only position dims (for high-DOF robots, path_np might be in joint space)
    state_dim = path_np.shape[-1]

    env_limits = to_numpy(env.limits) if torch.is_tensor(env.limits) else np.asarray(env.limits)
    env_min = env_limits[0][:state_dim]
    env_max = env_limits[1][:state_dim]
    env_range = env_max - env_min

    if difficulty == "easy":
        return _generate_easy_scenario(path_np, state_dim, env_min, env_max, env_range, rng)
    elif difficulty == "medium":
        return _generate_medium_scenario(path_np, state_dim, env_min, env_max, env_range, rng)
    elif difficulty == "hard":
        return _generate_hard_scenario(path_np, state_dim, env_min, env_max, env_range, rng)
    elif difficulty == "removal":
        return _generate_removal_scenario(env, rng)
    else:
        raise ValueError(f"Unknown difficulty: {difficulty}. Use 'easy', 'medium', 'hard', or 'removal'.")


def _generate_easy_scenario(path_np, state_dim, env_min, env_max, env_range, rng):
    """
    Easy: 1 small obstacle added AWAY from the path (minor detour needed).
    Place obstacle offset from the path midpoint by 1-2 radii.
    """
    radius = float(rng.uniform(0.05, 0.10) * np.mean(env_range))
    midpoint = path_np[len(path_np) // 2]

    # Offset perpendicular to path direction at midpoint
    offset_dir = _perpendicular_direction(path_np, len(path_np) // 2, state_dim, rng)
    offset_dist = radius * rng.uniform(2.0, 4.0)
    center = midpoint + offset_dir * offset_dist

    center = np.clip(center, env_min + radius, env_max - radius)

    return [{"type": "add_sphere", "center": center.tolist(), "radius": float(radius)}]


def _generate_medium_scenario(path_np, state_dim, env_min, env_max, env_range, rng):
    """
    Medium: 1 obstacle directly BLOCKING the path (must reroute a section).
    Place obstacle on or very near the path between 30%-70% along it.
    """
    radius = float(rng.uniform(0.06, 0.12) * np.mean(env_range))

    # Pick a point along the middle portion of the path
    idx = rng.integers(int(len(path_np) * 0.3), int(len(path_np) * 0.7))
    center = path_np[idx].copy()

    # Small random perturbation (within half a radius)
    perturb = rng.normal(0, radius * 0.3, size=state_dim)
    center += perturb
    center = np.clip(center, env_min + radius, env_max - radius)

    return [{"type": "add_sphere", "center": center.tolist(), "radius": float(radius)}]


def _generate_hard_scenario(path_np, state_dim, env_min, env_max, env_range, rng):
    """
    Hard: 2-3 obstacles added, substantially changing the free space.
    Place obstacles at different points along the path.
    """
    n_obstacles = rng.integers(2, 4)  # 2 or 3
    modifications = []

    # Spread obstacles along different segments of the path
    segment_size = len(path_np) // (n_obstacles + 1)
    for k in range(n_obstacles):
        radius = float(rng.uniform(0.06, 0.12) * np.mean(env_range))
        idx = segment_size * (k + 1) + rng.integers(-segment_size // 4, segment_size // 4)
        idx = np.clip(idx, 1, len(path_np) - 2)
        center = path_np[idx].copy()

        # Small random perturbation
        perturb = rng.normal(0, radius * 0.5, size=state_dim)
        center += perturb
        center = np.clip(center, env_min + radius, env_max - radius)

        modifications.append({"type": "add_sphere", "center": center.tolist(), "radius": float(radius)})

    return modifications


def _generate_removal_scenario(env, rng):
    """
    Removal: remove an existing obstacle (can the path exploit the shortcut?).
    Pick a random fixed obstacle to remove.
    """
    from torch_robotics.environments.primitives import MultiSphereField

    for obj in env.obj_fixed_list:
        for primitive in obj.fields:
            if isinstance(primitive, MultiSphereField):
                n_spheres = primitive.centers.shape[0]
                if n_spheres > 1:
                    idx = int(rng.integers(0, n_spheres))
                    return [{"type": "remove_sphere", "index": idx}]

    # Fallback: no removable obstacles found, return empty
    return []


def _perpendicular_direction(path_np, idx, state_dim, rng):
    """Compute a unit vector roughly perpendicular to the path at index idx."""
    if idx == 0:
        tangent = path_np[1] - path_np[0]
    elif idx >= len(path_np) - 1:
        tangent = path_np[-1] - path_np[-2]
    else:
        tangent = path_np[idx + 1] - path_np[idx - 1]

    norm = np.linalg.norm(tangent)
    if norm < 1e-8:
        return rng.normal(size=state_dim)

    tangent = tangent / norm

    if state_dim == 2:
        # Simple 2D perpendicular
        perp = np.array([-tangent[1], tangent[0]])
        if rng.random() < 0.5:
            perp = -perp
        return perp
    else:
        # For higher dims, pick a random direction and remove the tangent component
        random_dir = rng.normal(size=state_dim)
        random_dir -= np.dot(random_dir, tangent) * tangent
        norm = np.linalg.norm(random_dir)
        if norm < 1e-8:
            return rng.normal(size=state_dim)
        return random_dir / norm


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic sketch generation (Experiment 3)
# ─────────────────────────────────────────────────────────────────────────────

def generate_synthetic_sketch(
    valid_path,
    noise_sigma,
    rng=None,
    fix_endpoints=True,
):
    """
    Create a synthetic sketch by corrupting a valid path with Gaussian noise.

    The noise is applied in the state space, simulating a user drawing an
    approximate version of the intended path.

    Args:
        valid_path: (N, D) numpy array of a valid, collision-free path
        noise_sigma: standard deviation of Gaussian noise (relative to env scale)
        rng: numpy random Generator for reproducibility
        fix_endpoints: if True, keep start and goal positions unchanged

    Returns:
        sketch: (N, D) numpy array of the corrupted "sketch" path
    """
    if rng is None:
        rng = np.random.default_rng()

    path_np = to_numpy(valid_path) if torch.is_tensor(valid_path) else np.asarray(valid_path, dtype=np.float64)
    noise = rng.normal(0, noise_sigma, size=path_np.shape)

    sketch = path_np + noise

    if fix_endpoints:
        sketch[0] = path_np[0]
        sketch[-1] = path_np[-1]

    return sketch


def generate_synthetic_sketches_batch(
    valid_path,
    noise_sigmas,
    n_per_sigma=1,
    rng=None,
    fix_endpoints=True,
):
    """
    Generate multiple synthetic sketches at different noise levels.

    Args:
        valid_path: (N, D) numpy array of a valid path
        noise_sigmas: list of sigma values (e.g., [0.05, 0.1, 0.2, 0.3])
        n_per_sigma: number of sketches per sigma level
        rng: numpy random Generator
        fix_endpoints: if True, keep start/goal unchanged

    Returns:
        dict mapping sigma -> list of (N, D) sketch arrays
    """
    if rng is None:
        rng = np.random.default_rng()

    sketches = {}
    for sigma in noise_sigmas:
        sketches[sigma] = [
            generate_synthetic_sketch(valid_path, sigma, rng=rng, fix_endpoints=fix_endpoints)
            for _ in range(n_per_sigma)
        ]
    return sketches


# ─────────────────────────────────────────────────────────────────────────────
# Obstacle application helpers
# ─────────────────────────────────────────────────────────────────────────────

def apply_obstacle_modifications(env, modifications, tensor_args):
    """
    Apply a list of obstacle modifications to the environment.

    Args:
        env: the environment instance
        modifications: list of modification dicts from generate_obstacle_scenario
        tensor_args: device/dtype args

    Returns:
        list of ObjectField instances that were added (for potential cleanup)
    """
    from mpd.utils.obstacle_editing import add_sphere_obstacle, add_box_obstacle, remove_fixed_obstacle_by_index

    added_objects = []
    for mod in modifications:
        mod_type = mod["type"]
        if mod_type == "add_sphere":
            obj = add_sphere_obstacle(env, mod["center"], mod["radius"], tensor_args=tensor_args)
            added_objects.append(obj)
        elif mod_type == "add_box":
            obj = add_box_obstacle(env, mod["center"], mod["sizes"], tensor_args=tensor_args)
            added_objects.append(obj)
        elif mod_type == "remove_sphere":
            remove_fixed_obstacle_by_index(env, mod["index"], tensor_args=tensor_args)
    return added_objects


def save_fixed_obstacles_state(env):
    """
    Save a deep copy of the environment's fixed obstacle state so it can be
    restored after removal-type modifications (which mutate tensors in place).

    Call this ONCE at startup, before any experiments run.

    Args:
        env: the environment instance

    Returns:
        snapshot: opaque object to pass to restore_fixed_obstacles_state()
    """
    import torch as _torch
    from torch_robotics.environments.primitives import MultiSphereField

    snapshot = []
    for obj in env.obj_fixed_list:
        field_snapshots = []
        for primitive in obj.fields:
            if isinstance(primitive, MultiSphereField):
                field_snapshots.append({
                    "type": "MultiSphereField",
                    "centers": primitive.centers.clone(),
                    "radii": primitive.radii.clone(),
                })
            else:
                field_snapshots.append(None)
        snapshot.append(field_snapshots)
    return snapshot


def restore_fixed_obstacles_state(env, snapshot):
    """
    Restore fixed obstacles from a previously saved snapshot.

    Args:
        env: the environment instance
        snapshot: return value of save_fixed_obstacles_state()
    """
    from torch_robotics.environments.primitives import MultiSphereField

    for obj, field_snapshots in zip(env.obj_fixed_list, snapshot):
        for primitive, fs in zip(obj.fields, field_snapshots):
            if fs is not None and isinstance(primitive, MultiSphereField):
                primitive.centers = fs["centers"].clone()
                primitive.radii = fs["radii"].clone()

    env.build_sdf_grid(compute_sdf_obj_fixed=True, compute_sdf_obj_extra=True)


def reset_obstacle_modifications(env, tensor_args, fixed_snapshot=None):
    """
    Remove all dynamically added obstacles and optionally restore fixed obstacles.

    Args:
        env: the environment instance
        tensor_args: device/dtype args
        fixed_snapshot: if provided, also restore fixed obstacles from this snapshot
            (needed after 'removal' scenarios that mutate fixed obstacles in place)
    """
    from mpd.utils.obstacle_editing import remove_extra_obstacles
    remove_extra_obstacles(env)

    if fixed_snapshot is not None:
        restore_fixed_obstacles_state(env, fixed_snapshot)
