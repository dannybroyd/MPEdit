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
    robot=None,
    tensor_args=None,
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

    path_np = _path_to_obstacle_space(
        reference_path,
        env_dim=dim,
        robot=robot,
        tensor_args=tensor_args,
    )
    state_dim = path_np.shape[-1]

    env_limits = to_numpy(env.limits) if torch.is_tensor(env.limits) else np.asarray(env.limits)
    env_min = env_limits[0][:state_dim]
    env_max = env_limits[1][:state_dim]
    env_range = env_max - env_min
    nominal_radius = _estimate_nominal_obstacle_radius(env, env_range)

    if difficulty == "easy":
        return _generate_easy_scenario(
            path_np, state_dim, env_min, env_max, env_range, rng, dim, nominal_radius
        )
    elif difficulty == "medium":
        return _generate_medium_scenario(
            path_np, state_dim, env_min, env_max, env_range, rng, dim, nominal_radius
        )
    elif difficulty == "hard":
        return _generate_hard_scenario(
            path_np, state_dim, env_min, env_max, env_range, rng, dim, nominal_radius
        )
    elif difficulty == "removal":
        return _generate_removal_scenario(env, rng)
    else:
        raise ValueError(f"Unknown difficulty: {difficulty}. Use 'easy', 'medium', 'hard', or 'removal'.")


def _path_to_obstacle_space(path, env_dim, robot=None, tensor_args=None):
    """
    Convert a planning path to obstacle-space coordinates (env_dim).

    For high-DOF robots (e.g., Panda 7D joint-space paths in a 3D environment),
    this maps the path to task space via robot FK before obstacle placement.
    """
    path_np = to_numpy(path) if torch.is_tensor(path) else np.asarray(path, dtype=np.float64)
    if path_np.ndim != 2:
        raise ValueError(f"Expected path shape (N, D), got {path_np.shape}")

    # Already in obstacle space (2D/3D)
    if path_np.shape[-1] == env_dim:
        return path_np

    # High-DOF path requires FK to map to task space
    if path_np.shape[-1] > env_dim:
        if robot is None:
            raise ValueError(
                f"Path has dim {path_np.shape[-1]} but env_dim is {env_dim}. "
                f"Pass robot=... so FK can map joint-space path to obstacle space."
            )

        ta = tensor_args or {"device": "cpu", "dtype": torch.float32}
        q = torch.as_tensor(path_np, device=ta.get("device", "cpu"), dtype=ta.get("dtype", torch.float32))
        if q.dim() == 2:
            q = q.unsqueeze(0)  # (1, N, q_dim)

        fk = robot.fk_map_collision(q)
        fk_np = to_numpy(fk)

        # Typical shape: (1, N, n_links, 3)
        if fk_np.ndim == 4:
            fk_np = fk_np.squeeze(0)  # (N, n_links, 3)
        if fk_np.ndim == 3:
            # End-effector position
            return fk_np[:, -1, :env_dim]
        if fk_np.ndim == 2:
            return fk_np[:, :env_dim]

        raise ValueError(f"Unexpected FK output shape: {fk_np.shape}")

    raise ValueError(
        f"Path dim {path_np.shape[-1]} is smaller than env_dim {env_dim}; cannot place obstacles."
    )


def _estimate_nominal_obstacle_radius(env, env_range):
    """
    Estimate a representative obstacle radius from fixed obstacles in the env.
    Falls back to a fraction of workspace range if radii are unavailable.
    """
    radii = []
    for obj in getattr(env, "obj_fixed_list", []):
        for primitive in getattr(obj, "fields", []):
            if hasattr(primitive, "radii"):
                r = to_numpy(primitive.radii) if torch.is_tensor(primitive.radii) else np.asarray(primitive.radii)
                radii.extend(np.asarray(r, dtype=np.float64).reshape(-1).tolist())

    if radii:
        return float(np.median(radii))

    return float(0.05 * np.mean(env_range))


def _generate_easy_scenario(path_np, state_dim, env_min, env_max, env_range, rng, env_dim, nominal_radius):
    """
    Easy: 1 small obstacle added AWAY from the path (minor detour needed).
    Place obstacle offset from the path midpoint by 1-2 radii.
    """
    if env_dim >= 3:
        radius = float(rng.uniform(0.25, 0.40) * nominal_radius)
    else:
        radius = float(rng.uniform(0.05, 0.10) * np.mean(env_range))
    midpoint = path_np[len(path_np) // 2]

    # Offset perpendicular to path direction at midpoint
    offset_dir = _perpendicular_direction(path_np, len(path_np) // 2, state_dim, rng)
    offset_dist = radius * rng.uniform(2.0, 4.0)
    center = midpoint + offset_dir * offset_dist

    if env_dim >= 3:
        center = _push_center_away_from_endpoints(center, path_np, radius, env_min, env_max, rng, min_factor=4.5)
    center = np.clip(center, env_min + radius, env_max - radius)

    return [{"type": "add_sphere", "center": center.tolist(), "radius": float(radius)}]


def _generate_medium_scenario(path_np, state_dim, env_min, env_max, env_range, rng, env_dim, nominal_radius):
    """
    Medium: 1 obstacle directly BLOCKING the path (must reroute a section).
    Place obstacle on or very near the path between 30%-70% along it.
    """
    if env_dim >= 3:
        radius = float(rng.uniform(0.40, 0.65) * nominal_radius)
    else:
        radius = float(rng.uniform(0.06, 0.12) * np.mean(env_range))

    # Pick a point along the middle portion of the path
    if env_dim >= 3:
        idx = rng.integers(int(len(path_np) * 0.45), int(len(path_np) * 0.8))
    else:
        idx = rng.integers(int(len(path_np) * 0.3), int(len(path_np) * 0.7))
    center = path_np[idx].copy()

    # Small random perturbation (within half a radius)
    perturb_scale = 0.1 if env_dim >= 3 else 0.3
    perturb = rng.normal(0, radius * perturb_scale, size=state_dim)
    center += perturb
    if env_dim >= 3:
        center = _push_center_away_from_endpoints(center, path_np, radius, env_min, env_max, rng, min_factor=5.0)
    center = np.clip(center, env_min + radius, env_max - radius)

    return [{"type": "add_sphere", "center": center.tolist(), "radius": float(radius)}]


def _generate_hard_scenario(path_np, state_dim, env_min, env_max, env_range, rng, env_dim, nominal_radius):
    """
    Hard: 2-3 obstacles added, substantially changing the free space.
    Place obstacles at different points along the path.
    """
    n_obstacles = int(rng.integers(2, 3)) if env_dim >= 3 else int(rng.integers(2, 4))
    modifications = []

    # Spread obstacles along different segments of the path
    segment_size = len(path_np) // (n_obstacles + 1)
    for k in range(n_obstacles):
        if env_dim >= 3:
            radius = float(rng.uniform(0.35, 0.60) * nominal_radius)
        else:
            radius = float(rng.uniform(0.06, 0.12) * np.mean(env_range))
        idx = segment_size * (k + 1) + rng.integers(-segment_size // 4, segment_size // 4)
        idx = np.clip(idx, 1, len(path_np) - 2)
        center = path_np[idx].copy()

        # Small random perturbation
        perturb_scale = 0.2 if env_dim >= 3 else 0.5
        perturb = rng.normal(0, radius * perturb_scale, size=state_dim)
        center += perturb
        if env_dim >= 3:
            center = _push_center_away_from_endpoints(center, path_np, radius, env_min, env_max, rng, min_factor=5.0)
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


def _push_center_away_from_endpoints(center, path_np, radius, env_min, env_max, rng, min_factor=5.0):
    """
    Push obstacle center away from path endpoints to reduce start/goal invalidation.
    """
    center = np.asarray(center, dtype=np.float64)
    endpoints = [path_np[0], path_np[-1]]
    min_dist = float(min_factor * radius)

    for endpoint in endpoints:
        delta = center - endpoint
        dist = np.linalg.norm(delta)
        if dist < min_dist:
            if dist < 1e-8:
                direction = rng.normal(size=center.shape[0])
                norm = np.linalg.norm(direction)
                if norm < 1e-8:
                    direction = np.ones_like(center)
                    norm = np.linalg.norm(direction)
                direction = direction / norm
            else:
                direction = delta / dist
            center = endpoint + direction * min_dist

    center = np.clip(center, env_min + radius, env_max - radius)
    return center


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
