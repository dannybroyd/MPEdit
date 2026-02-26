"""
Utilities for dynamically modifying obstacles in the environment
for SDEdit-style re-planning scenarios.
"""

import torch
import numpy as np

from torch_robotics.torch_utils.torch_utils import to_torch, DEFAULT_TENSOR_ARGS


def add_sphere_obstacle(env, center, radius, tensor_args=DEFAULT_TENSOR_ARGS):
    """
    Add a sphere obstacle to the environment and recompute SDF.
    
    Args:
        env: the environment instance (e.g., EnvSimple2D)
        center: (dim,) center of the sphere
        radius: float radius
        tensor_args: device/dtype args
        
    Returns:
        The ObjectField that was added (for potential removal later)
    """
    from torch_robotics.environments.primitives import MultiSphereField, ObjectField

    centers = to_torch(np.array([center]), **tensor_args)
    radii = to_torch(np.array([radius]), **tensor_args)
    sphere_field = MultiSphereField(centers, radii, tensor_args=tensor_args)
    obj = ObjectField([sphere_field], "sdedit-added-sphere")
    env.add_objects_extra([obj])
    return obj


def add_box_obstacle(env, center, sizes, tensor_args=DEFAULT_TENSOR_ARGS):
    """
    Add a box obstacle to the environment and recompute SDF.
    
    Args:
        env: the environment instance
        center: (dim,) center of the box
        sizes: (dim,) half-extents of the box
        tensor_args: device/dtype args
        
    Returns:
        The ObjectField that was added
    """
    from torch_robotics.environments.primitives import MultiBoxField, ObjectField

    centers = to_torch(np.array([center]), **tensor_args)
    sizes = to_torch(np.array([sizes]), **tensor_args)
    box_field = MultiBoxField(centers, sizes, tensor_args=tensor_args)
    obj = ObjectField([box_field], "sdedit-added-box")
    env.add_objects_extra([obj])
    return obj


def remove_extra_obstacles(env):
    """
    Remove all extra obstacles from the environment and recompute SDF.
    
    Args:
        env: the environment instance
    """
    env.obj_extra_list.clear()
    env.update_objects_extra()


def remove_fixed_obstacle_by_index(env, obstacle_idx, tensor_args=DEFAULT_TENSOR_ARGS):
    """
    Remove a fixed obstacle by index. Since fixed obstacles are stored in a
    MultiSphereField, we rebuild it without the specified obstacle.
    
    Args:
        env: the environment instance
        obstacle_idx: index of the obstacle to remove within the fixed object list's first field
        tensor_args: device/dtype args
        
    Note: This modifies the fixed obstacle list and recomputes the SDF grid.
          Use with caution — the model was trained with all fixed obstacles present.
    """
    from torch_robotics.environments.primitives import MultiSphereField, ObjectField

    # Get the first (and typically only) fixed object field
    obj_fixed = env.obj_fixed_list[0]
    # The primitive fields inside the ObjectField
    for primitive in obj_fixed.fields:
        if isinstance(primitive, MultiSphereField):
            n_spheres = primitive.centers.shape[0]
            if obstacle_idx < 0 or obstacle_idx >= n_spheres:
                raise ValueError(f"obstacle_idx {obstacle_idx} out of range [0, {n_spheres})")

            # Create mask excluding the obstacle
            mask = torch.ones(n_spheres, dtype=torch.bool)
            mask[obstacle_idx] = False

            # Rebuild with remaining obstacles
            primitive.centers = primitive.centers[mask]
            primitive.radii = primitive.radii[mask]
            break

    # Recompute SDF grid for fixed objects
    env.build_sdf_grid(compute_sdf_obj_fixed=True, compute_sdf_obj_extra=True)
