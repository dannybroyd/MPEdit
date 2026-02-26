"""
Visualization utilities for SDEdit-style path regeneration results.

Provides:
  - plot_sdedit_before_after: 3-panel static figure (Before / Obstacle Edit / After)
  - animate_sdedit_denoising: mp4 video of the SDEdit denoising process
  - plot_sdedit_results: single-panel result plot
  - plot_noise_level_comparison: side-by-side noise level comparison
"""

import os

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.patches import Circle

from torch_robotics.torch_utils.torch_utils import to_numpy
from torch_robotics.visualizers.plot_utils import create_fig_and_axes, create_animation_video


# ─────────────────────────────────────────────────────────────────────────────
# 3-panel Before / Obstacle Edit / After
# ─────────────────────────────────────────────────────────────────────────────

def plot_sdedit_before_after(
    planning_task,
    q_pos_start,
    q_pos_goal,
    input_path,
    best_regen_path=None,
    all_regen_paths=None,
    obstacle_modification=None,
    title="SDEdit Re-planning",
    figsize=(24, 8),
    save_path=None,
):
    """
    Three-panel figure showing the full SDEdit story:
      Panel 1 – BEFORE: original obstacle map + start/goal + valid input path
      Panel 2 – OBSTACLE EDIT: the modified obstacle map (new/removed obstacle highlighted)
      Panel 3 – AFTER: modified obstacle map + regenerated paths + best path

    Uses planning_task.env.render(ax) for obstacle rendering (consistent with MPD visuals).

    Args:
        planning_task: the PlanningTask instance
        q_pos_start: (state_dim,) start position
        q_pos_goal:  (state_dim,) goal position
        input_path:  (N, state_dim) original valid path (numpy or tensor)
        best_regen_path: (N, state_dim) best regenerated path (optional)
        all_regen_paths: (n_samples, N, state_dim) all regenerated paths (optional)
        obstacle_modification: dict with keys 'type', 'center', 'radius' / 'sizes', 'index'
        save_path: if set, saves the figure to this path
    """
    env = planning_task.env
    robot = planning_task.robot
    dim = env.dim

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    ax_before, ax_edit, ax_after = axes

    start_np = to_numpy(q_pos_start)
    goal_np  = to_numpy(q_pos_goal)
    input_np = to_numpy(input_path) if torch.is_tensor(input_path) else np.asarray(input_path)

    # ── Panel 1: BEFORE ──────────────────────────────────────────────────
    ax_before.set_title("Before (original map + path)", fontsize=13)
    # Render the environment — for the "before" panel we want the original
    # obstacle layout.  If an obstacle was *added*, the env currently contains
    # the extra object, so we draw fixed only.  If one was *removed* we need
    # to convey the original (which we cannot easily restore), so we just
    # render the current env and annotate.
    _render_env_on_ax(ax_before, env, dim, draw_extra=False)
    _draw_path(ax_before, input_np, color="blue", linewidth=2.5, label="Valid Path", zorder=5)
    _draw_start_goal(ax_before, start_np, goal_np, robot, dim)
    ax_before.legend(loc="upper left", fontsize=9)

    # ── Panel 2: OBSTACLE EDIT ───────────────────────────────────────────
    ax_edit.set_title("Obstacle map after edit", fontsize=13)
    _render_env_on_ax(ax_edit, env, dim, draw_extra=True)
    _draw_path(ax_edit, input_np, color="blue", linewidth=1.5, alpha=0.4, linestyle="--",
               label="Old Path", zorder=4)
    _draw_start_goal(ax_edit, start_np, goal_np, robot, dim)
    _annotate_obstacle_mod(ax_edit, obstacle_modification)
    ax_edit.legend(loc="upper left", fontsize=9)

    # ── Panel 3: AFTER ───────────────────────────────────────────────────
    ax_after.set_title("After SDEdit (regenerated paths)", fontsize=13)
    _render_env_on_ax(ax_after, env, dim, draw_extra=True)
    _draw_path(ax_after, input_np, color="blue", linewidth=1.5, alpha=0.35, linestyle="--",
               label="Old Path", zorder=4)

    if all_regen_paths is not None:
        regen_np = to_numpy(all_regen_paths) if torch.is_tensor(all_regen_paths) else np.asarray(all_regen_paths)
        # Colour code: orange = collision, green = free  (use planning_task for validation)
        for i in range(min(regen_np.shape[0], 60)):
            ax_after.plot(regen_np[i, :, 0], regen_np[i, :, 1],
                          color="orange", alpha=0.15, linewidth=1.0, zorder=3)

    if best_regen_path is not None:
        best_np = to_numpy(best_regen_path) if torch.is_tensor(best_regen_path) else np.asarray(best_regen_path)
        ax_after.plot(best_np[:, 0], best_np[:, 1],
                      color="green", linewidth=2.5, label="Best Regenerated", zorder=6)

    _draw_start_goal(ax_after, start_np, goal_np, robot, dim)
    ax_after.legend(loc="upper left", fontsize=9)

    plt.suptitle(title, fontsize=15, y=1.01)
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved before/after plot to {save_path}")
    return fig, axes


# ─────────────────────────────────────────────────────────────────────────────
# SDEdit denoising animation (mp4 video)
# ─────────────────────────────────────────────────────────────────────────────

def animate_sdedit_denoising(
    planning_task,
    q_pos_start,
    q_pos_goal,
    input_path,
    trajs_pos_iters,
    traj_pos_best=None,
    video_filepath="sdedit_denoising.mp4",
    n_frames=None,
    anim_time=5.0,
):
    """
    Create an mp4 video showing the SDEdit denoising process frame by frame,
    similar to ``animate_opt_iters_joint_space_env`` but with the original input
    path drawn as a dashed blue reference.

    Args:
        planning_task: the PlanningTask instance
        q_pos_start: (state_dim,) start position
        q_pos_goal:  (state_dim,) goal position
        input_path:  (N, state_dim) the original input path (numpy or tensor)
        trajs_pos_iters: (S, B, H, D) denoising iterations tensor
            S = number of diffusion steps recorded, B = batch, H = horizon, D = state_dim
        traj_pos_best: (H, D) best trajectory (shown on last frame)
        video_filepath: output .mp4 path
        n_frames: number of frames in the video (defaults to S)
        anim_time: video duration in seconds
    """
    env = planning_task.env
    dim = env.dim

    assert trajs_pos_iters.ndim == 4, f"Expected (S, B, H, D), got {trajs_pos_iters.shape}"
    S, B, H, D = trajs_pos_iters.shape

    if n_frames is None:
        n_frames = max(2, S)

    idxs = np.round(np.linspace(0, S - 1, n_frames)).astype(int)
    trajs_selection = trajs_pos_iters[idxs]  # (n_frames, B, H, D)

    input_np = to_numpy(input_path) if torch.is_tensor(input_path) else np.asarray(input_path)
    start_np = to_numpy(q_pos_start)
    goal_np  = to_numpy(q_pos_goal)

    fig, ax = create_fig_and_axes(dim=dim)

    def animate_fn(i, ax):
        ax.clear()
        ax.set_title(f"SDEdit denoising — iter {idxs[i]}/{S-1}")

        # Environment obstacles
        env.render(ax)

        # Original input path (dashed reference)
        ax.plot(input_np[:, 0], input_np[:, 1], color="blue", linewidth=2.0,
                alpha=0.4, linestyle="--", zorder=4, label="Input Path")

        # Current iteration trajectories — colour by collision validity
        trajs_unvalid, trajs_valid = planning_task.get_trajs_unvalid_and_valid(trajs_selection[i])
        if trajs_unvalid is not None:
            for traj in trajs_unvalid:
                traj_np = to_numpy(traj)
                ax.plot(traj_np[:, 0], traj_np[:, 1], color="black",
                        linewidth=1.2, alpha=0.6, zorder=5)
        if trajs_valid is not None:
            for traj in trajs_valid:
                traj_np = to_numpy(traj)
                ax.plot(traj_np[:, 0], traj_np[:, 1], color="orange",
                        linewidth=1.2, alpha=0.6, zorder=5)

        # Best trajectory on the last frame
        if traj_pos_best is not None and i == n_frames - 1:
            best_np = to_numpy(traj_pos_best)
            ax.plot(best_np[:, 0], best_np[:, 1], color="green",
                    linewidth=2.5, zorder=10, label="Best")

        # Start / goal markers
        ax.scatter(start_np[0], start_np[1], color="blue", marker="X",
                   s=10**2.5, zorder=100, label="Start")
        ax.scatter(goal_np[0], goal_np[1], color="red", marker="X",
                   s=10**2.5, zorder=100, label="Goal")

        ax.legend(loc="upper left", fontsize=8)

    create_animation_video(
        fig, animate_fn, n_frames=n_frames, fargs=(ax,),
        anim_time=anim_time, video_filepath=video_filepath,
    )
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Single-panel result plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_sdedit_results(
    env,
    q_pos_start,
    q_pos_goal,
    input_path,
    regenerated_paths,
    best_path=None,
    title="SDEdit Path Regeneration",
    obstacle_modification=None,
    figsize=(10, 10),
    save_path=None,
):
    """
    Single plot: obstacles + input path + regenerated paths + best path.

    Args:
        env: environment instance (for obstacle rendering)
        q_pos_start: (2,) start position
        q_pos_goal: (2,) goal position
        input_path: (N, 2) original input path
        regenerated_paths: (n_samples, N, 2) regenerated paths
        best_path: (N, 2) best selected path (optional)
        obstacle_modification: dict describing what changed (for annotation)
        save_path: file path to save the figure
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    env.render(ax)

    regen_np = to_numpy(regenerated_paths) if torch.is_tensor(regenerated_paths) else regenerated_paths
    for i in range(min(regen_np.shape[0], 50)):
        ax.plot(regen_np[i, :, 0], regen_np[i, :, 1], color='orange', alpha=0.15, linewidth=1.0)

    input_np = to_numpy(input_path) if torch.is_tensor(input_path) else input_path
    ax.plot(input_np[:, 0], input_np[:, 1], color='blue', linewidth=2.5, label='Input Path', zorder=5)

    if best_path is not None:
        best_np = to_numpy(best_path) if torch.is_tensor(best_path) else best_path
        ax.plot(best_np[:, 0], best_np[:, 1], color='green', linewidth=2.5, label='Best Regenerated', zorder=6)

    start_np = to_numpy(q_pos_start) if torch.is_tensor(q_pos_start) else q_pos_start
    goal_np = to_numpy(q_pos_goal) if torch.is_tensor(q_pos_goal) else q_pos_goal
    ax.plot(*start_np, 'go', markersize=12, zorder=10, label='Start')
    ax.plot(*goal_np, 'r*', markersize=15, zorder=10, label='Goal')

    _annotate_obstacle_mod(ax, obstacle_modification)

    ax.set_aspect('equal')
    ax.legend(loc='upper left', fontsize=10)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    return fig, ax


# ─────────────────────────────────────────────────────────────────────────────
# Noise-level comparison (side-by-side panels)
# ─────────────────────────────────────────────────────────────────────────────

def plot_noise_level_comparison(
    env,
    q_pos_start,
    q_pos_goal,
    input_path,
    results_by_noise_level,
    figsize=(20, 5),
    save_path=None,
):
    """
    Side-by-side comparison of SDEdit results at different noise levels.

    Args:
        results_by_noise_level: dict {noise_level: (regenerated_paths, best_path)}
    """
    n_levels = len(results_by_noise_level)
    fig, axes = plt.subplots(1, n_levels, figsize=figsize)
    if n_levels == 1:
        axes = [axes]

    input_np = to_numpy(input_path) if torch.is_tensor(input_path) else input_path
    start_np = to_numpy(q_pos_start) if torch.is_tensor(q_pos_start) else q_pos_start
    goal_np = to_numpy(q_pos_goal) if torch.is_tensor(q_pos_goal) else q_pos_goal

    for ax, (noise_level, (regen_paths, best_path)) in zip(axes, sorted(results_by_noise_level.items())):
        env.render(ax)

        regen_np = to_numpy(regen_paths) if torch.is_tensor(regen_paths) else regen_paths
        for i in range(min(regen_np.shape[0], 30)):
            ax.plot(regen_np[i, :, 0], regen_np[i, :, 1], color='orange', alpha=0.2, linewidth=0.8)

        ax.plot(input_np[:, 0], input_np[:, 1], color='blue', linewidth=2.0, label='Input')

        if best_path is not None:
            best_np = to_numpy(best_path) if torch.is_tensor(best_path) else best_path
            ax.plot(best_np[:, 0], best_np[:, 1], color='green', linewidth=2.0, label='Best')

        ax.plot(*start_np, 'go', markersize=10, zorder=10)
        ax.plot(*goal_np, 'r*', markersize=12, zorder=10)

        ax.set_aspect('equal')
        ax.set_title(f't_noise = {noise_level}', fontsize=12)
        ax.grid(True, alpha=0.3)

    axes[0].legend(loc='upper left', fontsize=9)
    plt.suptitle('SDEdit: Noise Level Comparison', fontsize=14)
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    return fig, axes


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _render_env_on_ax(ax, env, dim, draw_extra=True):
    """Render environment obstacles using the existing env.render infrastructure."""
    # Draw fixed obstacles always
    if env.obj_fixed_list:
        for obj in env.obj_fixed_list:
            obj.render(ax)
    # Draw extra obstacles only when requested
    if draw_extra and env.obj_extra_list:
        for obj in env.obj_extra_list:
            obj.render(ax, color="red", cmap="Reds")

    ax.set_xlim(env.limits_np[0][0], env.limits_np[1][0])
    ax.set_ylim(env.limits_np[0][1], env.limits_np[1][1])
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")


def _draw_path(ax, path_np, **kwargs):
    """Draw a 2D path on an axis."""
    ax.plot(path_np[:, 0], path_np[:, 1], **kwargs)


def _draw_start_goal(ax, start_np, goal_np, robot, dim):
    """Draw start and goal markers. Uses robot.render when available for consistency."""
    # For 2D point-mass the robot is just a dot, so scatter is equivalent
    ax.scatter(start_np[0], start_np[1], color="blue", marker="X",
               s=10**2.7, zorder=100, label="Start")
    ax.scatter(goal_np[0], goal_np[1], color="red", marker="X",
               s=10**2.7, zorder=100, label="Goal")


def _annotate_obstacle_mod(ax, obstacle_modification):
    """Draw a highlight annotation for an obstacle modification."""
    if obstacle_modification is None:
        return
    mod_type = obstacle_modification.get("type", "")
    if mod_type == "add_sphere":
        center = obstacle_modification["center"]
        radius = obstacle_modification["radius"]
        circle = Circle(center, radius, fill=False, edgecolor="red",
                        linewidth=3, linestyle="--", zorder=90, label="Added Obstacle")
        ax.add_patch(circle)
    elif mod_type == "add_box":
        from matplotlib.patches import Rectangle
        center = np.asarray(obstacle_modification["center"])
        sizes = np.asarray(obstacle_modification["sizes"])
        rect = Rectangle(center - sizes, 2 * sizes[0], 2 * sizes[1],
                          fill=False, edgecolor="red", linewidth=3,
                          linestyle="--", zorder=90, label="Added Obstacle")
        ax.add_patch(rect)
    elif mod_type == "remove_sphere":
        idx = obstacle_modification.get("index", "?")
        ax.annotate(f"Obstacle #{idx} removed", xy=(0.5, 0.97),
                    xycoords="axes fraction", ha="center", va="top",
                    fontsize=11, color="red", fontweight="bold")
