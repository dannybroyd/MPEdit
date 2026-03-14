"""
MPEdit Paper Experiment Runner.

Runs all 5 experiments described in mpedit_paper_metrics.md:
  1. t₀ Sweep — success rate vs faithfulness tradeoff
  2. Re-planning baselines — SDEdit vs Full MPD vs RRT
  3. Sketch-to-path with synthetic sketches (2D only)
  4. Inference speed timing
  5. Diversity analysis

Usage:
  cd scripts/inference   # must be in this directory for config paths to resolve
  python ../experiments/run_experiment.py --config ../experiments/cfgs/experiment_2d.yaml
  python ../experiments/run_experiment.py --config ../experiments/cfgs/experiment_3d.yaml

  # Run a single experiment:
  python ../experiments/run_experiment.py --config ../experiments/cfgs/experiment_2d.yaml --only exp1

Results are saved as .pt files under the results directory for later aggregation.
"""

from mpd.utils.patches import numpy_monkey_patch
numpy_monkey_patch()

import argparse
import gc
import json
import os
import sys
import time
import traceback
from copy import deepcopy
from functools import partial
from pprint import pprint

import isaacgym  # noqa: F401 — must be imported before torch

import numpy as np
import torch
from dotmap import DotMap
from einops._torch_specific import allow_ops_in_compiled_graph

from mpd.inference.inference import EvaluationSamplesGenerator, GenerativeOptimizationPlanner
from mpd.inference.cost_guides import CostGuideManagerParametricTrajectory, NoCostException
from mpd.metrics.metrics import PlanningMetricsCalculator
from mpd.metrics.sdedit_metrics import compute_sdedit_metrics
from mpd.utils.loaders import get_planning_task_and_dataset, load_params_from_yaml, save_to_yaml
from mpd.utils.scenario_generation import (
    generate_obstacle_scenario,
    apply_obstacle_modifications,
    reset_obstacle_modifications,
    save_fixed_obstacles_state,
    generate_synthetic_sketch,
)
from torch_robotics.torch_utils.seed import fix_random_seed
from torch_robotics.torch_utils.torch_timer import TimerCUDA
from torch_robotics.torch_utils.torch_utils import get_torch_device, to_torch, to_numpy

allow_ops_in_compiled_graph()


# ─────────────────────────────────────────────────────────────────────────────
# Setup helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_experiment_config(config_path):
    """Load experiment config YAML."""
    return DotMap(load_params_from_yaml(config_path))


def setup_planning_infrastructure(exp_cfg, results_dir):
    """
    Load the model, environment, robot, and create the planner + evaluation generator.

    Returns:
        planning_task, train_subset, val_subset,
        evaluation_samples_generator, generative_optimization_planner,
        args_inference, args_train, tensor_args
    """
    device = get_torch_device(exp_cfg.common.device)
    tensor_args = {"device": device, "dtype": torch.float32}

    args_inference = DotMap(load_params_from_yaml(exp_cfg.base_inference_config))

    if args_inference.model_selection == "bspline":
        args_inference.model_dir = args_inference.model_dir_ddpm_bspline
    else:
        raise NotImplementedError(f"model_selection={args_inference.model_selection}")
    args_inference.model_dir = os.path.expandvars(args_inference.model_dir)

    os.makedirs(results_dir, exist_ok=True)

    args_train = DotMap(load_params_from_yaml(os.path.join(args_inference.model_dir, "args.yaml")))
    args_train.update(
        **args_inference,
        gripper=True,
        reload_data=False,
        results_dir=results_dir,
        load_indices=True,
        tensor_args=tensor_args,
    )
    planning_task, train_subset, _, val_subset, _ = get_planning_task_and_dataset(**args_train)

    evaluation_samples_generator = EvaluationSamplesGenerator(
        planning_task,
        train_subset,
        val_subset,
        selection_start_goal=exp_cfg.common.selection_start_goal,
        planner="RRTConnect",
        tensor_args=tensor_args,
        debug=False,
        render_pybullet=False,
        **args_inference,
    )

    generative_optimization_planner = GenerativeOptimizationPlanner(
        planning_task,
        train_subset.dataset,
        args_train,
        args_inference,
        tensor_args,
        sampling_based_planner_fn=partial(
            evaluation_samples_generator.generate_data_ompl_worker.run,
            planner_allowed_time=10.0,
            interpolate_num=args_inference.num_T_pts,
            simplify_path=True,
        ),
        debug=False,
    )

    # Save a snapshot of the fixed obstacles so removal scenarios can be undone
    fixed_obstacles_snapshot = save_fixed_obstacles_state(planning_task.env)

    return (
        planning_task, train_subset, val_subset,
        evaluation_samples_generator, generative_optimization_planner,
        args_inference, args_train, tensor_args,
        fixed_obstacles_snapshot,
    )


def sync_ompl_worker_with_planning_env(evaluation_samples_generator, planning_env):
    """
    Keep the OMPL worker collision world in sync with planning_task.env.

    The OMPL worker stores a deep-copied environment at construction time, so any
    later obstacle edits on planning_task.env must be explicitly mirrored.
    """
    worker = evaluation_samples_generator.generate_data_ompl_worker
    worker.env_tr = deepcopy(planning_env)
    worker.clear_obstacles()
    worker.add_obstacles()


def check_start_goal_validity(evaluation_samples_generator, q_pos_start, q_pos_goal):
    """
    Check whether start/goal states are valid in the OMPL worker's current world.
    """
    pbompl = evaluation_samples_generator.generate_data_ompl_worker.pbompl_interface
    start_valid = pbompl.is_state_valid(to_numpy(q_pos_start), check_bounds=True)
    goal_valid = pbompl.is_state_valid(to_numpy(q_pos_goal), check_bounds=True)
    return start_valid, goal_valid


def get_start_goal_and_reference_path(
    evaluation_samples_generator, idx, args_inference, tensor_args, planning_env,
):
    """
    Get a start/goal pair and generate a reference path with RRTConnect.

    Returns:
        (q_pos_start, q_pos_goal, ee_pose_goal, reference_path) or None if RRT fails
    """
    sync_ompl_worker_with_planning_env(evaluation_samples_generator, planning_env)

    try:
        q_pos_start, q_pos_goal, ee_pose_goal = evaluation_samples_generator.get_data_sample(idx)
    except RuntimeError as e:
        print(f"  Failed to find a valid start/goal sample for idx={idx}: {e}")
        return None

    q_pos_start_np = to_numpy(q_pos_start, dtype=np.float64)
    q_pos_goal_np = to_numpy(q_pos_goal, dtype=np.float64)

    results_plan_d = evaluation_samples_generator.generate_data_ompl_worker.run(
        1, q_pos_start_np, q_pos_goal_np,
        planner_allowed_time=10.0,
        interpolate_num=args_inference.num_T_pts,
        simplify_path=True,
        max_tries=3,
    )

    if results_plan_d is None or len(results_plan_d) == 0 or results_plan_d[0].get("sol_path") is None:
        return None

    reference_path = np.array(results_plan_d[0]["sol_path"])
    return q_pos_start, q_pos_goal, ee_pose_goal, reference_path


def rebuild_cost_guide(planner, planning_task, train_subset, args_inference, tensor_args):
    """Rebuild the cost guide after obstacle modifications."""
    planner.cost_guide = None
    try:
        planner.cost_guide = CostGuideManagerParametricTrajectory(
            planning_task, train_subset.dataset, args_inference, tensor_args, False,
        )
    except NoCostException:
        pass


def run_sdedit_and_collect(
    planner, q_pos_start, q_pos_goal, ee_pose_goal, input_path,
    t_noise_level, n_trajectory_samples=None,
):
    """
    Run SDEdit and return results DotMap.
    """
    results = DotMap(t_generator=0.0, t_guide=0.0)
    results = planner.plan_trajectory_sdedit(
        q_pos_start, q_pos_goal, ee_pose_goal,
        input_path=input_path,
        t_noise_level=t_noise_level,
        n_trajectory_samples=n_trajectory_samples,
        results_ns=results,
        debug=False,
    )
    return results


def run_full_mpd_and_collect(planner, q_pos_start, q_pos_goal, ee_pose_goal, n_trajectory_samples=None):
    """
    Run standard MPD (from pure noise) and return results DotMap.
    """
    results = DotMap(t_generator=0.0, t_guide=0.0)
    results = planner.plan_trajectory(
        q_pos_start, q_pos_goal, ee_pose_goal,
        n_trajectory_samples=n_trajectory_samples,
        results_ns=results,
        debug=False,
    )
    return results


def run_rrt_replan(
    evaluation_samples_generator,
    q_pos_start,
    q_pos_goal,
    args_inference,
    planning_env,
    allowed_time=10.0,
):
    """
    Run RRTConnect replanning and return timing + path.
    """
    sync_ompl_worker_with_planning_env(evaluation_samples_generator, planning_env)

    q_pos_start_np = to_numpy(q_pos_start, dtype=np.float64)
    q_pos_goal_np = to_numpy(q_pos_goal, dtype=np.float64)

    start_valid, goal_valid = check_start_goal_validity(
        evaluation_samples_generator, q_pos_start_np, q_pos_goal_np
    )
    if not start_valid or not goal_valid:
        print(
            f"Info:    RRTConnect re-plan skipped due invalid start/goal "
            f"(start_valid={start_valid}, goal_valid={goal_valid})"
        )
        return None, 0.0

    t_start = time.perf_counter()
    results_plan_d = evaluation_samples_generator.generate_data_ompl_worker.run(
        1, q_pos_start_np, q_pos_goal_np,
        planner_allowed_time=allowed_time,
        interpolate_num=args_inference.num_T_pts,
        simplify_path=True,
        max_tries=1,
    )
    t_elapsed = time.perf_counter() - t_start

    if results_plan_d is None or len(results_plan_d) == 0 or results_plan_d[0].get("sol_path") is None:
        return None, t_elapsed

    path = np.array(results_plan_d[0]["sol_path"])
    return path, t_elapsed


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 1: t₀ Sweep
# ─────────────────────────────────────────────────────────────────────────────

def run_exp1_t0_sweep(
    exp_cfg, planning_task, train_subset, val_subset,
    eval_gen, planner, args_inference, tensor_args, results_dir,
    fixed_obstacles_snapshot=None,
):
    """
    Sweep t_noise_level from 1 to 14, recording success rate and path similarity
    at each level for each start/goal pair.
    """
    cfg = exp_cfg.exp1_t0_sweep
    if not cfg.get("enabled", False):
        print("Experiment 1 (t₀ sweep) is disabled. Skipping.")
        return

    print(f"\n{'='*80}")
    print("EXPERIMENT 1: t₀ Sweep — The Money Plot")
    print(f"{'='*80}")

    exp_dir = os.path.join(results_dir, cfg.results_subdir)
    os.makedirs(exp_dir, exist_ok=True)

    noise_levels = cfg.noise_levels
    n_pairs = exp_cfg.common.n_start_goal_pairs
    rng = np.random.default_rng(exp_cfg.seed)
    require_rrt_feasible = cfg.get("require_rrt_feasible", planning_task.env.dim >= 3)
    max_obstacle_sampling_attempts = int(
        cfg.get("max_obstacle_sampling_attempts", 20 if planning_task.env.dim >= 3 else 5)
    )
    rrt_feasibility_time = float(cfg.get("rrt_feasibility_time", 8.0 if planning_task.env.dim >= 3 else 5.0))

    all_results = {t: [] for t in noise_levels}

    for idx_sg in range(n_pairs):
        print(f"\n--- Pair {idx_sg + 1}/{n_pairs} ---")

        # Get reference path (before obstacle modification)
        sample = get_start_goal_and_reference_path(
            eval_gen, idx_sg, args_inference, tensor_args, planning_task.env
        )
        if sample is None:
            print(f"  RRT failed for pair {idx_sg}. Skipping.")
            continue
        q_pos_start, q_pos_goal, ee_pose_goal, reference_path = sample

        # Apply obstacle modification (optionally enforce RRT feasibility in the modified map)
        sampled_feasible_modification = False
        for attempt in range(max_obstacle_sampling_attempts):
            reset_obstacle_modifications(planning_task.env, tensor_args, fixed_snapshot=fixed_obstacles_snapshot)
            modifications = generate_obstacle_scenario(
                planning_task.env, reference_path, cfg.obstacle_difficulty, rng=rng,
                robot=planning_task.robot, tensor_args=tensor_args,
            )
            apply_obstacle_modifications(planning_task.env, modifications, tensor_args)
            rebuild_cost_guide(planner, planning_task, train_subset, args_inference, tensor_args)

            # Reject scenarios that invalidate the fixed start/goal pair.
            sync_ompl_worker_with_planning_env(eval_gen, planning_task.env)
            start_valid, goal_valid = check_start_goal_validity(
                eval_gen, q_pos_start, q_pos_goal
            )
            if not start_valid or not goal_valid:
                print(
                    f"  Obstacle scenario invalidates start/goal "
                    f"(start_valid={start_valid}, goal_valid={goal_valid}). Resampling..."
                )
                continue

            if not require_rrt_feasible:
                sampled_feasible_modification = True
                break

            rrt_path_check, _ = run_rrt_replan(
                eval_gen, q_pos_start, q_pos_goal, args_inference, planning_task.env,
                allowed_time=rrt_feasibility_time,
            )
            if rrt_path_check is not None:
                sampled_feasible_modification = True
                break

            print(
                f"  Obstacle scenario infeasible for RRT replan "
                f"(attempt {attempt + 1}/{max_obstacle_sampling_attempts}). Resampling..."
            )

        if not sampled_feasible_modification:
            print(
                "  Could not sample a feasible obstacle scenario after "
                f"{max_obstacle_sampling_attempts} attempts. Skipping pair."
            )
            reset_obstacle_modifications(planning_task.env, tensor_args, fixed_snapshot=fixed_obstacles_snapshot)
            rebuild_cost_guide(planner, planning_task, train_subset, args_inference, tensor_args)
            continue

        for t_noise in noise_levels:
            print(f"  t_noise={t_noise}", end=" ", flush=True)
            try:
                results = run_sdedit_and_collect(
                    planner, q_pos_start, q_pos_goal, ee_pose_goal,
                    reference_path, t_noise,
                    n_trajectory_samples=exp_cfg.common.n_trajectory_samples,
                )
                # Compute SDEdit-specific metrics
                sdedit_m = compute_sdedit_metrics(
                    reference_path,
                    to_numpy(results.q_trajs_pos_iter_0),
                    best_path=to_numpy(results.q_trajs_pos_best) if results.q_trajs_pos_best is not None else None,
                    planning_task=planning_task,
                    robot=planning_task.robot,
                    compute_pairwise_diversity=False,
                )
                sdedit_m["t_inference_total"] = results.t_inference_total
                sdedit_m["pair_idx"] = idx_sg
                all_results[t_noise].append(sdedit_m)
                print(f"✓ success={sdedit_m.get('success_rate', '?'):.2f} "
                      f"frechet={sdedit_m.get('frechet_to_input_mean', '?'):.4f}")
            except Exception as e:
                print(f"✗ error: {e}"); traceback.print_exc()

        # Cleanup for next pair
        reset_obstacle_modifications(planning_task.env, tensor_args, fixed_snapshot=fixed_obstacles_snapshot)
        rebuild_cost_guide(planner, planning_task, train_subset, args_inference, tensor_args)
        gc.collect()
        torch.cuda.empty_cache()

    # Save results
    torch.save(all_results, os.path.join(exp_dir, "exp1_results.pt"))
    print(f"\nExperiment 1 results saved to {exp_dir}/exp1_results.pt")

    # Print summary
    print("\n--- Experiment 1 Summary ---")
    print(f"{'t_noise':>8} | {'Success%':>9} | {'Fréchet':>10} | {'L2':>10} | {'Time(s)':>8}")
    print("-" * 55)
    for t_noise in noise_levels:
        if all_results[t_noise]:
            sr = np.mean([r["success_rate"] for r in all_results[t_noise] if r["success_rate"] is not None])
            fr = np.mean([r["frechet_to_input_mean"] for r in all_results[t_noise]])
            l2 = np.mean([r["l2_to_input_mean"] for r in all_results[t_noise]])
            tm = np.mean([r["t_inference_total"] for r in all_results[t_noise]])
            print(f"{t_noise:>8} | {sr:>8.1%} | {fr:>10.4f} | {l2:>10.4f} | {tm:>8.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 2: Re-planning Baselines
# ─────────────────────────────────────────────────────────────────────────────

def run_exp2_baselines(
    exp_cfg, planning_task, train_subset, val_subset,
    eval_gen, planner, args_inference, tensor_args, results_dir,
    fixed_obstacles_snapshot=None,
):
    """
    Compare SDEdit, Full MPD, and RRTConnect re-planning across difficulty scenarios.
    """
    cfg = exp_cfg.exp2_baselines
    if not cfg.get("enabled", False):
        print("Experiment 2 (baselines) is disabled. Skipping.")
        return

    print(f"\n{'='*80}")
    print("EXPERIMENT 2: Re-planning — SDEdit vs Baselines")
    print(f"{'='*80}")

    exp_dir = os.path.join(results_dir, cfg.results_subdir)
    os.makedirs(exp_dir, exist_ok=True)

    scenarios = cfg.scenarios
    n_pairs = exp_cfg.common.n_start_goal_pairs
    rng = np.random.default_rng(exp_cfg.seed + 100)
    max_obstacle_sampling_attempts = int(
        cfg.get("max_obstacle_sampling_attempts", 12 if planning_task.env.dim >= 3 else 3)
    )

    all_results = {scenario: {"sdedit": [], "full_mpd": [], "rrt": []} for scenario in scenarios}

    for scenario in scenarios:
        print(f"\n{'─'*60}")
        print(f"Scenario: {scenario}")
        print(f"{'─'*60}")

        for idx_sg in range(n_pairs):
            print(f"\n  Pair {idx_sg + 1}/{n_pairs}")

            sample = get_start_goal_and_reference_path(
                eval_gen, idx_sg, args_inference, tensor_args, planning_task.env
            )
            if sample is None:
                print(f"    RRT failed for initial path. Skipping.")
                continue
            q_pos_start, q_pos_goal, ee_pose_goal, reference_path = sample

            # Apply obstacle modifications for this scenario
            sampled_valid_modification = False
            for attempt in range(max_obstacle_sampling_attempts):
                reset_obstacle_modifications(
                    planning_task.env, tensor_args, fixed_snapshot=fixed_obstacles_snapshot
                )
                modifications = generate_obstacle_scenario(
                    planning_task.env, reference_path, scenario, rng=rng,
                    robot=planning_task.robot, tensor_args=tensor_args,
                )
                apply_obstacle_modifications(planning_task.env, modifications, tensor_args)
                rebuild_cost_guide(planner, planning_task, train_subset, args_inference, tensor_args)

                # Avoid benchmarking methods on impossible endpoint constraints.
                sync_ompl_worker_with_planning_env(eval_gen, planning_task.env)
                start_valid, goal_valid = check_start_goal_validity(eval_gen, q_pos_start, q_pos_goal)
                if start_valid and goal_valid:
                    sampled_valid_modification = True
                    break

                print(
                    f"    Obstacle scenario invalidates start/goal "
                    f"(start_valid={start_valid}, goal_valid={goal_valid}) "
                    f"[attempt {attempt + 1}/{max_obstacle_sampling_attempts}]. Resampling..."
                )

            if not sampled_valid_modification:
                print(
                    f"    Could not sample a valid obstacle scenario after "
                    f"{max_obstacle_sampling_attempts} attempts. Skipping pair."
                )
                reset_obstacle_modifications(
                    planning_task.env, tensor_args, fixed_snapshot=fixed_obstacles_snapshot
                )
                rebuild_cost_guide(planner, planning_task, train_subset, args_inference, tensor_args)
                continue

            # ── Method 1: SDEdit ──
            print(f"    SDEdit (t={cfg.sdedit_noise_level})...", end=" ", flush=True)
            try:
                results_sdedit = run_sdedit_and_collect(
                    planner, q_pos_start, q_pos_goal, ee_pose_goal,
                    reference_path, cfg.sdedit_noise_level,
                    n_trajectory_samples=exp_cfg.common.n_trajectory_samples,
                )
                m_sdedit = compute_sdedit_metrics(
                    reference_path,
                    to_numpy(results_sdedit.q_trajs_pos_iter_0),
                    best_path=to_numpy(results_sdedit.q_trajs_pos_best) if results_sdedit.q_trajs_pos_best is not None else None,
                    planning_task=planning_task,
                    robot=planning_task.robot,
                    compute_pairwise_diversity=False,
                )
                m_sdedit["t_inference_total"] = results_sdedit.t_inference_total
                m_sdedit["pair_idx"] = idx_sg
                all_results[scenario]["sdedit"].append(m_sdedit)
                print(f"✓")
            except Exception as e:
                print(f"✗ {e}"); traceback.print_exc()

            # ── Method 2: Full MPD (from pure noise) ──
            print(f"    Full MPD...", end=" ", flush=True)
            try:
                results_mpd = run_full_mpd_and_collect(
                    planner, q_pos_start, q_pos_goal, ee_pose_goal,
                    n_trajectory_samples=exp_cfg.common.n_trajectory_samples,
                )
                m_mpd = compute_sdedit_metrics(
                    reference_path,
                    to_numpy(results_mpd.q_trajs_pos_iter_0),
                    best_path=to_numpy(results_mpd.q_trajs_pos_best) if results_mpd.q_trajs_pos_best is not None else None,
                    planning_task=planning_task,
                    robot=planning_task.robot,
                    compute_pairwise_diversity=False,
                )
                m_mpd["t_inference_total"] = results_mpd.t_inference_total
                m_mpd["pair_idx"] = idx_sg
                all_results[scenario]["full_mpd"].append(m_mpd)
                print(f"✓")
            except Exception as e:
                print(f"✗ {e}"); traceback.print_exc()

            # ── Method 3: RRTConnect re-plan ──
            print(f"    RRT re-plan...", end=" ", flush=True)
            try:
                rrt_path, rrt_time = run_rrt_replan(
                    eval_gen, q_pos_start, q_pos_goal, args_inference,
                    planning_task.env,
                    allowed_time=cfg.rrt_allowed_time,
                )
                if rrt_path is not None:
                    m_rrt = compute_sdedit_metrics(
                        reference_path,
                        rrt_path[np.newaxis, ...],  # (1, N, D) for batch interface
                        best_path=rrt_path,
                        planning_task=planning_task,
                        robot=planning_task.robot,
                        compute_pairwise_diversity=False,
                    )
                    m_rrt["t_inference_total"] = rrt_time
                    m_rrt["success_rate"] = 1.0  # RRT only returns valid paths
                else:
                    m_rrt = {
                        "success_rate": 0.0,
                        "t_inference_total": rrt_time,
                        "frechet_to_input_mean": None,
                        "l2_to_input_mean": None,
                    }
                m_rrt["pair_idx"] = idx_sg
                all_results[scenario]["rrt"].append(m_rrt)
                print(f"✓")
            except Exception as e:
                print(f"✗ {e}"); traceback.print_exc()

            # Cleanup
            reset_obstacle_modifications(planning_task.env, tensor_args, fixed_snapshot=fixed_obstacles_snapshot)
            rebuild_cost_guide(planner, planning_task, train_subset, args_inference, tensor_args)
            gc.collect()
            torch.cuda.empty_cache()

    # Save results
    torch.save(all_results, os.path.join(exp_dir, "exp2_results.pt"))
    print(f"\nExperiment 2 results saved to {exp_dir}/exp2_results.pt")

    # Print summary table
    print("\n--- Experiment 2 Summary ---")
    for scenario in scenarios:
        print(f"\n  Scenario: {scenario}")
        print(f"  {'Method':>12} | {'Success%':>9} | {'Fréchet':>10} | {'L2':>10} | {'Time(s)':>8}")
        print(f"  {'-'*60}")
        for method in ["sdedit", "full_mpd", "rrt"]:
            entries = all_results[scenario][method]
            if entries:
                sr = np.mean([r.get("success_rate", 0) for r in entries if r.get("success_rate") is not None])
                fr_vals = [r["frechet_to_input_mean"] for r in entries if r.get("frechet_to_input_mean") is not None]
                fr = np.mean(fr_vals) if fr_vals else float("nan")
                l2_vals = [r["l2_to_input_mean"] for r in entries if r.get("l2_to_input_mean") is not None]
                l2 = np.mean(l2_vals) if l2_vals else float("nan")
                tm = np.mean([r["t_inference_total"] for r in entries])
                print(f"  {method:>12} | {sr:>8.1%} | {fr:>10.4f} | {l2:>10.4f} | {tm:>8.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 3: Sketch-to-Path (2D only)
# ─────────────────────────────────────────────────────────────────────────────

def run_exp3_sketch(
    exp_cfg, planning_task, train_subset, val_subset,
    eval_gen, planner, args_inference, tensor_args, results_dir,
    fixed_obstacles_snapshot=None,
):
    """
    Synthetic sketch experiment: corrupt valid paths with Gaussian noise,
    then run SDEdit at multiple t₀ values.
    """
    cfg = exp_cfg.exp3_sketch
    if not cfg.get("enabled", False):
        print("Experiment 3 (sketch-to-path) is disabled. Skipping.")
        return

    if planning_task.env.dim > 2:
        print("Experiment 3 (sketch-to-path) is 2D only. Skipping for 3D environment.")
        return

    print(f"\n{'='*80}")
    print("EXPERIMENT 3: Sketch-to-Path (Synthetic Sketches)")
    print(f"{'='*80}")

    exp_dir = os.path.join(results_dir, cfg.results_subdir)
    os.makedirs(exp_dir, exist_ok=True)

    noise_sigmas = cfg.noise_sigmas
    noise_levels = cfg.noise_levels
    n_pairs = exp_cfg.common.n_start_goal_pairs
    n_sketches = cfg.n_sketches_per_sigma
    rng = np.random.default_rng(exp_cfg.seed + 200)

    # Results: sigma -> t_noise -> list of metric dicts
    all_results = {
        sigma: {t: [] for t in noise_levels}
        for sigma in noise_sigmas
    }

    for idx_sg in range(n_pairs):
        print(f"\n--- Pair {idx_sg + 1}/{n_pairs} ---")

        sample = get_start_goal_and_reference_path(
            eval_gen, idx_sg, args_inference, tensor_args, planning_task.env
        )
        if sample is None:
            print(f"  RRT failed. Skipping.")
            continue
        q_pos_start, q_pos_goal, ee_pose_goal, reference_path = sample

        for sigma in noise_sigmas:
            for sketch_idx in range(n_sketches):
                sketch = generate_synthetic_sketch(reference_path, sigma, rng=rng, fix_endpoints=True)

                for t_noise in noise_levels:
                    print(f"  σ={sigma}, sketch={sketch_idx}, t={t_noise}", end=" ", flush=True)
                    try:
                        results = run_sdedit_and_collect(
                            planner, q_pos_start, q_pos_goal, ee_pose_goal,
                            sketch, t_noise,
                            n_trajectory_samples=exp_cfg.common.n_trajectory_samples,
                        )
                        m = compute_sdedit_metrics(
                            sketch,
                            to_numpy(results.q_trajs_pos_iter_0),
                            best_path=to_numpy(results.q_trajs_pos_best) if results.q_trajs_pos_best is not None else None,
                            planning_task=planning_task,
                            robot=planning_task.robot,
                            compute_pairwise_diversity=False,
                        )
                        # Also compute distance to the ORIGINAL valid path
                        from mpd.metrics.sdedit_metrics import discrete_frechet_distance, mean_l2_distance
                        if results.q_trajs_pos_best is not None:
                            m["best_frechet_to_original"] = discrete_frechet_distance(
                                reference_path, to_numpy(results.q_trajs_pos_best)
                            )
                            m["best_l2_to_original"] = mean_l2_distance(
                                reference_path, to_numpy(results.q_trajs_pos_best)
                            )

                        m["t_inference_total"] = results.t_inference_total
                        m["pair_idx"] = idx_sg
                        m["sketch_idx"] = sketch_idx
                        m["noise_sigma"] = sigma
                        all_results[sigma][t_noise].append(m)
                        print(f"✓")
                    except Exception as e:
                        print(f"✗ {e}"); traceback.print_exc()

        gc.collect()
        torch.cuda.empty_cache()

    # Save results
    torch.save(all_results, os.path.join(exp_dir, "exp3_results.pt"))
    print(f"\nExperiment 3 results saved to {exp_dir}/exp3_results.pt")

    # Print summary
    print("\n--- Experiment 3 Summary ---")
    print(f"{'σ':>6} | {'t_noise':>8} | {'Success%':>9} | {'Sketch Fréchet':>14}")
    print("-" * 50)
    for sigma in noise_sigmas:
        for t_noise in noise_levels:
            entries = all_results[sigma][t_noise]
            if entries:
                sr = np.mean([r.get("success_rate", 0) for r in entries if r.get("success_rate") is not None])
                fr = np.mean([r["frechet_to_input_mean"] for r in entries])
                print(f"{sigma:>6.2f} | {t_noise:>8} | {sr:>8.1%} | {fr:>14.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 4: Inference Speed
# ─────────────────────────────────────────────────────────────────────────────

def run_exp4_speed(
    exp_cfg, planning_task, train_subset, val_subset,
    eval_gen, planner, args_inference, tensor_args, results_dir,
    fixed_obstacles_snapshot=None,
):
    """
    Time SDEdit at various t₀ levels, plus Full MPD and RRT baselines.
    """
    cfg = exp_cfg.exp4_speed
    if not cfg.get("enabled", False):
        print("Experiment 4 (speed) is disabled. Skipping.")
        return

    print(f"\n{'='*80}")
    print("EXPERIMENT 4: Inference Speed")
    print(f"{'='*80}")

    exp_dir = os.path.join(results_dir, cfg.results_subdir)
    os.makedirs(exp_dir, exist_ok=True)

    noise_levels = cfg.noise_levels
    n_runs = cfg.n_timing_runs

    # Get a single start/goal pair for consistent timing
    sample = get_start_goal_and_reference_path(
        eval_gen, 0, args_inference, tensor_args, planning_task.env
    )
    if sample is None:
        print("Failed to get reference path for timing. Aborting Experiment 4.")
        return
    q_pos_start, q_pos_goal, ee_pose_goal, reference_path = sample

    # Apply a medium obstacle to have a realistic scenario
    modifications = generate_obstacle_scenario(
        planning_task.env, reference_path, "medium",
        rng=np.random.default_rng(exp_cfg.seed + 300),
        robot=planning_task.robot, tensor_args=tensor_args,
    )
    apply_obstacle_modifications(planning_task.env, modifications, tensor_args)
    rebuild_cost_guide(planner, planning_task, train_subset, args_inference, tensor_args)

    timing_results = {"sdedit": {}, "full_mpd": [], "rrt": []}

    # Warmup
    print("Warmup run...", flush=True)
    _ = run_sdedit_and_collect(
        planner, q_pos_start, q_pos_goal, ee_pose_goal,
        reference_path, 7, n_trajectory_samples=10,
    )
    torch.cuda.synchronize()

    # Time SDEdit at each noise level
    for t_noise in noise_levels:
        times = []
        print(f"  SDEdit t={t_noise}: ", end="", flush=True)
        for i in range(n_runs):
            torch.cuda.synchronize()
            t_start = time.perf_counter()
            _ = run_sdedit_and_collect(
                planner, q_pos_start, q_pos_goal, ee_pose_goal,
                reference_path, t_noise,
                n_trajectory_samples=exp_cfg.common.n_trajectory_samples,
            )
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t_start)
        timing_results["sdedit"][t_noise] = times
        print(f"mean={np.mean(times):.4f}s ± {np.std(times):.4f}s")

    # Time Full MPD
    if cfg.get("time_full_mpd", True):
        print(f"  Full MPD: ", end="", flush=True)
        for i in range(n_runs):
            torch.cuda.synchronize()
            t_start = time.perf_counter()
            _ = run_full_mpd_and_collect(
                planner, q_pos_start, q_pos_goal, ee_pose_goal,
                n_trajectory_samples=exp_cfg.common.n_trajectory_samples,
            )
            torch.cuda.synchronize()
            timing_results["full_mpd"].append(time.perf_counter() - t_start)
        print(f"mean={np.mean(timing_results['full_mpd']):.4f}s ± {np.std(timing_results['full_mpd']):.4f}s")

    # Time RRT
    if cfg.get("time_rrt", True):
        print(f"  RRT: ", end="", flush=True)
        for i in range(n_runs):
            _, t_rrt = run_rrt_replan(
                eval_gen, q_pos_start, q_pos_goal, args_inference,
                planning_task.env,
                allowed_time=cfg.rrt_allowed_time,
            )
            timing_results["rrt"].append(t_rrt)
        print(f"mean={np.mean(timing_results['rrt']):.4f}s ± {np.std(timing_results['rrt']):.4f}s")

    # Cleanup
    reset_obstacle_modifications(planning_task.env, tensor_args, fixed_snapshot=fixed_obstacles_snapshot)
    rebuild_cost_guide(planner, planning_task, train_subset, args_inference, tensor_args)

    # Save results
    torch.save(timing_results, os.path.join(exp_dir, "exp4_results.pt"))
    print(f"\nExperiment 4 results saved to {exp_dir}/exp4_results.pt")

    # Print summary
    print("\n--- Experiment 4 Summary ---")
    print(f"{'Method':>12} | {'Mean(s)':>10} | {'Std(s)':>10} | {'Speedup':>8}")
    print("-" * 50)
    full_mpd_mean = np.mean(timing_results["full_mpd"]) if timing_results["full_mpd"] else float("nan")
    for t_noise in noise_levels:
        t_mean = np.mean(timing_results["sdedit"][t_noise])
        t_std = np.std(timing_results["sdedit"][t_noise])
        speedup = full_mpd_mean / t_mean if t_mean > 0 else float("inf")
        print(f"  SDEdit t={t_noise:>2} | {t_mean:>10.4f} | {t_std:>10.4f} | {speedup:>7.2f}x")
    if timing_results["full_mpd"]:
        print(f"  {'Full MPD':>12} | {full_mpd_mean:>10.4f} | {np.std(timing_results['full_mpd']):>10.4f} | {'1.00x':>8}")
    if timing_results["rrt"]:
        rrt_mean = np.mean(timing_results["rrt"])
        print(f"  {'RRT':>12} | {rrt_mean:>10.4f} | {np.std(timing_results['rrt']):>10.4f} | {'-':>8}")


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 5: Diversity Analysis
# ─────────────────────────────────────────────────────────────────────────────

def run_exp5_diversity(
    exp_cfg, planning_task, train_subset, val_subset,
    eval_gen, planner, args_inference, tensor_args, results_dir,
    fixed_obstacles_snapshot=None,
):
    """
    Analyze output diversity (pairwise distances) as a function of t₀.
    """
    cfg = exp_cfg.exp5_diversity
    if not cfg.get("enabled", False):
        print("Experiment 5 (diversity) is disabled. Skipping.")
        return

    print(f"\n{'='*80}")
    print("EXPERIMENT 5: Diversity Analysis")
    print(f"{'='*80}")

    exp_dir = os.path.join(results_dir, cfg.results_subdir)
    os.makedirs(exp_dir, exist_ok=True)

    noise_levels = cfg.noise_levels
    n_pairs = cfg.get("n_start_goal_pairs", 10)
    n_samples = cfg.get("n_trajectory_samples", 100)
    rng = np.random.default_rng(exp_cfg.seed + 400)

    all_results = {t: [] for t in noise_levels}

    for idx_sg in range(n_pairs):
        print(f"\n--- Pair {idx_sg + 1}/{n_pairs} ---")

        sample = get_start_goal_and_reference_path(
            eval_gen, idx_sg, args_inference, tensor_args, planning_task.env
        )
        if sample is None:
            print(f"  RRT failed. Skipping.")
            continue
        q_pos_start, q_pos_goal, ee_pose_goal, reference_path = sample

        # Apply obstacle modification
        reset_obstacle_modifications(planning_task.env, tensor_args, fixed_snapshot=fixed_obstacles_snapshot)
        modifications = generate_obstacle_scenario(
            planning_task.env, reference_path, cfg.get("obstacle_difficulty", "medium"), rng=rng,
            robot=planning_task.robot, tensor_args=tensor_args,
        )
        apply_obstacle_modifications(planning_task.env, modifications, tensor_args)
        rebuild_cost_guide(planner, planning_task, train_subset, args_inference, tensor_args)

        for t_noise in noise_levels:
            print(f"  t_noise={t_noise}", end=" ", flush=True)
            try:
                results = run_sdedit_and_collect(
                    planner, q_pos_start, q_pos_goal, ee_pose_goal,
                    reference_path, t_noise,
                    n_trajectory_samples=n_samples,
                )
                regen_np = to_numpy(results.q_trajs_pos_iter_0)
                m = compute_sdedit_metrics(
                    reference_path, regen_np,
                    planning_task=planning_task,
                    robot=planning_task.robot,
                )
                m["pair_idx"] = idx_sg
                all_results[t_noise].append(m)
                print(f"✓ diversity_l2={m.get('pairwise_diversity_l2', '?'):.4f}")
            except Exception as e:
                print(f"✗ {e}"); traceback.print_exc()

        # Cleanup
        reset_obstacle_modifications(planning_task.env, tensor_args, fixed_snapshot=fixed_obstacles_snapshot)
        rebuild_cost_guide(planner, planning_task, train_subset, args_inference, tensor_args)
        gc.collect()
        torch.cuda.empty_cache()

    # Save results
    torch.save(all_results, os.path.join(exp_dir, "exp5_results.pt"))
    print(f"\nExperiment 5 results saved to {exp_dir}/exp5_results.pt")

    # Print summary
    print("\n--- Experiment 5 Summary ---")
    print(f"{'t_noise':>8} | {'Diversity (L2)':>15} | {'Diversity (Fréchet)':>20}")
    print("-" * 50)
    for t_noise in noise_levels:
        entries = all_results[t_noise]
        if entries:
            div_l2 = np.mean([r["pairwise_diversity_l2"] for r in entries])
            div_fr = np.mean([r["pairwise_diversity_frechet"] for r in entries])
            print(f"{t_noise:>8} | {div_l2:>15.4f} | {div_fr:>20.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MPEdit Paper Experiments")
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to experiment config YAML (e.g., ../experiments/cfgs/experiment_2d.yaml)",
    )
    parser.add_argument(
        "--only", type=str, default=None,
        help="Run only a specific experiment: exp1, exp2, exp3, exp4, exp5",
    )
    parser.add_argument(
        "--results_dir", type=str, default=None,
        help=(
            "Base directory for experiment results. "
            "If omitted, uses config key `results_dir` when present, "
            "otherwise falls back to 'logs_experiments'."
        ),
    )
    args = parser.parse_args()

    exp_cfg = load_experiment_config(args.config)
    fix_random_seed(exp_cfg.seed)

    results_dir = args.results_dir or exp_cfg.get("results_dir", "logs_experiments")
    os.makedirs(results_dir, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"MPEdit Paper Experiments")
    print(f"Config: {args.config}")
    print(f"Results: {results_dir}")
    print(f"{'='*80}")

    # Save experiment config for reproducibility
    save_to_yaml(exp_cfg.toDict(), os.path.join(results_dir, "experiment_config.yaml"))

    # Setup planning infrastructure (shared across all experiments)
    (
        planning_task, train_subset, val_subset,
        eval_gen, planner,
        args_inference, args_train, tensor_args,
        fixed_obstacles_snapshot,
    ) = setup_planning_infrastructure(exp_cfg, results_dir)

    shared_args = (
        exp_cfg, planning_task, train_subset, val_subset,
        eval_gen, planner, args_inference, tensor_args, results_dir,
        fixed_obstacles_snapshot,
    )

    # Run experiments
    experiments = {
        "exp1": run_exp1_t0_sweep,
        "exp2": run_exp2_baselines,
        "exp3": run_exp3_sketch,
        "exp4": run_exp4_speed,
        "exp5": run_exp5_diversity,
    }

    if args.only:
        if args.only in experiments:
            experiments[args.only](*shared_args)
        else:
            print(f"Unknown experiment: {args.only}. Available: {list(experiments.keys())}")
            sys.exit(1)
    else:
        for name, fn in experiments.items():
            fn(*shared_args)

    # Cleanup
    eval_gen.generate_data_ompl_worker.terminate()
    gc.collect()
    torch.cuda.empty_cache()

    print(f"\n{'='*80}")
    print("All experiments complete!")
    print(f"Results saved to: {results_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
