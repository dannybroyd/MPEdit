"""
SDEdit-style inference script for Motion Planning Diffusion.

Supports two modes:
  - replan: Modify obstacles and regenerate from an existing valid path
  - sketch: Denoise an illegal "sketch" path to make it collision-free

Usage:
  cd scripts/inference
  python inference_sdedit.py
"""

from mpd.utils.patches import numpy_monkey_patch
numpy_monkey_patch()

import time
from functools import partial

import isaacgym

from dotmap import DotMap

import gc
import os
from pprint import pprint

import numpy as np
import torch
from einops._torch_specific import allow_ops_in_compiled_graph

from experiment_launcher import single_experiment_yaml, run_experiment
from mpd.inference.inference import EvaluationSamplesGenerator, GenerativeOptimizationPlanner, render_results
from mpd.metrics.metrics import PlanningMetricsCalculator
from mpd.plotting.sdedit_plots import (
    plot_sdedit_results, plot_noise_level_comparison,
    plot_sdedit_before_after, animate_sdedit_denoising,
)
from mpd.utils.loaders import get_planning_task_and_dataset, load_params_from_yaml, save_to_yaml
from mpd.utils.obstacle_editing import add_sphere_obstacle, add_box_obstacle, remove_fixed_obstacle_by_index
from torch_robotics.torch_utils.seed import fix_random_seed
from torch_robotics.torch_utils.torch_utils import get_torch_device, to_torch, to_numpy

allow_ops_in_compiled_graph()


@single_experiment_yaml
def experiment(
    ########################################################################
    cfg_inference_path: str = './cfgs/config_EnvSimple2D-RobotPointMass2D_sdedit.yaml',
    ########################################################################
    selection_start_goal: str = "validation",
    ########################################################################
    n_start_goal_states: int = 3,
    ########################################################################
    # SDEdit options
    t_noise_levels: str = "7",  # comma-separated noise levels to test (e.g., "3,5,7,9,11")
    sdedit_mode: str = "replan",  # 'replan' or 'sketch'
    ########################################################################
    # Visualization
    render_before_after: bool = True,
    render_denoising_video: bool = True,
    render_joint_space_time_iters: bool = False,
    render_joint_space_env_iters: bool = False,
    render_env_robot_opt_iters: bool = True,
    render_pybullet: bool = False,
    ########################################################################
    device: str = "cuda:0",
    debug: bool = False,
    ########################################################################
    seed: int = 2,
    results_dir: str = "logs_sdedit",
    ########################################################################
    **kwargs,
):
    fix_random_seed(seed)

    device = get_torch_device(device)
    tensor_args = {"device": device, "dtype": torch.float32}

    args_inference = DotMap(load_params_from_yaml(cfg_inference_path))

    # Model selection
    if args_inference.model_selection == "bspline":
        args_inference.model_dir = args_inference.model_dir_ddpm_bspline
    else:
        raise NotImplementedError(f"model_selection={args_inference.model_selection} not supported for SDEdit")
    args_inference.model_dir = os.path.expandvars(args_inference.model_dir)

    os.makedirs(results_dir, exist_ok=True)
    save_to_yaml(args_inference.toDict(), os.path.join(results_dir, "args_inference_sdedit.yaml"))

    print(f"\n{'='*80}")
    print(f"SDEdit Inference")
    print(f"Mode: {sdedit_mode}")
    print(f"Noise levels: {t_noise_levels}")
    print(f"Model: {args_inference.model_dir}")
    print(f"{'='*80}")

    ####################################################################################################################
    # Load dataset, environment, robot and planning task
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

    ####################################################################################################################
    # Generator of evaluation samples (also provides OMPL for generating reference paths)
    evaluation_samples_generator = EvaluationSamplesGenerator(
        planning_task,
        train_subset,
        val_subset,
        selection_start_goal=selection_start_goal,
        planner="RRTConnect",
        tensor_args=tensor_args,
        debug=debug,
        render_pybullet=render_pybullet,
        **args_inference,
    )

    ####################################################################################################################
    # Load the generative model planner
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
        debug=debug,
    )

    ####################################################################################################################
    # Apply obstacle modifications (for replan mode)
    sdedit_cfg = args_inference.get("sdedit", DotMap())
    obstacle_mod = sdedit_cfg.get("obstacle_modification", None)

    if sdedit_mode == "replan" and obstacle_mod is not None:
        mod_type = obstacle_mod.get("type", None)
        if mod_type == "add_sphere":
            center = obstacle_mod["center"]
            radius = obstacle_mod["radius"]
            print(f"\nAdding sphere obstacle: center={center}, radius={radius}")
            add_sphere_obstacle(planning_task.env, center, radius, tensor_args=tensor_args)
            # Rebuild cost guide to pick up new obstacles
            generative_optimization_planner.cost_guide = None
            from mpd.inference.cost_guides import CostGuideManagerParametricTrajectory, NoCostException
            try:
                generative_optimization_planner.cost_guide = CostGuideManagerParametricTrajectory(
                    planning_task, train_subset.dataset, args_inference, tensor_args, debug,
                )
            except NoCostException:
                pass
        elif mod_type == "remove_sphere":
            idx = obstacle_mod.get("index", 0)
            print(f"\nRemoving fixed obstacle at index {idx}")
            remove_fixed_obstacle_by_index(planning_task.env, idx, tensor_args=tensor_args)
        elif mod_type == "add_box":
            center = obstacle_mod["center"]
            sizes = obstacle_mod["sizes"]
            print(f"\nAdding box obstacle: center={center}, sizes={sizes}")
            add_box_obstacle(planning_task.env, center, sizes, tensor_args=tensor_args)

    ####################################################################################################################
    # Parse noise levels
    noise_levels = [int(x.strip()) for x in t_noise_levels.split(",")]
    print(f"Testing noise levels: {noise_levels}")

    ####################################################################################################################
    # Metrics calculator
    planning_metrics_calculator = PlanningMetricsCalculator(planning_task)

    ####################################################################################################################
    # Plan for several start and goal states
    if selection_start_goal == "training":
        idx_sample_l = np.random.choice(np.arange(len(train_subset)), n_start_goal_states)
    else:
        idx_sample_l = np.random.choice(np.arange(len(val_subset)), n_start_goal_states)

    for idx_sg, idx_sample in enumerate(idx_sample_l):
        print(f"\n{'='*80}")
        print(f"PLANNING {idx_sg+1}/{n_start_goal_states}")
        print(f"{'='*80}")

        q_pos_start, q_pos_goal, ee_pose_goal = evaluation_samples_generator.get_data_sample(idx_sg)
        print(f"q_pos_start: {q_pos_start}")
        print(f"q_pos_goal: {q_pos_goal}")

        ############################################################################################################
        # Step 1: Generate a reference path using RRTConnect (before obstacle modification for replan mode)
        print(f"\nGenerating reference path with RRTConnect...")
        q_pos_start_np = to_numpy(q_pos_start, dtype=np.float64)
        q_pos_goal_np = to_numpy(q_pos_goal, dtype=np.float64)
        results_plan_d = evaluation_samples_generator.generate_data_ompl_worker.run(
            1, q_pos_start_np, q_pos_goal_np,
            planner_allowed_time=10.0,
            interpolate_num=args_inference.num_T_pts,
            simplify_path=True,
        )

        if results_plan_d is None or len(results_plan_d) == 0 or results_plan_d[0].get("sol_path") is None:
            print("RRTConnect failed to find a reference path. Skipping this sample.")
            continue

        input_path = np.array(results_plan_d[0]["sol_path"])
        print(f"Reference path: {input_path.shape[0]} waypoints")

        ############################################################################################################
        # Step 2: Run SDEdit at each noise level
        results_by_noise_level = {}
        for t_noise in noise_levels:
            print(f"\n--- SDEdit with t_noise_level={t_noise} ---")
            results_single = DotMap(t_generator=0.0, t_guide=0.0)

            results_single = generative_optimization_planner.plan_trajectory_sdedit(
                q_pos_start, q_pos_goal, ee_pose_goal,
                input_path=input_path,
                t_noise_level=t_noise,
                results_ns=results_single,
                debug=debug,
            )

            print(f"t_inference_total: {results_single.t_inference_total:.3f} sec")
            print(f"t_generator: {results_single.t_generator:.3f} sec")
            print(f"t_guide: {results_single.t_guide:.3f} sec")

            # Compute metrics
            results_single.metrics = planning_metrics_calculator.compute_metrics(results_single)
            print(f"metrics:")
            pprint(results_single.metrics)

            results_by_noise_level[t_noise] = (
                results_single.q_trajs_pos_iter_0,
                results_single.q_trajs_pos_best,
            )

            # Save per-noise-level results
            torch.save(
                results_single,
                os.path.join(results_dir, f"results_sdedit-{idx_sg:03d}-noise{t_noise:02d}.pt"),
                _use_new_zipfile_serialization=True,
            )

        ############################################################################################################
        # Step 3: Visualize

        # 3a. Before / Obstacle Edit / After  (3-panel static figure)
        if render_before_after:
            # Use the default noise level results for the after panel
            default_noise = sdedit_cfg.get("t_noise_level", noise_levels[0])
            default_regen, default_best = results_by_noise_level.get(
                default_noise, results_by_noise_level[noise_levels[0]]
            )
            fig_ba, _ = plot_sdedit_before_after(
                planning_task,
                q_pos_start, q_pos_goal,
                input_path,
                best_regen_path=default_best,
                all_regen_paths=default_regen,
                obstacle_modification=obstacle_mod if sdedit_mode == "replan" else None,
                title=f"SDEdit Re-planning (t_noise={default_noise})" if sdedit_mode == "replan"
                      else f"SDEdit Sketch→Path (t_noise={default_noise})",
                save_path=os.path.join(results_dir, f"sdedit_before_after-{idx_sg:03d}.png"),
            )
            plt.close(fig_ba)

        # 3b. Denoising animation video (shows progressive denoising like the paper's GIF)
        if render_denoising_video:
            default_noise = sdedit_cfg.get("t_noise_level", noise_levels[0])
            # results_single contains the iters from the last noise level tested —
            # reload the default one if needed
            results_for_video = results_single
            if results_for_video.q_trajs_pos_iters is not None:
                animate_sdedit_denoising(
                    planning_task,
                    q_pos_start, q_pos_goal,
                    input_path,
                    trajs_pos_iters=results_for_video.q_trajs_pos_iters,
                    traj_pos_best=results_for_video.q_trajs_pos_best,
                    video_filepath=os.path.join(results_dir, f"sdedit_denoising-{idx_sg:03d}.mp4"),
                    anim_time=args_inference.trajectory_duration,
                )

        # 3c. Single-panel result or noise-level comparison
        if len(noise_levels) == 1:
            regen_paths, best_path = results_by_noise_level[noise_levels[0]]
            fig, ax = plot_sdedit_results(
                planning_task.env,
                q_pos_start, q_pos_goal,
                input_path,
                to_numpy(regen_paths) if regen_paths is not None else np.zeros((0, input_path.shape[0], 2)),
                best_path=to_numpy(best_path) if best_path is not None else None,
                title=f"SDEdit (t_noise={noise_levels[0]}, mode={sdedit_mode})",
                obstacle_modification=obstacle_mod if sdedit_mode == "replan" else None,
                save_path=os.path.join(results_dir, f"sdedit_result-{idx_sg:03d}.png"),
            )
            plt.close(fig)
        else:
            regen_dict = {}
            for t_noise, (regen_paths, best_path) in results_by_noise_level.items():
                regen_dict[t_noise] = (
                    to_numpy(regen_paths) if regen_paths is not None else np.zeros((0, input_path.shape[0], 2)),
                    to_numpy(best_path) if best_path is not None else None,
                )
            fig, axes_cmp = plot_noise_level_comparison(
                planning_task.env,
                q_pos_start, q_pos_goal,
                input_path,
                regen_dict,
                save_path=os.path.join(results_dir, f"sdedit_comparison-{idx_sg:03d}.png"),
            )
            plt.close(fig)

        ############################################################################################################
        # 3d. Standard MPD-style denoising iteration rendering (from existing infrastructure)
        default_noise = sdedit_cfg.get("t_noise_level", noise_levels[0])
        if default_noise in results_by_noise_level:
            render_results(
                args_inference,
                planning_task,
                q_pos_start, q_pos_goal,
                results_single,
                idx_sg,
                results_dir,
                render_joint_space_time_iters=render_joint_space_time_iters,
                render_joint_space_env_iters=render_joint_space_env_iters,
                render_planning_env_robot_opt_iters=render_env_robot_opt_iters,
                debug=debug,
            )

        gc.collect()
        torch.cuda.empty_cache()

    ####################################################################################################################
    evaluation_samples_generator.generate_data_ompl_worker.terminate()
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    # Need matplotlib for saving figures
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt

    run_experiment(experiment)
