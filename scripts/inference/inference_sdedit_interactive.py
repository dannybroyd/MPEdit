"""
Interactive SDEdit inference with GUI editors.

Launches an obstacle editor or path sketcher before running the SDEdit pipeline.

Usage:
  cd scripts/inference

  # Interactive re-planning (edit obstacles, then regenerate)
  python inference_sdedit_interactive.py --sdedit_mode replan

  # Interactive sketching (draw a path, then denoise it)
  python inference_sdedit_interactive.py --sdedit_mode sketch
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
from mpd.interactive.obstacle_editor import ObstacleEditor
from mpd.interactive.path_sketcher import PathSketcher
from mpd.utils.loaders import get_planning_task_and_dataset, load_params_from_yaml, save_to_yaml
from mpd.utils.obstacle_editing import add_sphere_obstacle, add_box_obstacle, remove_fixed_obstacle_by_index
from mpd.inference.cost_guides import CostGuideManagerParametricTrajectory, NoCostException
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
    n_start_goal_states: int = 1,
    ########################################################################
    # SDEdit options
    t_noise_level: int = 7,
    sdedit_mode: str = "replan",  # 'replan' or 'sketch'
    ########################################################################
    # Visualization
    render_before_after: bool = True,
    render_denoising_video: bool = True,
    ########################################################################
    device: str = "cuda:0",
    debug: bool = False,
    ########################################################################
    seed: int = 2,
    results_dir: str = "logs_sdedit_interactive",
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

    print(f"\n{'='*80}")
    print(f"Interactive SDEdit Inference")
    print(f"Mode: {sdedit_mode}")
    print(f"Noise level: {t_noise_level}")
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
    # Generator of evaluation samples (also provides OMPL)
    evaluation_samples_generator = EvaluationSamplesGenerator(
        planning_task,
        train_subset,
        val_subset,
        selection_start_goal=selection_start_goal,
        planner="RRTConnect",
        tensor_args=tensor_args,
        debug=debug,
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
    # Pick a start/goal pair
    if selection_start_goal == "training":
        idx_sample_l = np.random.choice(np.arange(len(train_subset)), n_start_goal_states)
    else:
        idx_sample_l = np.random.choice(np.arange(len(val_subset)), n_start_goal_states)

    for idx_sg, idx_sample in enumerate(idx_sample_l):
        print(f"\n{'='*80}")
        print(f"SAMPLE {idx_sg+1}/{n_start_goal_states}")
        print(f"{'='*80}")

        q_pos_start, q_pos_goal, ee_pose_goal = evaluation_samples_generator.get_data_sample(idx_sg)
        print(f"Start: {q_pos_start}")
        print(f"Goal:  {q_pos_goal}")

        ########################################################################################################
        # Generate a reference path with RRTConnect (used in replan mode; also shows in sketch mode as guide)
        print(f"\nGenerating reference path with RRTConnect...")
        q_pos_start_np = to_numpy(q_pos_start, dtype=np.float64)
        q_pos_goal_np = to_numpy(q_pos_goal, dtype=np.float64)
        results_plan_d = evaluation_samples_generator.generate_data_ompl_worker.run(
            1, q_pos_start_np, q_pos_goal_np,
            planner_allowed_time=10.0,
            interpolate_num=args_inference.num_T_pts,
            simplify_path=True,
        )

        rrt_path = None
        if results_plan_d is not None and len(results_plan_d) > 0 and results_plan_d[0].get("sol_path") is not None:
            rrt_path = np.array(results_plan_d[0]["sol_path"])
            print(f"RRT path: {rrt_path.shape[0]} waypoints")
        else:
            print("RRTConnect did not find a path (you can still sketch one).")

        ########################################################################################################
        # MODE: REPLAN — launch obstacle editor, then SDEdit from the RRT path
        if sdedit_mode == "replan":
            if rrt_path is None:
                print("Cannot replan without a reference path. Skipping.")
                continue

            print("\n>>> Opening Obstacle Editor — modify the obstacle map, then click Done <<<")
            editor = ObstacleEditor(
                env=planning_task.env,
                tensor_args=tensor_args,
                existing_path=rrt_path,
                q_pos_start=q_pos_start,
                q_pos_goal=q_pos_goal,
            )
            modifications = editor.run()

            if not modifications:
                print("No modifications made. Skipping SDEdit.")
                continue

            print(f"\nObstacle modifications ({len(modifications)}):")
            for m in modifications:
                pprint(m)

            # Rebuild cost guide to pick up the modified environment
            generative_optimization_planner.cost_guide = None
            try:
                generative_optimization_planner.cost_guide = CostGuideManagerParametricTrajectory(
                    planning_task, train_subset.dataset, args_inference, tensor_args, debug,
                )
            except NoCostException:
                pass

            input_path = rrt_path
            obstacle_mod = modifications[0] if len(modifications) == 1 else {
                "type": "multiple", "modifications": modifications
            }

        ########################################################################################################
        # MODE: SKETCH — launch path sketcher, then SDEdit from the sketch
        elif sdedit_mode == "sketch":
            print("\n>>> Opening Path Sketcher — draw a path from start to goal, then click Done <<<")
            sketcher = PathSketcher(
                env=planning_task.env,
                q_pos_start=q_pos_start,
                q_pos_goal=q_pos_goal,
                n_waypoints=args_inference.num_T_pts,
                tensor_args=tensor_args,
            )
            sketch_path = sketcher.run()

            if sketch_path is None or len(sketch_path) < 3:
                print("No path was sketched. Skipping SDEdit.")
                continue

            print(f"Sketch path: {sketch_path.shape[0]} waypoints")
            input_path = sketch_path
            obstacle_mod = None

        else:
            raise ValueError(f"Unknown sdedit_mode: {sdedit_mode}")

        ########################################################################################################
        # Run SDEdit
        print(f"\n--- Running SDEdit with t_noise_level={t_noise_level} ---")
        results_single = DotMap(t_generator=0.0, t_guide=0.0)

        results_single = generative_optimization_planner.plan_trajectory_sdedit(
            q_pos_start, q_pos_goal, ee_pose_goal,
            input_path=input_path,
            t_noise_level=t_noise_level,
            results_ns=results_single,
            debug=debug,
        )

        print(f"t_inference_total: {results_single.t_inference_total:.3f} sec")

        # Metrics
        planning_metrics_calculator = PlanningMetricsCalculator(planning_task)
        results_single.metrics = planning_metrics_calculator.compute_metrics(results_single)
        print("Metrics:")
        pprint(results_single.metrics)

        # Save results
        torch.save(
            results_single,
            os.path.join(results_dir, f"results_interactive-{idx_sg:03d}-noise{t_noise_level:02d}.pt"),
            _use_new_zipfile_serialization=True,
        )

        ########################################################################################################
        # Visualize

        # 3-panel before/after
        if render_before_after:
            fig_ba, _ = plot_sdedit_before_after(
                planning_task,
                q_pos_start, q_pos_goal,
                input_path,
                best_regen_path=results_single.q_trajs_pos_best,
                all_regen_paths=results_single.q_trajs_pos_iter_0,
                obstacle_modification=obstacle_mod if sdedit_mode == "replan" else None,
                title=f"Interactive SDEdit ({sdedit_mode}, t_noise={t_noise_level})",
                save_path=os.path.join(results_dir, f"interactive_before_after-{idx_sg:03d}.png"),
            )
            plt.close(fig_ba)

        # Denoising video
        if render_denoising_video and results_single.q_trajs_pos_iters is not None:
            animate_sdedit_denoising(
                planning_task,
                q_pos_start, q_pos_goal,
                input_path,
                trajs_pos_iters=results_single.q_trajs_pos_iters,
                traj_pos_best=results_single.q_trajs_pos_best,
                video_filepath=os.path.join(results_dir, f"interactive_denoising-{idx_sg:03d}.mp4"),
                anim_time=5.0,
            )

        # Single result plot
        fig, ax = plot_sdedit_results(
            planning_task.env,
            q_pos_start, q_pos_goal,
            input_path,
            to_numpy(results_single.q_trajs_pos_iter_0) if results_single.q_trajs_pos_iter_0 is not None
                else np.zeros((0, input_path.shape[0], 2)),
            best_path=to_numpy(results_single.q_trajs_pos_best) if results_single.q_trajs_pos_best is not None
                else None,
            title=f"Interactive SDEdit ({sdedit_mode}, t_noise={t_noise_level})",
            obstacle_modification=obstacle_mod if sdedit_mode == "replan" else None,
            save_path=os.path.join(results_dir, f"interactive_result-{idx_sg:03d}.png"),
        )
        plt.close(fig)

        gc.collect()
        torch.cuda.empty_cache()

    ####################################################################################################################
    evaluation_samples_generator.generate_data_ompl_worker.terminate()
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    # Use an interactive backend (NOT Agg) since we need GUI windows
    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib import pyplot as plt

    run_experiment(experiment)
