"""
Interactive SDEdit inference with GUI editors.

Launches an obstacle editor or path sketcher before running the SDEdit pipeline.
Automatically detects 2D vs 3D environments and uses the appropriate editor.

Usage:
  cd scripts/inference

  # 2D Interactive re-planning (edit obstacles, then regenerate)
  python inference_sdedit_interactive.py --sdedit_mode replan

  # 2D Interactive sketching (draw a path, then denoise it)
  python inference_sdedit_interactive.py --sdedit_mode sketch

  # 2D Interactive with randomized base map + randomized start/goal
  python inference_sdedit_interactive.py \
    --sdedit_mode sketch \
    --randomize_base_obstacle_map True --randomize_start_goal True

  # 3D Interactive re-planning (add/remove spheres in 3D view)
  python inference_sdedit_interactive.py \\
    --cfg_inference_path ./cfgs/config_EnvSpheres3D-RobotPanda_sdedit.yaml \\
    --sdedit_mode replan

  # Export a Panda robot GIF from optimization iterations
  python inference_sdedit_interactive.py \\
    --cfg_inference_path ./cfgs/config_EnvSpheres3D-RobotPanda_sdedit.yaml \\
    --sdedit_mode replan --render_env_robot_opt_iters True --render_env_robot_opt_iters_gif True
"""

from mpd.utils.patches import numpy_monkey_patch
numpy_monkey_patch()

import time
from copy import deepcopy
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
from mpd.interactive.obstacle_editor_3d import ObstacleEditor3D
from mpd.interactive.path_sketcher import PathSketcher
from mpd.utils.loaders import get_planning_task_and_dataset, load_params_from_yaml, save_to_yaml
from mpd.utils.obstacle_editing import add_sphere_obstacle, add_box_obstacle, remove_fixed_obstacle_by_index
from mpd.utils.scenario_generation import save_fixed_obstacles_state, reset_obstacle_modifications
from mpd.inference.cost_guides import CostGuideManagerParametricTrajectory, NoCostException
from torch_robotics import environments as tr_environments
from torch_robotics.torch_utils.seed import fix_random_seed
from torch_robotics.torch_utils.torch_utils import get_torch_device, to_torch, to_numpy

allow_ops_in_compiled_graph()


def _sync_ompl_worker_with_planning_env(evaluation_samples_generator, planning_env):
    worker = evaluation_samples_generator.generate_data_ompl_worker
    worker.env_tr = deepcopy(planning_env)
    worker.clear_obstacles()
    worker.add_obstacles()


def _check_start_goal_validity(evaluation_samples_generator, q_pos_start, q_pos_goal):
    pbompl = evaluation_samples_generator.generate_data_ompl_worker.pbompl_interface
    start_valid = pbompl.is_state_valid(to_numpy(q_pos_start), check_bounds=True)
    goal_valid = pbompl.is_state_valid(to_numpy(q_pos_goal), check_bounds=True)
    return bool(start_valid), bool(goal_valid)


def _run_reference_rrt(evaluation_samples_generator, q_pos_start_np, q_pos_goal_np, interpolate_num):
    results_plan_d = evaluation_samples_generator.generate_data_ompl_worker.run(
        1, q_pos_start_np, q_pos_goal_np,
        planner_allowed_time=10.0,
        interpolate_num=interpolate_num,
        simplify_path=True,
        max_tries=1,
    )
    if results_plan_d is None or len(results_plan_d) == 0:
        return None
    sol_path = results_plan_d[0].get("sol_path")
    if sol_path is None:
        return None
    return np.asarray(sol_path)


def _apply_random_base_obstacles(
    env,
    tensor_args,
    rng,
    n_obstacles=3,
    radius_min_fraction=0.04,
    radius_max_fraction=0.10,
    keepout_points=None,
):
    keepout_points = [] if keepout_points is None else keepout_points
    n_obstacles = max(0, int(n_obstacles))

    env_min = np.asarray(env.limits_np[0], dtype=np.float64)
    env_max = np.asarray(env.limits_np[1], dtype=np.float64)
    env_range = env_max - env_min
    workspace_scale = float(np.min(env_range))
    radius_min = max(1e-3, float(radius_min_fraction) * workspace_scale)
    radius_max = max(radius_min + 1e-3, float(radius_max_fraction) * workspace_scale)

    modifications = []
    for _ in range(n_obstacles):
        placed = False
        for _ in range(100):
            radius = float(rng.uniform(radius_min, radius_max))
            center = rng.uniform(env_min + radius, env_max - radius)

            too_close = False
            for pt in keepout_points:
                pt_np = np.asarray(pt, dtype=np.float64).reshape(-1)
                if pt_np.shape[0] != env.dim:
                    continue
                if np.linalg.norm(center - pt_np) < (2.0 * radius):
                    too_close = True
                    break
            if too_close:
                continue

            add_sphere_obstacle(env, center.tolist(), radius, tensor_args=tensor_args)
            modifications.append({"type": "add_sphere", "center": center.tolist(), "radius": radius})
            placed = True
            break

        if not placed:
            break

    return modifications


def _select_random_sketch_env_id(rng, candidate_env_ids):
    available = [env_id for env_id in candidate_env_ids if hasattr(tr_environments, env_id)]
    if not available:
        return None
    return str(rng.choice(np.asarray(available, dtype=object)))


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
    t_noise_level: int = -1,
    sdedit_mode: str = "replan",  # 'replan' or 'sketch'
    randomize_start_goal: bool = True,
    randomize_base_obstacle_map: bool = False,
    random_base_obstacles_count: int = 3,
    random_base_radius_min_fraction: float = 0.04,
    random_base_radius_max_fraction: float = 0.10,
    random_base_map_max_attempts: int = 10,
    randomize_sketch_env_map: bool = True,
    sketch_env_candidates: str = "EnvSimple2D,EnvGridCircles2D,EnvCircle2D",
    ########################################################################
    # Visualization
    render_before_after: bool = True,
    render_denoising_video: bool = True,
    render_denoising_gif: bool = False,
    render_env_robot_opt_iters: bool = False,
    render_env_robot_opt_iters_gif: bool = False,
    ########################################################################
    device: str = "cuda:0",
    debug: bool = False,
    ########################################################################
    seed: int = -1,
    results_dir: str = "logs_sdedit_interactive",
    ########################################################################
    **kwargs,
):
    auto_seed = seed < 0
    if auto_seed:
        seed = int(np.random.SeedSequence().generate_state(1)[0])
    fix_random_seed(seed)
    rng = np.random.default_rng(seed + 12345)

    device = get_torch_device(device)
    tensor_args = {"device": device, "dtype": torch.float32}

    args_inference = DotMap(load_params_from_yaml(cfg_inference_path))
    selected_sketch_env_id = None

    cfg_t_noise_level = int(getattr(getattr(args_inference, "sdedit", DotMap()), "t_noise_level", 7))
    if t_noise_level < 0:
        t_noise_level = cfg_t_noise_level

    # Model selection
    if args_inference.model_selection == "bspline":
        args_inference.model_dir = args_inference.model_dir_ddpm_bspline
    else:
        raise NotImplementedError(f"model_selection={args_inference.model_selection} not supported for SDEdit")
    args_inference.model_dir = os.path.expandvars(args_inference.model_dir)

    os.makedirs(results_dir, exist_ok=True)

    ####################################################################################################################
    # Load dataset, environment, robot and planning task
    args_train = DotMap(load_params_from_yaml(os.path.join(args_inference.model_dir, "args.yaml")))
    dataset_subdir = str(args_train.get("dataset_subdir", ""))
    is_pointmass_2d_model = "RobotPointMass2D" in dataset_subdir
    if sdedit_mode == "sketch" and randomize_sketch_env_map and is_pointmass_2d_model:
        candidate_env_ids = [x.strip() for x in sketch_env_candidates.split(",") if x.strip()]
        selected_sketch_env_id = _select_random_sketch_env_id(rng, candidate_env_ids)
        if selected_sketch_env_id is not None:
            args_inference.env_id_replace = selected_sketch_env_id
    elif sdedit_mode == "sketch" and randomize_sketch_env_map and not is_pointmass_2d_model:
        print("Sketch map randomization skipped: current model is not 2D PointMass.")

    print(f"\n{'='*80}")
    print(f"Interactive SDEdit Inference")
    print(f"Mode: {sdedit_mode}")
    print(f"Seed: {seed}{' (auto)' if auto_seed else ''}")
    print(f"Noise level: {t_noise_level}")
    print(f"Random start/goal: {randomize_start_goal}")
    if selected_sketch_env_id is not None:
        print(f"Sketch env map: {selected_sketch_env_id}")
    print(f"Random base obstacle map: {randomize_base_obstacle_map}")
    print(f"{'='*80}")

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

    # Snapshot fixed obstacles once; every sample starts from this baseline
    fixed_obstacles_snapshot = save_fixed_obstacles_state(planning_task.env)
    # Optional style override when sketch mode also uses random extra obstacles.
    planning_task.env.render_extra_as_fixed_color = (sdedit_mode == "sketch" and randomize_base_obstacle_map)

    ####################################################################################################################
    # Pick a start/goal pair
    if selection_start_goal == "training":
        n_candidates = len(train_subset)
    else:
        n_candidates = len(val_subset)

    if randomize_start_goal:
        replace = n_start_goal_states > n_candidates
        idx_sample_l = np.random.choice(np.arange(n_candidates), n_start_goal_states, replace=replace)
    else:
        idx_sample_l = np.arange(n_start_goal_states)

    for idx_sg, idx_sample in enumerate(idx_sample_l):
        print(f"\n{'='*80}")
        print(f"SAMPLE {idx_sg+1}/{n_start_goal_states}")
        print(f"{'='*80}")

        # Start every sample from the original environment
        reset_obstacle_modifications(
            planning_task.env,
            tensor_args,
            fixed_snapshot=fixed_obstacles_snapshot,
        )
        _sync_ompl_worker_with_planning_env(evaluation_samples_generator, planning_task.env)

        sample_idx = int(idx_sample) if randomize_start_goal else int(idx_sg)
        try:
            q_pos_start, q_pos_goal, ee_pose_goal = evaluation_samples_generator.get_data_sample(sample_idx)
        except RuntimeError as e:
            print(f"Could not get a valid start/goal sample for idx={sample_idx}. Skipping. ({e})")
            continue
        print(f"Start: {q_pos_start}")
        print(f"Goal:  {q_pos_goal}")

        ########################################################################################################
        q_pos_start_np = to_numpy(q_pos_start, dtype=np.float64)
        q_pos_goal_np = to_numpy(q_pos_goal, dtype=np.float64)
        rrt_path = None

        if randomize_base_obstacle_map:
            print("\nRandomizing base obstacle map...")
            map_ready = False
            max_attempts = max(1, int(random_base_map_max_attempts))
            for map_try in range(max_attempts):
                reset_obstacle_modifications(
                    planning_task.env,
                    tensor_args,
                    fixed_snapshot=fixed_obstacles_snapshot,
                )
                random_mods = _apply_random_base_obstacles(
                    planning_task.env,
                    tensor_args=tensor_args,
                    rng=rng,
                    n_obstacles=random_base_obstacles_count,
                    radius_min_fraction=random_base_radius_min_fraction,
                    radius_max_fraction=random_base_radius_max_fraction,
                    keepout_points=[q_pos_start_np, q_pos_goal_np],
                )
                _sync_ompl_worker_with_planning_env(evaluation_samples_generator, planning_task.env)

                start_valid, goal_valid = _check_start_goal_validity(
                    evaluation_samples_generator, q_pos_start, q_pos_goal
                )
                if not (start_valid and goal_valid):
                    print(
                        f"  random map attempt {map_try + 1}/{max_attempts} rejected "
                        f"(invalid start/goal)"
                    )
                    continue

                if sdedit_mode == "replan":
                    candidate_rrt_path = _run_reference_rrt(
                        evaluation_samples_generator,
                        q_pos_start_np,
                        q_pos_goal_np,
                        interpolate_num=args_inference.num_T_pts,
                    )
                    if candidate_rrt_path is None:
                        print(
                            f"  random map attempt {map_try + 1}/{max_attempts} rejected "
                            f"(RRT failed)"
                        )
                        continue
                    rrt_path = candidate_rrt_path

                map_ready = True
                print(f"  random map attempt {map_try + 1}/{max_attempts} accepted "
                      f"({len(random_mods)} added obstacles)")
                break

            if not map_ready:
                print("  Failed to find a valid randomized base map. Falling back to original base map.")
                reset_obstacle_modifications(
                    planning_task.env,
                    tensor_args,
                    fixed_snapshot=fixed_obstacles_snapshot,
                )
                _sync_ompl_worker_with_planning_env(evaluation_samples_generator, planning_task.env)

        # Generate a reference path with RRTConnect (used in replan mode; also shown in sketch mode as guide)
        print(f"\nGenerating reference path with RRTConnect...")
        if rrt_path is None:
            rrt_path = _run_reference_rrt(
                evaluation_samples_generator,
                q_pos_start_np,
                q_pos_goal_np,
                interpolate_num=args_inference.num_T_pts,
            )
        if rrt_path is not None:
            print(f"RRT path: {rrt_path.shape[0]} waypoints")
        else:
            print("RRTConnect did not find a path (you can still sketch one).")

        ########################################################################################################
        # MODE: REPLAN — launch obstacle editor, then SDEdit from the RRT path
        if sdedit_mode == "replan":
            if rrt_path is None:
                print("Cannot replan without a reference path. Skipping.")
                continue

            env_before_plot = deepcopy(planning_task.env)
            env_dim = planning_task.env.dim

            if env_dim >= 3:
                print("\n>>> Opening 3D Obstacle Editor — use sliders to place spheres, then click Done <<<")
                editor = ObstacleEditor3D(
                    env=planning_task.env,
                    tensor_args=tensor_args,
                    robot=planning_task.robot,
                    existing_path=rrt_path,
                    q_pos_start=q_pos_start,
                    q_pos_goal=q_pos_goal,
                )
            else:
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
            env_dim = planning_task.env.dim
            if env_dim >= 3:
                print("\n⚠  Sketch mode is not supported for 3D/high-DOF environments.")
                print("   The path sketcher works in task space (2D), but the Panda robot")
                print("   plans in 7D joint space. Use 'replan' mode instead for 3D.")
                continue

            env_before_plot = deepcopy(planning_task.env)
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
                sdedit_mode=sdedit_mode,
                env_before=env_before_plot,
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
                make_gif=render_denoising_gif,
            )

        # Robot animation in the planning environment (Panda-friendly in 3D)
        if render_env_robot_opt_iters:
            render_results(
                args_inference,
                planning_task,
                q_pos_start, q_pos_goal,
                results_single,
                idx_sg,
                results_dir,
                render_planning_env_robot_opt_iters=True,
                make_gif=render_env_robot_opt_iters_gif,
                debug=debug,
            )

        # Single result plot
        fig, ax = plot_sdedit_results(
            planning_task.env,
            q_pos_start, q_pos_goal,
            input_path,
            to_numpy(results_single.q_trajs_pos_iter_0) if results_single.q_trajs_pos_iter_0 is not None
                else np.zeros((0, input_path.shape[0], input_path.shape[1])),
            best_path=to_numpy(results_single.q_trajs_pos_best) if results_single.q_trajs_pos_best is not None
                else None,
            title=f"Interactive SDEdit ({sdedit_mode}, t_noise={t_noise_level})",
            obstacle_modification=obstacle_mod if sdedit_mode == "replan" else None,
            save_path=os.path.join(results_dir, f"interactive_result-{idx_sg:03d}.png"),
            robot=planning_task.robot,
            tensor_args=tensor_args,
        )
        plt.close(fig)

        # Interactive 3D result viewer — let user rotate and inspect
        if planning_task.env.dim >= 3:
            print("\n>>> Opening interactive 3D result viewer (close window to continue) <<<")
            fig_3d, ax_3d = plot_sdedit_results(
                planning_task.env,
                q_pos_start, q_pos_goal,
                input_path,
                to_numpy(results_single.q_trajs_pos_iter_0) if results_single.q_trajs_pos_iter_0 is not None
                    else np.zeros((0, input_path.shape[0], input_path.shape[1])),
                best_path=to_numpy(results_single.q_trajs_pos_best) if results_single.q_trajs_pos_best is not None
                    else None,
                title=f"Interactive Result — {sdedit_mode}, t_noise={t_noise_level} (rotate to inspect)",
                obstacle_modification=obstacle_mod if sdedit_mode == "replan" else None,
                robot=planning_task.robot,
                tensor_args=tensor_args,
            )
            plt.show(block=True)

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
