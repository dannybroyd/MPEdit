# MPEdit: Motion Planning with SDEdit-Style Path Regeneration

This repository extends Motion Planning Diffusion (MPD) with SDEdit-style stochastic editing for adaptive robot path planning. This work is a final project for a Master's Computer Science course.

## Final Project Report

The complete project report (LaTeX format) is available in the root directory:

- **Main report**: `report.tex` - 6-page LaTeX document with comprehensive experimental evaluation
- **Bibliography**: `references.bib` - All citations
- **Compilation instructions**: `REPORT_README.md` - Detailed instructions for compiling the report

### Quick Compilation

```bash
# Using the provided script
bash compile_report.sh

# Or using Make
make          # Full compilation with bibliography
make quick    # Quick single-pass compilation
make view     # Open the PDF
make help     # See all options
```

The report includes:
- Abstract, introduction, and background on MPD and SDEdit
- Detailed methodology for SDEdit-style motion planning
- Five comprehensive experiments with results and analysis
- Appendix with sketch mode examples and demonstrations
- Placeholders for all experimental figures (or actual figures if experiments have been run)

---

# Installation

Pre-requisites:
- Ubuntu 22.04 (maybe works with newer versions)
- [miniconda](https://docs.conda.io/projects/miniconda/en/latest/index.html)

Clone this repository
```bash
mkdir -p ~/Projects/MotionPlanningDiffusion/
cd ~/Projects/MotionPlanningDiffusion/
git clone --recurse-submodules git@github.com:joaoamcarvalho/mpd-splines-public.git mpd-splines-public
cd mpd-splines-public
```

Download [IsaacGym Preview 4](https://developer.nvidia.com/isaac-gym) and extract it under `deps/isaacgym`
```bash
mv ~/Downloads/IsaacGym_Preview_4_Package.tar.gz ~/Projects/MotionPlanningDiffusion/mpd-splines-public/deps/
cd ~/Projects/MotionPlanningDiffusion/mpd-splines-public/deps
tar -xvf IsaacGym_Preview_4_Package.tar.gz
```

Run the bash setup script to install everything (it can take a while).
```bash
bash setup.sh
```

Make sure to set environment variables and activate the conda environment before running any scripts.
```bash
source set_env_variables.sh
conda activate mpd-splines-public
```

---
## Download the datasets and pre-trained models

Download https://drive.google.com/file/d/1KG5ejn0g0KkDuUK6tPUqfmRYCNoKzK4K/view?usp=drive_link

```bash
tar -xvf data_public.tar.gz
ln -s data_public/data_trajectories data_trajectories
ln -s data_public/data_trained_models data_trained_models
```


---
## SDEdit-Style Path Regeneration

This extension adds **SDEdit-style path editing** to MPD — instead of generating trajectories from pure noise, you can start from an existing path, add noise to an intermediate diffusion timestep, and denoise to produce new collision-free trajectories. This is useful for two scenarios:

1. **Re-planning**: You have a valid path, then the obstacle map changes slightly (add/remove an obstacle). Rather than planning from scratch, you noise the old path and denoise under the updated cost guide.
2. **Sketch-to-path**: You have an approximate or illegal "sketch" of a desired path. You noise it and denoise to produce a legal, collision-free version that preserves the sketch's characteristics.

### Quick Start

```bash
source set_env_variables.sh
conda activate mpd-splines-public
cd scripts/inference
python inference_sdedit.py
```

To sweep multiple noise levels:
```bash
python inference_sdedit.py --t_noise_levels "3,5,7,9,11"
```

**Interactive mode** — GUI editors for obstacle editing and path sketching:
```bash
# Re-plan: click to add/remove obstacles, then auto-regenerate the path
python inference_sdedit_interactive.py --sdedit_mode replan

# Sketch: draw a freehand path with the mouse, then denoise it
python inference_sdedit_interactive.py --sdedit_mode sketch
```

### New Files Reference

#### `scripts/inference/inference_sdedit_interactive.py` — Interactive SDEdit with GUI Editors

Launches a matplotlib GUI window before running the SDEdit pipeline. In **replan** mode, opens the Obstacle Editor; in **sketch** mode, opens the Path Sketcher.

**Key arguments**:

| Argument | Type | Default | Description |
|---|---|---|---|
| `sdedit_mode` | str | `"replan"` | `"replan"` (obstacle editor → regenerate) or `"sketch"` (path sketcher → denoise) |
| `t_noise_level` | int | `7` | Single DDIM step index for the noise level |
| `n_start_goal_states` | int | `1` | Number of start/goal pairs (GUI opens once per pair) |
| `render_before_after` | bool | `True` | Generate 3-panel Before/Edit/After figure |
| `render_denoising_video` | bool | `True` | Generate denoising animation video |

**Workflow — Replan mode**:
1. Environment and model load automatically
2. RRTConnect generates a reference path (shown in blue dashes)
3. **Obstacle Editor GUI opens** → click to add/remove obstacles → click "Done"
4. SDEdit runs: noise the old path → denoise with updated cost guide
5. Results saved to `logs_sdedit_interactive/`

**Workflow — Sketch mode**:
1. Environment and model load automatically
2. **Path Sketcher GUI opens** → draw a path from start to goal → click "Done"
3. SDEdit runs: noise the sketch → denoise to make it collision-free
4. Results saved to `logs_sdedit_interactive/`

---

#### `mpd/interactive/obstacle_editor.py` — Interactive Obstacle Editor

Matplotlib-based GUI for adding and removing obstacles on the 2D environment map.

**Controls**:

| Action | Effect |
|---|---|
| **Left-click** on map | Add obstacle (sphere or box) at click position |
| **Right-click** on map | Remove the nearest obstacle (extra first, then fixed) |
| **Size slider** | Adjust radius (sphere) or half-extent (box) before placing |
| **Radio buttons** | Toggle between "sphere" and "box" obstacle types |
| **U key** | Undo last action |
| **Clear All** button | Remove all modifications |
| **Done ✓** / Enter | Accept modifications and close |

**Programmatic usage**:
```python
from mpd.interactive.obstacle_editor import ObstacleEditor

editor = ObstacleEditor(
    env=planning_task.env,
    tensor_args=tensor_args,
    existing_path=rrt_path,           # optional reference path (shown as blue dashes)
    q_pos_start=q_pos_start,          # optional start marker
    q_pos_goal=q_pos_goal,            # optional goal marker
    initial_radius=0.08,              # default sphere radius
)
modifications = editor.run()  # blocks until Done
# modifications = [{"type": "add_sphere", "center": [0.2, 0.3], "radius": 0.08}, ...]
```

**Returns**: list of modification dicts. Each dict has `"type"` (`"add_sphere"`, `"add_box"`, `"remove_sphere"`) plus relevant parameters (`"center"`, `"radius"`, `"sizes"`, `"index"`).

---

#### `mpd/interactive/path_sketcher.py` — Interactive Path Sketcher

Matplotlib-based GUI for drawing freehand paths on the 2D environment map.

**Controls**:

| Action | Effect |
|---|---|
| **Click + drag** | Draw a freehand path |
| **Release mouse** | Auto-smooth and resample to N waypoints |
| **Right-click** | Clear the sketch and start over |
| **Clear Sketch** button | Clear the sketch |
| **Done ✓** / Enter | Accept the path and close |

The sketched path is automatically:
- Snapped to start/goal endpoints
- Smoothed with a B-spline fit
- Resampled to `n_waypoints` evenly-spaced points

**Programmatic usage**:
```python
from mpd.interactive.path_sketcher import PathSketcher

sketcher = PathSketcher(
    env=planning_task.env,
    q_pos_start=q_pos_start,
    q_pos_goal=q_pos_goal,
    n_waypoints=64,                   # must match model's num_T_pts
)
path = sketcher.run()  # blocks until Done
# path: (64, 2) numpy array, or None if cancelled
```

**Returns**: `(n_waypoints, 2)` numpy array of the smoothed sketch, or `None` if no path was drawn.

---

#### `scripts/inference/inference_sdedit.py` — Main SDEdit Inference Script (batch mode)

The primary entry point for running SDEdit experiments. Supports both re-planning and sketch modes.

**Key arguments** (passed via CLI or YAML config):

| Argument | Type | Default | Description |
|---|---|---|---|
| `cfg_inference_path` | str | `./cfgs/config_..._sdedit.yaml` | Path to the SDEdit YAML configuration |
| `t_noise_levels` | str | `"7"` | Comma-separated DDIM step indices to test (e.g., `"3,5,7,9,11"`) |
| `sdedit_mode` | str | `"replan"` | `"replan"` (modify obstacles + regenerate) or `"sketch"` (denoise illegal path) |
| `n_start_goal_states` | int | `3` | Number of start/goal pairs to evaluate |
| `selection_start_goal` | str | `"validation"` | Source for start/goal states (`"training"` or `"validation"`) |
| `render_before_after` | bool | `True` | Generate 3-panel Before / Obstacle Edit / After figure (`.png`) |
| `render_denoising_video` | bool | `True` | Generate denoising animation video (`.mp4`) showing progressive SDEdit refinement |
| `render_env_robot_opt_iters` | bool | `True` | Use existing MPD rendering for denoising iteration frames |
| `results_dir` | str | `"logs_sdedit"` | Directory for output results, plots, and videos |

**Pipeline**:
1. Loads model, environment, and dataset (same setup as standard `inference.py`)
2. Optionally modifies the obstacle map (for re-planning mode)
3. Generates a reference path using RRTConnect
4. Runs SDEdit at each specified noise level: noises the path → denoises with cost guidance
5. Saves results (`.pt`), renders comparison plots (`.png`), and denoising videos (`.mp4`)

**Output files per sample** (in `results_dir/`):
| File | Generated by |
|---|---|
| `sdedit_before_after-{idx}.png` | 3-panel Before/Edit/After figure (`render_before_after`) |
| `sdedit_denoising-{idx}.mp4` | Denoising animation video (`render_denoising_video`) |
| `sdedit_result-{idx}.png` | Single-panel result (when 1 noise level) |
| `sdedit_comparison-{idx}.png` | Multi-noise-level comparison (when >1 noise levels) |
| `results_sdedit-{idx}-noise{t}.pt` | Serialized results per noise level |

**Example — Re-planning with a new obstacle**:
```bash
python inference_sdedit.py \
  --sdedit_mode replan \
  --t_noise_levels "5,7,9" \
  --n_start_goal_states 5
```

**Example — Sketch mode (denoise an illegal path)**:
```bash
python inference_sdedit.py --sdedit_mode sketch --t_noise_levels "7"
```

---

#### `scripts/inference/cfgs/config_EnvSimple2D-RobotPointMass2D_sdedit.yaml` — SDEdit Configuration

YAML configuration file extending the standard 2D PointMass config with SDEdit-specific parameters.

**SDEdit-specific section**:
```yaml
sdedit:
  t_noise_level: 7          # Default DDIM step to noise to (0=full noise, 14=barely noised)
  mode: 'replan'             # 'replan' or 'sketch'
  obstacle_modification:     # Only used in replan mode
    type: 'add_sphere'       # 'add_sphere', 'remove_sphere', or 'add_box'
    center: [0.0, 0.0]       # Center of the new obstacle
    radius: 0.15             # Radius (for spheres)
```

**Key tuning parameters**:
- `ddim.ddim_eta`: Controls stochasticity during denoising. Lower values (e.g., `0.3`) keep outputs closer to the input path. Higher values (e.g., `0.5`–`1.0`) produce more diversity.
- `ddim.ddim_sampling_timesteps`: Total DDIM steps (default `15`). The `t_noise_level` is an index into this schedule.
- `planner_alg`: Should be `'mpd'` for SDEdit to leverage cost guidance for collision avoidance with new obstacles.

---

#### `scripts/inference/cfgs/config_EnvSpheres3D-RobotPanda_sdedit.yaml` — 3D SDEdit Configuration

YAML configuration for running SDEdit on the **EnvSpheres3D-RobotPanda** environment (7-DOF Panda arm navigating among spherical obstacles in 3D space).

> **Note:** Only **replan** mode is supported for 3D/high-DOF environments. Sketch mode requires drawing in joint space, which is not practical for 7-DOF robots.

**SDEdit section** (same structure as 2D, but with 3D coordinates):
```yaml
sdedit:
  t_noise_level: 7
  mode: 'replan'
  obstacle_modification:
    type: 'add_sphere'
    center: [0.0, 0.0, 0.5]   # 3D center
    radius: 0.15
```

**Running 3D SDEdit** (batch mode):
```bash
cd scripts/inference

# Batch replan — uses obstacle from config
python inference_sdedit.py \
  --cfg_inference_path ./cfgs/config_EnvSpheres3D-RobotPanda_sdedit.yaml \
  --sdedit_mode replan
```

**Running 3D SDEdit** (interactive mode):
```bash
# Interactive replan — opens 3D obstacle editor
python inference_sdedit_interactive.py \
  --cfg_inference_path ./cfgs/config_EnvSpheres3D-RobotPanda_sdedit.yaml \
  --sdedit_mode replan
```

---

#### `mpd/interactive/obstacle_editor_3d.py` — Interactive 3D Obstacle Editor

A matplotlib-based 3D interactive editor for placing/removing spherical obstacles in 3D environments. Automatically used when the environment has `dim >= 3`.

**Controls**:

| Input | Action |
|---|---|
| **Left-click** on 3D plot | Add a sphere at the clicked (X, Y) position and the current Z-slider value |
| **Right-click** | Remove the nearest added obstacle (projected distance) |
| **Z pos slider** | Set the Z coordinate for the next placed obstacle |
| **Radius slider** | Set the radius for the next placed obstacle |
| **U key** | Undo the last add/remove |
| **Clear All** button | Remove all added obstacles |
| **Done ✓** button / **Enter** | Accept modifications and close |

A translucent horizontal "ghost plane" is rendered at the current Z level to help visualize placement depth.

**Usage**:
```python
from mpd.interactive.obstacle_editor_3d import ObstacleEditor3D

editor = ObstacleEditor3D(
    env=planning_task.env,
    tensor_args=tensor_args,
    robot=planning_task.robot,
)
modifications = editor.run()   # blocks until Done
# modifications: [{"type": "add_sphere", "center": [x, y, z], "radius": r}, ...]
```

**Returns**: list of modification dicts (same format as the 2D editor), or empty list if no changes.

---

#### 3D vs 2D — Automatic Environment Detection

The interactive inference script (`inference_sdedit_interactive.py`) automatically detects the environment dimensionality via `planning_task.env.dim`:

- **2D** (`dim == 2`): Uses the standard `ObstacleEditor` and `PathSketcher`
- **3D** (`dim >= 3`): Uses `ObstacleEditor3D` for replan mode; sketch mode is disabled with a warning

The plotting functions in `sdedit_plots.py` also auto-detect dimensionality and render 3D axes (with `projection="3d"`) when appropriate.

Converts a raw waypoint path into normalized B-spline control points that the diffusion model can process.

**Function**: `path_to_normalized_control_points(path, parametric_trajectory, dataset, tensor_args=None)`

| Parameter | Type | Description |
|---|---|---|
| `path` | ndarray or Tensor `(N, state_dim)` | Waypoint path to convert |
| `parametric_trajectory` | `ParametricTrajectoryBspline` | From `planning_task.parametric_trajectory` |
| `dataset` | `TrajectoryDatasetBspline` | Provides normalization statistics |
| `tensor_args` | dict | `{"device": ..., "dtype": ...}` |

**Returns**: `(n_learnable_control_points, state_dim)` normalized tensor in `[-1, 1]`.

**Usage**:
```python
from mpd.utils.path_conversion import path_to_normalized_control_points

# path is (N, 2) numpy array of waypoints
cps_normalized = path_to_normalized_control_points(
    path,
    planning_task.parametric_trajectory,
    dataset,
    tensor_args={"device": "cuda:0", "dtype": torch.float32},
)
```

---

#### `mpd/utils/obstacle_editing.py` — Dynamic Obstacle Modification

Utilities for adding/removing obstacles from the environment at runtime, enabling re-planning scenarios without reloading.

**Functions**:

| Function | Description |
|---|---|
| `add_sphere_obstacle(env, center, radius, tensor_args)` | Add a sphere to the environment and recompute SDF. Returns the `ObjectField` for later removal. |
| `add_box_obstacle(env, center, sizes, tensor_args)` | Add a box obstacle. `sizes` are half-extents. |
| `remove_extra_obstacles(env)` | Remove all dynamically added obstacles and recompute SDF. |
| `remove_fixed_obstacle_by_index(env, obstacle_idx, tensor_args)` | Remove a fixed obstacle by its index within the `MultiSphereField`. Use with caution. |

**Usage**:
```python
from mpd.utils.obstacle_editing import add_sphere_obstacle, remove_extra_obstacles

# Add a new obstacle
obj = add_sphere_obstacle(planning_task.env, center=[0.3, -0.2], radius=0.1, tensor_args=tensor_args)

# ... run SDEdit planning ...

# Clean up
remove_extra_obstacles(planning_task.env)
```

**Important**: After modifying obstacles, you must rebuild the cost guide so it sees the updated SDF:
```python
from mpd.inference.cost_guides import CostGuideManagerParametricTrajectory
planner.cost_guide = CostGuideManagerParametricTrajectory(planning_task, dataset, args_inference, tensor_args)
```

---

#### `mpd/plotting/sdedit_plots.py` — SDEdit Visualization

Plotting utilities for visualizing SDEdit results. **All functions auto-detect 2D vs 3D** environments and render appropriately (flat 2D axes or `projection="3d"` axes with wireframe sphere annotations).

**Functions**:

| Function | Description |
|---|---|
| `plot_sdedit_results(env, q_pos_start, q_pos_goal, input_path, regenerated_paths, ...)` | Single plot showing input path (blue), regenerated paths (orange), best path (green), obstacles, and optional added obstacle annotation. |
| `plot_noise_level_comparison(env, q_pos_start, q_pos_goal, input_path, results_by_noise_level, ...)` | Side-by-side comparison of SDEdit outputs at different noise levels. |
| `plot_sdedit_before_after(planning_task, q_pos_start, q_pos_goal, input_path, best_regen_path, ...)` | **3-panel figure** — Before (original obstacles + valid path), Obstacle Edit (modified map), After (modified obstacles + regenerated path). Uses `env.render(ax)` and `planning_task` infrastructure. |
| `animate_sdedit_denoising(planning_task, q_pos_start, q_pos_goal, input_path, trajs_pos_iters, ...)` | **MP4 video** of the SDEdit denoising process — shows progressive trajectory refinement frame-by-frame with collision coloring (valid=green, colliding=red), input path overlay (dashed blue), and start/goal markers. |

All functions accept a `save_path` (or `video_filepath` for animations) argument to save output to disk.

**Usage — 3-panel Before/After figure**:
```python
from mpd.plotting.sdedit_plots import plot_sdedit_before_after

fig, axes = plot_sdedit_before_after(
    planning_task,
    q_pos_start, q_pos_goal,
    input_path,                        # (N, 2) original waypoints
    best_regen_path=best_path,         # (N, 2) best collision-free path
    all_regen_paths=all_paths,         # (n_samples, N, 2) all SDEdit outputs (optional)
    obstacle_modification={"type": "add_sphere", "center": [0, 0], "radius": 0.15},
    title="Re-planning after obstacle added",
    save_path="results/before_after.png",
)
```

**Usage — Denoising animation video**:
```python
from mpd.plotting.sdedit_plots import animate_sdedit_denoising

animate_sdedit_denoising(
    planning_task,
    q_pos_start, q_pos_goal,
    input_path,                        # (N, 2) original waypoints (shown as overlay)
    trajs_pos_iters=q_trajs_pos_iters, # list of (n_samples, N, 2) at each denoising step
    traj_pos_best=best_path,           # (N, 2) final best path (shown in last frame)
    video_filepath="results/sdedit_denoising.mp4",
    anim_time=5.0,                     # seconds for the animation
)
```

**Usage — Single-panel result**:
```python
from mpd.plotting.sdedit_plots import plot_sdedit_results

fig, ax = plot_sdedit_results(
    planning_task.env,
    q_pos_start, q_pos_goal,
    input_path,                    # (N, 2) original waypoints
    regenerated_paths,             # (n_samples, N, 2) SDEdit outputs
    best_path=best_path,           # (N, 2) best collision-free path
    title="Re-planning after obstacle added",
    obstacle_modification={"type": "add_sphere", "center": [0, 0], "radius": 0.15},
    save_path="results/sdedit_example.png",
)
```

---

### SDEdit Core Methods (for developers)

The following methods were added to `GaussianDiffusionModel` in `mpd/models/diffusion_models/diffusion_model_base.py`:

| Method | Description |
|---|---|
| `ddim_sdedit_sample_loop(x_start, t_noise_level, ...)` | DDIM SDEdit: forward-noises `x_start` to the diffusion timestep corresponding to DDIM step `t_noise_level`, then denoises through the remaining DDIM steps with full guidance support. |
| `p_sdedit_sample_loop(x_start, t_noise_level, ...)` | DDPM SDEdit: forward-noises to step `t_noise_level`, then runs the standard DDPM reverse loop from there. |
| `conditional_sample_sdedit(x_start, t_noise_level, ...)` | Dispatcher that routes to DDIM or DDPM SDEdit based on `method` argument. |
| `run_sdedit_inference(x_start, t_noise_level, ...)` | High-level wrapper: handles batching `x_start` for `n_samples`, repeating context/hard_conds, and rearranging output chains. |

The `GenerativeOptimizationPlanner` in `mpd/inference/inference.py` gained:

| Method | Description |
|---|---|
| `plan_trajectory_sdedit(q_pos_start, q_pos_goal, EE_pose_goal, input_path, t_noise_level, ...)` | Full SDEdit planning pipeline: converts `input_path` waypoints to normalized control points, runs SDEdit inference with cost guidance, post-processes to trajectories, filters valid results, and selects the best. |


---

## Paper Experiments

The experiment infrastructure for the MPEdit paper is in `scripts/experiments/`. It runs all 5 experiments described in the experimental plan, saves structured results to disk, and generates publication-quality figures and tables.

### Running Experiments

```bash
source set_env_variables.sh
conda activate mpd-splines-public
cd scripts/inference   # must be here so base config paths resolve

# Run all experiments for 2D environment
python ../experiments/run_experiment.py --config ../experiments/cfgs/experiment_2d.yaml

# Run all experiments for 3D environment
python ../experiments/run_experiment.py --config ../experiments/cfgs/experiment_3d.yaml

# Run a single experiment (exp1, exp2, exp3, exp4, or exp5)
python ../experiments/run_experiment.py --config ../experiments/cfgs/experiment_2d.yaml --only exp1

# Custom results directory
python ../experiments/run_experiment.py --config ../experiments/cfgs/experiment_2d.yaml --results_dir my_results
```

### Experiments

| # | Name | Description |
|---|---|---|
| **exp1** | t₀ Sweep | Sweep noise level 1–14, measure success rate vs path faithfulness (Fréchet distance). Produces the "money plot" tradeoff curve. |
| **exp2** | Baselines | Compare SDEdit re-planning vs Full MPD (from pure noise) vs RRTConnect across Easy/Medium/Hard/Removal scenarios. Reports success rate, Fréchet distance, L2 distance, inference time, path length, and smoothness. |
| **exp3** | Sketch-to-Path | 2D only. Corrupt valid paths with Gaussian noise (σ = 0.05–0.3) to create synthetic sketches, then run SDEdit at multiple t₀ values. |
| **exp4** | Inference Speed | Wall-clock timing of SDEdit at each t₀ vs Full MPD and RRT baselines. Reports speedup factors. |
| **exp5** | Diversity | Measure output diversity (pairwise Fréchet/L2 distance among 100 trajectories) as a function of t₀. |

### Generating Figures and Tables

After running experiments:

```bash
cd scripts/inference

# Generate all paper figures from saved results
python ../experiments/plot_results.py --results_dir logs_experiments

# Generate a single figure
python ../experiments/plot_results.py --results_dir logs_experiments --only fig2
```

**Output figures** (saved to `logs_experiments/figures/`):

| File | Description |
|---|---|
| `fig2_t0_sweep.pdf` | Dual Y-axis tradeoff: success rate vs Fréchet distance |
| `fig_baselines_comparison.pdf` | Grouped bar chart: SDEdit vs Full MPD vs RRT |
| `table1_baselines.tex` | LaTeX table for the paper |
| `fig4_sketch_heatmap.pdf` | Success/faithfulness heatmap over (σ, t₀) |
| `fig5_speed.pdf` | Inference time vs t₀ with baseline horizontal lines |
| `fig6_diversity.pdf` | Pairwise diversity vs t₀ |

### Experiment Configuration

Experiment parameters are in YAML files under `scripts/experiments/cfgs/`:

- `experiment_2d.yaml` — EnvSimple2D + RobotPointMass2D (all 5 experiments)
- `experiment_3d.yaml` — EnvSpheres3D + RobotPanda (exp3 disabled — sketch mode impractical for 7-DOF)

Key parameters you may want to tune:

| Parameter | Location | Description |
|---|---|---|
| `common.n_start_goal_pairs` | experiment config | Number of test scenarios per experiment (50 default) |
| `common.n_trajectory_samples` | experiment config | SDEdit samples per run (100 default) |
| `exp2_baselines.sdedit_noise_level` | experiment config | Optimal t₀ from Experiment 1 (update after running exp1) |
| `exp3_sketch.noise_sigmas` | experiment config | Gaussian noise levels for synthetic sketches |
| `exp4_speed.n_timing_runs` | experiment config | Repetitions for stable timing (50 default) |

### New Files Reference

| File | Description |
|---|---|
| `scripts/experiments/run_experiment.py` | Main experiment runner (all 5 experiments) |
| `scripts/experiments/plot_results.py` | Results aggregation and paper figure generation |
| `scripts/experiments/cfgs/experiment_2d.yaml` | 2D experiment configuration |
| `scripts/experiments/cfgs/experiment_3d.yaml` | 3D experiment configuration |
| `mpd/metrics/sdedit_metrics.py` | SDEdit-specific metrics: discrete Fréchet distance, mean L2 distance, pairwise diversity |
| `mpd/utils/scenario_generation.py` | Obstacle placement by difficulty, synthetic sketch generation |

---
## Inference with pre-trained models

The configuration files under [scripts/inference/cfgs](scripts/inference/cfgs) contain the hyperparameters for inference.\
Inside the file `scripts/inference/inference.py` you can change the `cfg_inference_path` parameter to try models trained for different environments.

```bash
cd scripts/inference
python inference.py
```


---
# Training the prior models (from scratch)


## Data generation

Generating data takes a long time, so we recommend [downloading the dataset](#download-the-datasets-and-pre-trained-models).
But if anyway you want to generate your own data, you can do it with the scripts in the `scripts/generate_data` folder.

Go to the `scripts/generate_data` folder.

The base script is
```bash
python generate_trajectories.py
```

To generate multiple datasets in parallel, adapt the `launch_generate_trajectories.py` script.
```bash
python launch_generate_trajectories.py
```

After generating the data, run the post-processing file to combine all data into a hdf5 file.
Then you can double the dataset by flipping the trajectory paths.
```bash
python post_process_trajectories.py --help
python flip_solution_paths.py  (change the PATH_TO_DATASETS variable)
```

To visualize the generated data, use the `visualize_trajectories.py` script.
```bash
python visualize_trajectories.py
```

---
## Training the models

The training scripts are in the `scripts/train` folder.

The base script is
```bash
cd scripts/train
python train.py
```

To train multiple models in parallel, use the `launch_train_*` files.


---
## Citation

If you use our work or code base, please cite our articles:
```latex
@article{carvalho2025motion,
  title={Motion planning diffusion: Learning and adapting robot motion planning with diffusion models},
  author={Carvalho, Jo{\~a}o and Le, An T and Kicki, Piotr and Koert, Dorothea and Peters, Jan},
  journal={IEEE Transactions on Robotics},
  year={2025},
  publisher={IEEE}
}

@inproceedings{carvalho2023motion,
  title={Motion planning diffusion: Learning and planning of robot motions with diffusion models},
  author={Carvalho, Jo{\~a}o and Le, An T and Baierl, Mark and Koert, Dorothea and Peters, Jan},
  booktitle={IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year={2023}
}
```


---