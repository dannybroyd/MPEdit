# MPEdit

MPEdit extends MPD (Motion Planning Diffusion) with SDEdit-style trajectory editing.

Instead of generating a path from pure noise, MPEdit starts from an existing path (or sketch), noises it to a chosen diffusion step `t0`, then denoises into a valid trajectory.

Main use cases:
- **Replan**: update an existing path after obstacle changes.
- **Sketch**: turn a rough 2D sketch into a collision-free path.


## Installation Guide

### Prerequisites
- Ubuntu 22.04+ (or compatible Linux)
- Miniconda
- NVIDIA GPU/CUDA setup


### 1) Clone repository
```bash
mkdir -p ~/Projects/MotionPlanningDiffusion
cd ~/Projects/MotionPlanningDiffusion
git clone --recurse-submodules https://github.com/dannybroyd/MPEdit.git MPEdit
cd MPEdit
```


### 2) Install Isaac Gym Preview 4
Download Isaac Gym Preview 4 from NVIDIA and extract it to `deps/isaacgym`:
```bash
mv ~/Downloads/IsaacGym_Preview_4_Package.tar.gz ./deps/
cd deps
tar -xvf IsaacGym_Preview_4_Package.tar.gz
cd ..
```


### 3) Install dependencies
This will take a while.
```bash
bash setup.sh
```


### 4) Download datasets and pretrained models
This will take a while.
Download `data_public.tar.gz` from:
`https://drive.google.com/file/d/1KG5ejn0g0KkDuUK6tPUqfmRYCNoKzK4K/view?usp=drive_link`

Then:
```bash
tar -xvf data_public.tar.gz
ln -s data_public/data_trajectories data_trajectories
ln -s data_public/data_trained_models data_trained_models
```


### 5) Activate environment (every new terminal)
```bash
source set_env_variables.sh
conda activate mpd-splines-public
```


## Inference Examples

Run all inference commands from:
```bash
cd scripts/inference
```


### 1) Sketch (2D, interactive)
```bash
python inference_sdedit_interactive.py --sdedit_mode sketch
```

Notes:
- Default 2D `t_noise_level` is `11` (from config).
- Sketch mode randomizes start/goal by default.
- Sketch mode randomly selects from similar circle-style 2D maps by default:
  `EnvSimple2D`, `EnvGridCircles2D`, `EnvCircle2D`.


### 2) Replan (2D, interactive)
```bash
python inference_sdedit_interactive.py --sdedit_mode replan
```


### 3) Replan (3D Panda)
```bash
python inference_sdedit_interactive.py \
  --cfg_inference_path ./cfgs/config_EnvSpheres3D-RobotPanda_sdedit.yaml \
  --sdedit_mode replan
```

Optional GIF export:
```bash
python inference_sdedit_interactive.py \
  --cfg_inference_path ./cfgs/config_EnvSpheres3D-RobotPanda_sdedit.yaml \
  --sdedit_mode replan \
  --render_env_robot_opt_iters True \
  --render_env_robot_opt_iters_gif True
```

Note: sketch mode is not supported for 3D Panda.


### Batch inference (optional)
```bash
# Replan over multiple t0 values
python inference_sdedit.py --sdedit_mode replan --t_noise_levels "3,5,7,9,11"

# Sketch batch
python inference_sdedit.py --sdedit_mode sketch --t_noise_levels "11"
```


### Inference command options

`inference_sdedit_interactive.py` (interactive):

| Option | Default | Description |
|---|---|---|
| `--cfg_inference_path` | `./cfgs/config_EnvSimple2D-RobotPointMass2D_sdedit.yaml` | Inference YAML path |
| `--sdedit_mode` | `replan` | `replan` or `sketch` |
| `--t_noise_level` | `-1` | `<0` uses config default (`sdedit.t_noise_level`) |
| `--selection_start_goal` | `validation` | Start/goal source split |
| `--n_start_goal_states` | `1` | Number of interactive cases |
| `--randomize_start_goal` | `True` | Randomize start/goal sample selection |
| `--randomize_sketch_env_map` | `True` | Randomize sketch map from candidate envs |
| `--sketch_env_candidates` | `EnvSimple2D,EnvGridCircles2D,EnvCircle2D` | Candidate sketch maps |
| `--randomize_base_obstacle_map` | `False` | Add random extra obstacles (optional) |
| `--random_base_obstacles_count` | `3` | Number of random extra obstacles |
| `--random_base_radius_min_fraction` | `0.04` | Min radius fraction of workspace scale |
| `--random_base_radius_max_fraction` | `0.10` | Max radius fraction of workspace scale |
| `--random_base_map_max_attempts` | `10` | Retry budget for valid random maps |
| `--render_before_after` | `True` | Save summary figure |
| `--render_denoising_video` | `True` | Save denoising MP4 |
| `--render_denoising_gif` | `False` | Save denoising GIF |
| `--render_env_robot_opt_iters` | `False` | Save robot optimization animation |
| `--render_env_robot_opt_iters_gif` | `False` | Save robot optimization GIF |
| `--device` | `cuda:0` | Torch device |
| `--seed` | `-1` | `<0` uses a fresh random seed each run |
| `--results_dir` | `logs_sdedit_interactive` | Output directory |
| `--debug` | `False` | Debug mode |

`inference_sdedit.py` (batch):

| Option | Default | Description |
|---|---|---|
| `--cfg_inference_path` | `./cfgs/config_EnvSimple2D-RobotPointMass2D_sdedit.yaml` | Inference YAML path |
| `--sdedit_mode` | `replan` | `replan` or `sketch` |
| `--t_noise_levels` | `7` | Comma-separated DDIM indices |
| `--selection_start_goal` | `validation` | Start/goal source split |
| `--n_start_goal_states` | `3` | Number of evaluated pairs |
| `--render_before_after` | `True` | Save summary figure |
| `--render_denoising_video` | `True` | Save denoising MP4 |
| `--render_denoising_gif` | `False` | Save denoising GIF |
| `--render_joint_space_time_iters` | `False` | Render joint-time iteration animation |
| `--render_joint_space_env_iters` | `False` | Render joint-env iteration animation |
| `--render_env_robot_opt_iters` | `True` | Render robot optimization animation |
| `--render_env_robot_opt_iters_gif` | `False` | Save robot optimization GIF |
| `--render_env_robot_trajectories` | `False` | Render robot trajectory playback |
| `--render_env_robot_trajectories_gif` | `False` | Save trajectory playback GIF |
| `--render_pybullet` | `False` | Enable pybullet rendering |
| `--device` | `cuda:0` | Torch device |
| `--seed` | `2` | Random seed |
| `--results_dir` | `logs_sdedit` | Output directory |
| `--debug` | `False` | Debug mode |

Tip: run `python <script>.py --help` for generated CLI help.


## Experiments

Run from:
```bash
cd scripts/inference
```


### Run full suites
```bash
# 2D suite
python ../experiments/run_experiment.py --config ../experiments/cfgs/experiment_2d.yaml

# 3D suite
python ../experiments/run_experiment.py --config ../experiments/cfgs/experiment_3d.yaml
```


### Run a single experiment
```bash
python ../experiments/run_experiment.py \
  --config ../experiments/cfgs/experiment_2d.yaml \
  --only exp1
```


### Generate figures/tables
```bash
# 2D outputs
python ../experiments/plot_results.py --results_dir logs_experiments_2d

# 3D outputs
python ../experiments/plot_results.py --results_dir logs_experiments_3d
```


### Experiment command options

`run_experiment.py`:

| Option | Required | Default | Description |
|---|---|---|---|
| `--config` | Yes | — | Experiment config YAML path |
| `--only` | No | `None` | Run one experiment: `exp1`..`exp5` |
| `--results_dir` | No | config value | Override output root directory |

`plot_results.py`:

| Option | Required | Default | Description |
|---|---|---|---|
| `--results_dir` | No | `logs_experiments` | Input results directory |
| `--save_dir` | No | `None` | Output directory for figures/tables |
| `--only` | No | `None` | Plot one figure/table target |


### Experiment set
- `exp1`: `t0` sweep (success vs faithfulness)
- `exp2`: baselines (SDEdit vs Full MPD vs RRTConnect)
- `exp3`: sketch-to-path (2D only)
- `exp4`: inference speed
- `exp5`: diversity

Default outputs:
- 2D: `scripts/inference/logs_experiments_2d/`
- 3D: `scripts/inference/logs_experiments_3d/`


## Overall Structure

```text
MPEdit/
├── mpd/
│   ├── inference/                 # planners + inference pipeline
│   ├── interactive/               # obstacle editors + sketcher
│   ├── plotting/                  # SDEdit plotting/animation
│   ├── utils/                     # obstacle editing, path conversion, scenarios
│   └── models/                    # diffusion model code
├── scripts/
│   ├── inference/                 # interactive and batch inference entry points
│   │   └── cfgs/                  # SDEdit configs (2D/3D)
│   └── experiments/               # paper experiments + plotting
│       └── cfgs/                  # experiment configs
├── data_trained_models -> data_public/data_trained_models
├── data_trajectories  -> data_public/data_trajectories
└── deps/                           # submodules/dependencies
```
