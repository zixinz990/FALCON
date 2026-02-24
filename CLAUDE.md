# CLAUDE.md — FALCON Codebase Guide

## Project Overview

**FALCON** (Learning Force-Adaptive Humanoid Loco-Manipulation) is an RL-based framework for training and deploying humanoid robot locomotion and manipulation policies. It targets the Unitree G1 (29-DoF) and Booster T1 (29-DoF) robots. Published at RSS 2025. Built on [ASAP](https://github.com/LeCAR-Lab/ASAP) and [HumanoidVerse](https://github.com/LeCAR-Lab/HumanoidVerse).

## Repository Structure

```
FALCON/
├── humanoidverse/          # Core RL training environment (IsaacGym/IsaacSim)
│   ├── train_agent.py      # Training entry point (Hydra CLI)
│   ├── eval_agent.py       # Evaluation + ONNX export
│   ├── collect_data.py     # Data collection from trained policies
│   ├── agents/             # RL algorithms (PPO, decoupled PPO)
│   ├── envs/               # Environment implementations
│   ├── simulator/          # Physics simulator backends
│   ├── config/             # Hydra config hierarchy
│   └── utils/              # Helpers, inference, motion lib
├── humanoid_linear_distill/ # Git submodule: linear policy distillation
│   └── src/humanoid_linear_distill/
│       ├── train.py        # Distillation training
│       ├── export_models.py
│       └── utils/          # Networks (KNet), data processing
├── linear-dpg/             # Git submodule: linear DPG optimizer
│   ├── run_solver.py       # Solver entry point
│   └── linear_dpg/
│       ├── solver.py       # DPG solver (JAX)
│       └── humanoid_reward.py  # Reward function (42 terms, JAX)
├── sim2real/               # Deployment: sim2sim (Mujoco) and real robot
│   ├── rl_policy/          # Policy wrappers (dec_loco, loco_manip)
│   ├── sim_env/            # Mujoco simulation
│   └── utils/              # SDK bridges (Unitree, Booster), IK, comms
├── isaac_utils/            # IsaacGym/IsaacSim utility wrappers
├── unitree_sdk2_python/    # Git submodule: Unitree SDK
├── claude_memory/          # Detailed architecture docs (observation layout, pipeline, etc.)
├── setup.py                # Main package install
└── requirements.txt        # Pinned dependencies
```

## Environment Setup

```bash
# Python virtual environment (NOT conda)
source /home/zixin/Dev/FALCON/fcgym/bin/activate

# Install (editable)
pip install -e .
pip install -e isaac_utils
pip install -e humanoid_linear_distill
pip install -e linear-dpg
```

Python >= 3.8. Requires IsaacGym Preview 4 installed separately.

## Key Commands

### Training (Hydra CLI)
```bash
python humanoidverse/train_agent.py \
  +exp=decoupled_locomotion_stand_height_waist_wbc_diff_force_ma_ppo_ma_env \
  +simulator=isaacgym \
  +domain_rand=domain_rand_rl_gym \
  +rewards=dec_loco/reward_dec_loco_stand_height_ma_diff_force \
  +robot=g1/g1_29dof_waist_fakehand \
  +terrain=terrain_locomotion_plane \
  +obs=dec_loco/g1_29dof_obs_diff_force_history_wolinvel_ma \
  num_envs=4096 project_name=g1_29dof_falcon experiment_name=g1_29dof_falcon
```

### Evaluation
```bash
python humanoidverse/eval_agent.py +checkpoint=<path_to_ckpt>
```

### Tests
```bash
# linear-dpg tests (pytest with coverage)
cd linear-dpg && pytest

# humanoid_linear_distill tests
cd humanoid_linear_distill && pytest
```

## Configuration System

Uses **Hydra** with hierarchical YAML configs under `humanoidverse/config/`:
- `algo/` — Algorithm (PPO)
- `env/` — Environment parameters
- `exp/` — Full experiment presets
- `obs/` — Observation specifications
- `rewards/` — Reward functions and scales
- `robot/` — Robot definitions (G1, T1)
- `simulator/` — Backend selection (isaacgym, isaacsim, genesis)
- `terrain/` — Terrain configs
- `domain_rand/` — Domain randomization
- `opt/` — Logging (WandB, TensorBoard)

Override any config value via CLI: `key=value` or `+group=name`.

## Critical Architecture Details

### Observation Buffer Ordering

**Observation fields are concatenated in ALPHABETICAL order**, not YAML config file order. This is due to `sorted(obs_config)` in `legged_robot_base_ma.py:810`.

- **Actor obs**: 115 dims/frame x 5 history = 575 total. History: [oldest, ..., newest].
- **Critic obs**: 128 dims, 1 frame (no history).
- See `claude_memory/observation_buffers.md` for the exact field-by-index layout.

### state_206 Format

Used for distillation: `state_206 = cat(critic_obs[128], extended_rigid_body[78])`.
- `[0:128]`: critic_obs in alphabetical field order
- `[128:206]`: extended rigid body state from `collect_data.py:extract_extended_state()`

### Decoupled Architecture

Separate actor (lower body control, partial obs) and critic (full body, privileged info) networks. The actor uses 5-frame observation history; the critic sees a single frame with extra state (base_lin_vel, base_orientation, end-effector forces).

### Distillation Pipeline

1. **Data collection** (`collect_data.py`) — run trained NN policy, record state_206 + actions + rewards
2. **Autoencoder + linear policy** (`humanoid_linear_distill/train.py`) — train phi/psi encoders + KNet
3. **Linear DPG** (`linear-dpg/run_solver.py`) — optimize K matrix using differentiable reward (JAX)
4. **Evaluation** (`eval_distilled_agent.py`) — compare original vs distilled in IsaacGym

## Git Submodules

Two submodules (update with `git submodule update --init --recursive`):
- `humanoid_linear_distill` → `github.com/zixinz990/humanoid-linear-distill`
- `linear-dpg` → `github.com/zixinz990/linear-dpg`

## Key Dependencies

- **PyTorch** (2.4.1) — NN training and inference
- **JAX** (0.4.13) — linear-dpg solver (autodiff on reward)
- **Hydra** — configuration management
- **IsaacGym** — primary physics simulator
- **ONNX/ONNXRuntime** — model export for deployment
- **WandB** — experiment tracking
- **Mujoco** — sim2sim validation (sim2real/)

## Outputs and Logs

Training artifacts go to `logs/`, `logs_eval/`, `runs/`, `wandb/`, `outputs/`, `results/` — all gitignored. Model checkpoints (`.pt`, `.pkl`) and ONNX files are also gitignored except for specific motion data and sim2real deployment models.
