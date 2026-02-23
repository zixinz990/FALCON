# Linear Distillation Pipeline Overview

## Purpose

Distill a neural network locomotion policy (trained in IsaacGym) into a linear control policy of the form `u = K * phi(x)`, where:
- `phi(x)`: Neural network encoder mapping 206-dim state to a latent feature space
- `K`: Linear gain matrix (can be a neural network `KNet` or raw matrix)
- The linear structure enables the linear-DPG solver to optimize K using the full training reward

## Pipeline Stages

### 1. Data Collection (`humanoidverse/collect_data.py`)

Runs the trained neural network policy in IsaacGym and records:
- `observations.npz` — actor_obs (575-dim) and critic_obs (128-dim)
- `actions.npy` — policy outputs (29-dim)
- `state_206.npy` — 206-dim state vector = cat(critic_obs, extended_rigid_body)
- `reset_flags.npy` — which environments failed (for filtering)

Config: Hydra-based, loads training checkpoint config and merges with overrides.

### 2. Autoencoder + Linear Policy Training (`humanoid_linear_distill/`)

**Data loading** (`utils/data_processor.py`):
- Loads state_206.npy (falls back to actor_obs for backward compatibility)
- Filters out failed environments using reset_flags
- Skips first 50 steps (1 second warmup)
- Creates (x, u, x_next) triplets for supervised learning
- Splits into train/val datasets

**Network architecture** (`utils/networks.py`):
- `ObsNet` (phi): 206 → 2048 → 2048 → 1024 (encoder)
- `ObsNet` (psi): 1024 → 2048 → 2048 → 206 (decoder)
- `KNet`: 1024 → 29 (linear policy, single linear layer)
- `ANet`: 1024 → 1024 (state transition in feature space)
- `BNet`: 29 → 1024 (control input matrix in feature space)

**Training objective** (`train.py`):
- Reconstruction: `psi(phi(x)) ≈ x`
- Policy imitation: `K(phi(x)) ≈ u`
- Linear dynamics: `A(phi(x)) + B(u) ≈ phi(x_next)`

### 3. Linear DPG Optimization (`linear-dpg/`)

**Solver** (`linear_dpg/solver.py`):
- Takes A, B, K matrices and phi/psi encoders (exported as ONNX)
- Rolls out the linear dynamics: `z_{t+1} = A @ z_t + B @ K @ z_t`
- Decodes to state space: `x_t = psi(z_t)`
- Evaluates reward: `r_t = R(x_t, u_t, x_{t+1})`
- Optimizes K via gradient-based methods (steepest descent, CG, BFGS, L-BFGS)

**Reward function** (`linear_dpg/humanoid_reward.py`):
- Implements 42 of 45 training reward terms
- Parses 206-dim state vector (alphabetical critic_obs + extended rigid body)
- Un-scales obs to true physical values using inverse obs_scales
- Written in JAX for automatic differentiation

### 4. Evaluation (`humanoidverse/eval_distilled_agent.py`)

Compares three policies side-by-side in IsaacGym:
1. **Original**: Full neural network policy using 575-dim actor_obs
2. **Distilled NN**: `u = KNet(phi(state_206))` — K as neural network
3. **Matrix K**: `u = phi(state_206) @ K_matrix.T` — K as raw matrix

Environments are split into 3 groups (triplets per motion), each group runs one policy.

## Key Dimensions

| Quantity        | Value | Notes                           |
|----------------|-------|---------------------------------|
| actor_obs      | 575   | 115 per frame x 5 history       |
| critic_obs     | 128   | 1 frame, alphabetical order     |
| state_206      | 206   | critic_obs(128) + extended(78)   |
| actions        | 29    | G1 29-DOF joint targets          |
| feature_dim    | 1024  | Latent space dimension           |
| num_envs       | 3960  | Data collection default          |
| num_steps      | 500   | Data collection default          |

## File Structure

```
FALCON/
├── humanoidverse/
│   ├── collect_data.py              # Stage 1: data collection
│   ├── eval_distilled_agent.py      # Stage 4: evaluation
│   └── config/obs/dec_loco/         # Observation configs
├── humanoid_linear_distill/
│   └── src/humanoid_linear_distill/
│       ├── train.py                 # Stage 2: training
│       ├── load_data_example.ipynb  # Data inspection notebook
│       └── utils/
│           ├── data_processor.py    # Data loading & preparation
│           └── networks.py          # Network definitions
└── linear-dpg/
    ├── run_solver.py                # Stage 3: optimization entry point
    └── linear_dpg/
        ├── solver.py                # DPG solver
        └── humanoid_reward.py       # 206-dim reward function
```
