# Humanoid Reward Function

## Location

`linear-dpg/linear_dpg/humanoid_reward.py`

## Purpose

Implements the FALCON training reward for the Unitree G1 29-DOF humanoid, written in JAX for use with the linear-DPG solver. The solver needs differentiable rewards to compute policy gradients.

## Interface

```python
class HumanoidReward(RewardFunction):
    def __call__(self, x: jnp.ndarray, u: jnp.ndarray, x_next: jnp.ndarray) -> float:
```

- `x`: Current 206-dim state vector
- `u`: 29-dim action vector (raw, not scaled)
- `x_next`: Next 206-dim state vector

## State Parsing

The `_parse_state()` method extracts named fields from the 206-dim vector and un-scales the critic_obs portion by dividing by obs_scales. See `state_206_layout.md` for the full index layout.

## Reward Terms (42 of 45)

### Tracking Rewards (positive, encourage desired behavior)
| Term | Scale | Description |
|------|-------|-------------|
| tracking_lin_vel_x | +2.0 | Track commanded forward velocity |
| tracking_lin_vel_y | +1.5 | Track commanded lateral velocity |
| tracking_ang_vel | +4.0 | Track commanded yaw rate |
| tracking_walk_base_height | +1.0 | Track height while walking |
| tracking_stance_base_height | +4.0 | Track height while standing |
| tracking_waist_dofs_tapping | +0.5 | Track waist targets while walking |
| tracking_waist_dofs_stance | +3.0 | Track waist targets while standing |
| tracking_upper_body_dofs | +4.0 | Track upper body reference poses |

### Penalty Terms (negative, discourage undesirable behavior)
28 penalty terms covering: velocity penalties, orientation, torques, DOF velocities/accelerations, action rates, contact patterns, foot orientation/height, hip positions, stance symmetry, ankle roll, knee limits, zero-command drift, end-effector accelerations.

### Excluded Terms (3)
- `termination` (-250.0): No reset logic in solver rollouts
- `feet_air_time` (+4.0): Requires stateful air-time tracking
- `penalty_diff_feet_air_time` (-5.0): Same as above

## DOF Index Groups

- **Lower body** [0:15]: left leg [0:6], right leg [6:12], waist [12:15]
- **Upper body** [15:29]: left arm [15:22], right arm [22:29]
- **Waist**: [12, 13, 14]
- **Knees**: [3, 9]
- **Ankle rolls**: left=5, right=11
- **Hip roll/yaw**: [1, 2, 7, 8]

## Quaternion Convention

All quaternions use [x, y, z, w] format (IsaacGym convention).

## Source Reward Implementations

The reward terms were ported from:
- `humanoidverse/envs/decoupled_locomotion/decoupled_locomotion_stand_height_waist_wbc_ma.py`
- `humanoidverse/envs/decoupled_locomotion/decoupled_locomotion_stand_ma.py`
- `humanoidverse/envs/locomotion/locomotion_ma.py`
- `humanoidverse/envs/legged_base_task/legged_robot_base_ma.py`
