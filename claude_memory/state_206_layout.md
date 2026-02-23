# 206-dim State Vector Layout

## Overview

The 206-dim state vector used by the linear distillation pipeline and the linear-DPG solver is constructed as:

```
state_206 = cat(critic_obs[128], extended_rigid_body[78])
```

It is built in `humanoidverse/collect_data.py` at each simulation step and saved as `state_206.npy`.

## Critical: Alphabetical Field Ordering

The critic_obs fields (first 128 dims) are concatenated in **ALPHABETICAL order**, not in the YAML config file order. This is due to `sorted(obs_config)` in:

```
humanoidverse/envs/legged_base_task/legged_robot_base_ma.py:810
```

```python
def _post_config_observation_callback(self):
    for obs_key, obs_config in self.config.obs.obs_dict.items():
        obs_keys = sorted(obs_config)  # <-- ALPHABETICAL SORT
        current_obs_buf = torch.cat(
            [self.obs_buf_dict_raw[obs_key][key] for key in obs_keys], dim=-1
        )
```

## Full 206-dim Layout

### Critic Obs (128 dims) — Alphabetical Order, obs_scales Applied

| Index     | Field              | Dim | obs_scale | Notes                        |
|-----------|--------------------|-----|-----------|------------------------------|
| [0:29]    | actions            | 29  | 1.0       | Previous actions             |
| [29:32]   | base_ang_vel       | 3   | 0.25      | Angular velocity, base frame |
| [32:35]   | base_lin_vel       | 3   | 2.0       | Linear velocity, base frame  |
| [35:39]   | base_orientation   | 4   | 1.0       | Quaternion [x,y,z,w]         |
| [39:40]   | command_ang_vel    | 1   | 1.0       | Commanded yaw rate           |
| [40:41]   | command_base_height| 1   | 2.0       | Desired base height          |
| [41:43]   | command_lin_vel    | 2   | 1.0       | Commanded vx, vy             |
| [43:44]   | command_stand      | 1   | 1.0       | 0=stance, 1=walking          |
| [44:47]   | command_waist_dofs | 3   | 1.0       | Waist yaw/roll/pitch targets |
| [47:76]   | dof_pos            | 29  | 1.0       | Joint positions              |
| [76:105]  | dof_vel            | 29  | 0.05      | Joint velocities             |
| [105:108] | left_ee_force      | 3   | 0.1       | Left hand force              |
| [108:111] | projected_gravity  | 3   | 1.0       | Gravity in base frame        |
| [111:125] | ref_upper_dof_pos  | 14  | 1.0       | Upper body reference         |
| [125:128] | right_ee_force     | 3   | 0.1       | Right hand force             |

### Extended Rigid Body (78 dims) — Fixed Order, Raw Physical Values

| Index     | Field              | Dim | Notes                        |
|-----------|--------------------|-----|------------------------------|
| [128:129] | base_height        | 1   | Root z position (meters)     |
| [129:158] | torques            | 29  | Joint torques (Nm)           |
| [158:161] | left_foot_pos      | 3   | World position               |
| [161:164] | right_foot_pos     | 3   | World position               |
| [164:168] | left_foot_rot      | 4   | Quaternion [x,y,z,w]         |
| [168:172] | right_foot_rot     | 4   | Quaternion [x,y,z,w]         |
| [172:173] | left_foot_contact_z| 1   | Contact force z (N)          |
| [173:174] | right_foot_contact_z| 1  | Contact force z (N)          |
| [174:177] | left_foot_vel      | 3   | Linear velocity              |
| [177:180] | right_foot_vel     | 3   | Linear velocity              |
| [180:184] | torso_rot          | 4   | Quaternion [x,y,z,w]         |
| [184:187] | torso_ang_vel      | 3   | Angular velocity             |
| [187:190] | pelvis_pos         | 3   | World position               |
| [190:194] | pelvis_rot         | 4   | Quaternion [x,y,z,w]         |
| [194:197] | left_hand_vel      | 3   | Linear velocity              |
| [197:200] | right_hand_vel     | 3   | Linear velocity              |
| [200:203] | left_hand_ang_vel  | 3   | Angular velocity             |
| [203:206] | right_hand_ang_vel | 3   | Angular velocity             |

## Obs Scales (Un-scaling)

The critic_obs fields have obs_scales baked in. To recover true physical values, divide by the obs_scale:

```python
true_value = scaled_value / obs_scale
```

For example, `base_lin_vel` has scale 2.0, so `true_vel = state_206[32:35] / 2.0`.

## Body Names to Indices (G1 Robot)

Used by `extract_extended_state()` to look up rigid body tensors:

| Body Part   | Body Name              | Typical Index |
|-------------|------------------------|---------------|
| Left foot   | left_ankle_roll_link   | 6             |
| Right foot  | right_ankle_roll_link  | 12            |
| Torso       | torso_link             | 15            |
| Left hand   | left_rubber_hand       | 23            |
| Right hand  | right_rubber_hand      | 31            |

Indices are looked up dynamically via `env.body_names.index("...")` in case they change.

## Clipping

All observation values are clipped to [-100, 100] (`clip_observations: 100.0` in the env config).
