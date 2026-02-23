# Observation Buffer Architecture

## Actor Obs (575 dims)

- **Per-frame dimension**: 115
- **History length**: 5 (stacked temporally)
- **Total**: 115 x 5 = 575
- **History order**: [oldest, ..., newest] — most recent frame is at indices [460:575]
  - Confirmed by `_post_config_observation_callback` which appends `current_obs_buf` at the end

### Per-Frame Layout (115 dims, Alphabetical)

| Index   | Field              | Dim |
|---------|--------------------|-----|
| [0:29]  | actions            | 29  |
| [29:32] | base_ang_vel       | 3   |
| [32:33] | command_ang_vel    | 1   |
| [33:34] | command_base_height| 1   |
| [34:36] | command_lin_vel    | 2   |
| [36:37] | command_stand      | 1   |
| [37:40] | command_waist_dofs | 3   |
| [40:69] | dof_pos            | 29  |
| [69:98] | dof_vel            | 29  |
| [98:101]| projected_gravity  | 3   |
| [101:115]| ref_upper_dof_pos | 14  |

**Fields NOT in actor_obs** (critic-only): `base_orientation`, `base_lin_vel`, `left_ee_apply_force`, `right_ee_apply_force`

## Critic Obs (128 dims)

- **Per-frame dimension**: 128
- **History length**: 1 (no history)
- **Layout**: See `state_206_layout.md` for full alphabetical field listing

## Overlap Between Actor and Critic

All 115 actor_obs fields (per frame) also appear in critic_obs. The 13 extra dims in critic_obs are:
- `base_lin_vel` (3) — at critic [32:35]
- `base_orientation` (4) — at critic [35:39]
- `left_ee_apply_force` (3) — at critic [105:108]
- `right_ee_apply_force` (3) — at critic [125:128]

### Shared Field Index Mapping

| Field              | Critic Slice | Actor Last-Frame Slice |
|--------------------|-------------|------------------------|
| actions            | [0:29]      | [460:489]              |
| base_ang_vel       | [29:32]     | [489:492]              |
| command_ang_vel    | [39:40]     | [492:493]              |
| command_base_height| [40:41]     | [493:494]              |
| command_lin_vel    | [41:43]     | [494:496]              |
| command_stand      | [43:44]     | [496:497]              |
| command_waist_dofs | [44:47]     | [497:500]              |
| dof_pos            | [47:76]     | [500:529]              |
| dof_vel            | [76:105]    | [529:558]              |
| projected_gravity  | [108:111]   | [558:561]              |
| ref_upper_dof_pos  | [111:125]   | [561:575]              |

These should be exactly equal when `add_noise: False` in the obs config. When noise is enabled (`add_noise: True`), `dof_pos` (noise_scale=0.01) and `dof_vel` (noise_scale=0.1) will differ due to independent noise sampling.

## Noise Configuration

From `g1_29dof_obs_diff_force_history_wolinvel_ma.yaml`:

- `add_noise: False` — when False, all noise_scales are zeroed out (line 47-50 of `decoupled_locomotion_stand_ma.py`)
- Noise formula: `(raw_obs + uniform[-1,1] * noise_scale * noise_level) * obs_scale`
- Non-zero noise_scales: `dof_pos: 0.01`, `dof_vel: 0.1`
- Noise is sampled independently for actor_obs and critic_obs (separate calls to `parse_observation`)

## Key Source Files

- Config: `humanoidverse/config/obs/dec_loco/g1_29dof_obs_diff_force_history_wolinvel_ma.yaml`
- Obs computation: `humanoidverse/envs/legged_base_task/legged_robot_base_ma.py:791` (`_compute_observations`)
- History stacking: `humanoidverse/envs/legged_base_task/legged_robot_base_ma.py:808` (`_post_config_observation_callback`)
- Scale+noise: `humanoidverse/utils/helpers.py:94` (`parse_observation`)
- Noise zeroing: `humanoidverse/envs/decoupled_locomotion/decoupled_locomotion_stand_ma.py:47`
