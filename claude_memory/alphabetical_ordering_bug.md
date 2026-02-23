# Alphabetical Ordering Bug — Discovery and Fix

## The Bug

The observation buffers in humanoidverse concatenate fields in **alphabetical order** (due to Python's `sorted()` on field name lists), but all downstream code initially assumed the fields were in **YAML config file order**.

This caused the `humanoid_reward.py` `_parse_state()` method and the data inspection notebook to parse completely wrong data at each index — for example, reading `actions` data where it expected `base_orientation`.

## Root Cause

In `humanoidverse/envs/legged_base_task/legged_robot_base_ma.py`, line 810:

```python
def _post_config_observation_callback(self):
    for obs_key, obs_config in self.config.obs.obs_dict.items():
        obs_keys = sorted(obs_config)  # <-- sorts field names alphabetically
        current_obs_buf = torch.cat(
            [self.obs_buf_dict_raw[obs_key][key] for key in obs_keys], dim=-1
        )
```

The YAML config lists fields in a logical order:
```yaml
critic_obs: [base_orientation, base_lin_vel, base_ang_vel, ...]
```

But `sorted()` produces alphabetical order:
```
[actions, base_ang_vel, base_lin_vel, base_orientation, command_ang_vel, ...]
```

## Impact

The wrong ordering affected:
1. **`humanoid_reward.py`**: `_parse_state()` mapped wrong indices to all 15 critic_obs fields
2. **Notebook Cell 3**: Field inspection labels were all wrong
3. **Notebook verification cell**: Cross-checking actor_obs vs critic_obs used wrong index pairs
4. **Notebook visualization cell**: Plotted wrong fields

## How It Was Discovered

A verification cell in `load_data_example.ipynb` compared overlapping fields between `critic_obs` and `actor_obs` using assumed (wrong) indices. Several fields showed FAIL, prompting investigation of the obs construction code.

## Fix Applied

All index mappings were updated to use alphabetical order:

### Old (wrong, YAML config order):
```
[0:4]   base_orientation
[4:7]   base_lin_vel
[7:10]  base_ang_vel
...
[93:122] actions
```

### New (correct, alphabetical):
```
[0:29]  actions
[29:32] base_ang_vel
[32:35] base_lin_vel
[35:39] base_orientation
...
```

## Files Fixed

- `linear-dpg/linear_dpg/humanoid_reward.py` — `_parse_state()` and docstring
- `humanoid_linear_distill/src/humanoid_linear_distill/load_data_example.ipynb` — Cells 3, 5, 6

## Lesson

Always verify observation buffer field ordering against the actual runtime code, not just the config file. The `sorted()` call is easy to miss and completely changes the index layout.
