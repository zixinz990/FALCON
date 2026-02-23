# Claude Memory — FALCON Project

Documentation of key findings, architecture, and lessons learned from working on the FALCON linear distillation pipeline.

## Files

| File | Description |
|------|-------------|
| [pipeline_overview.md](pipeline_overview.md) | End-to-end pipeline: data collection → training → optimization → evaluation |
| [state_206_layout.md](state_206_layout.md) | Complete 206-dim state vector layout with indices, dims, and obs_scales |
| [observation_buffers.md](observation_buffers.md) | Actor/critic obs architecture, history stacking, noise, field overlap |
| [humanoid_reward.md](humanoid_reward.md) | Reward function: 42 terms, DOF groups, quaternion conventions |
| [alphabetical_ordering_bug.md](alphabetical_ordering_bug.md) | Major bug: fields sorted alphabetically, not by config order |
