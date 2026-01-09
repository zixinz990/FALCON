"""
Paired policy comparison: runs original and distilled policies side-by-side.
Even-indexed envs use original policy, odd-indexed use distilled (u = K(phi(x))).
"""

import os
import sys
import types
import logging
from pathlib import Path
from datetime import datetime

import isaacgym
from isaacgym import gymapi, gymutil
import torch
import numpy as np
import hydra
from omegaconf import OmegaConf
from loguru import logger
from tqdm import tqdm
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate

from utils.config_utils import *
from humanoidverse.utils.logging import HydraLoggerBridge
from humanoidverse.utils.helpers import pre_process_config
from humanoidverse.agents.base_algo.base_algo import BaseAlgo
from humanoid_linear_distill.utils.networks import ObsNet, KNet

# Visual marker settings for policy distinction
ORIGINAL_POLICY_COLOR = (0.0, 0.5, 1.0)  # Blue
DISTILLED_POLICY_COLOR = (1.0, 0.3, 0.0)  # Orange
MARKER_HEIGHT_OFFSET = 1.5  # meters above robot
MARKER_RADIUS = 0.15

EVAL_OUTPUT_BASE_DIR = "/home/zixin/Dev/FALCON/logs_eval/linear_distill"


def setup_logging(output_dir):
    """Configure file and console logging."""
    if HydraConfig.initialized():
        hydra_log_path = os.path.join(output_dir, "eval_distilled_agent.log")
        logger.remove()
        logger.add(hydra_log_path, level="DEBUG")
    console_log_level = os.environ.get("LOGURU_LEVEL", "INFO").upper()
    logger.add(sys.stdout, level=console_log_level, colorize=True)
    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger().addHandler(HydraLoggerBridge())


def load_config(override_config):
    """Load and merge training config with overrides. Validates required paths."""
    if override_config.checkpoint is None:
        logger.error("Please provide a checkpoint path via checkpoint=/path/to/ckpt.pt")
        sys.exit(1)
    if override_config.get("distilled_model_dir") is None:
        logger.error(
            "Please provide a distilled model directory via distilled_model_dir=/path/to/model"
        )
        sys.exit(1)

    checkpoint = Path(override_config.checkpoint)
    config_path = checkpoint.parent / "config.yaml"
    if not config_path.exists():
        config_path = checkpoint.parent.parent / "config.yaml"
    if not config_path.exists():
        logger.error(f"Could not find config path: {config_path}")
        sys.exit(1)

    logger.info(f"Loading training config file from {config_path}")
    with open(config_path) as file:
        train_config = OmegaConf.load(file)
    config = OmegaConf.merge(train_config, override_config)
    return config, checkpoint


def configure_env_settings(config, checkpoint):
    """Configure env settings: disable domain rand, set motion file, ensure even num_envs."""
    simulator_type = config.simulator["_target_"].split(".")[-1]
    if simulator_type != "IsaacGym":
        logger.error(
            f"This script only supports IsaacGym simulator. Found: {simulator_type}"
        )
        sys.exit(1)

    pre_process_config(config)
    config.headless = False

    # Ensure even number for paired comparison
    num_envs = config.get("data_collection_num_envs", 3960)
    if num_envs % 2 != 0:
        num_envs = num_envs + 1
    config.num_envs = num_envs
    config.env.config.locomotion_command_resampling_time = 4.0

    # Disable all domain randomizations for fair comparison
    logger.info("Disabling ALL domain randomizations for fair comparison...")
    domain_rand_keys_to_disable = [
        "push_robots",
        "randomize_friction",
        "randomize_base_mass",
        "randomize_link_mass",
        "randomize_base_com",
        "randomize_pd_gain",
        "randomize_torque_rfi",
        "randomize_rfi_lim",
        "randomize_ctrl_delay",
        "randomize_motion_ref_xyz",
        "motion_package_loss",
        "born_offset",
        "born_offset_curriculum",
        "born_heading_randomization",
        "born_heading_curriculum",
    ]
    for key in domain_rand_keys_to_disable:
        if key in config.domain_rand:
            config.domain_rand[key] = False
            logger.debug(f"  Disabled: domain_rand.{key}")

    config.rewards.command_height_scale_initial_scale = 0.0
    config.rewards.force_scale_initial_scale = 0.0
    logger.info(
        f"Number of environments: {config.env.config.num_envs} (paired: {config.env.config.num_envs // 2})"
    )

    # Set motion file based on dataset
    if config.eval_dataset == "cmu":
        config.robot.motion.motion_file = (
            "humanoidverse/data/motions/g1_29dof/v1/cmu_all.pkl"
        )
    elif config.eval_dataset == "accad":
        config.robot.motion.motion_file = (
            "humanoidverse/data/motions/g1_29dof/v1/accad_all.pkl"
        )
    else:
        logger.error(f"Invalid evaluation dataset: {config.eval_dataset}")
        sys.exit(1)

    config.env.config.termination.terminate_when_motion_end = True
    ckpt_num = config.checkpoint.split("/")[-1].split("_")[-1].split(".")[0]
    config.env.config.save_rendering_dir = str(
        checkpoint.parent / "renderings_data" / f"ckpt_{ckpt_num}"
    )


def monkey_patch_paired_reset(env, device):
    """Patch reset to sync paired envs: when one resets, both reset with same state."""
    original_reset_envs_idx = env.reset_envs_idx
    num_envs = env.num_envs
    env.reset_flag = torch.zeros(num_envs, dtype=torch.bool, device=device)
    env.policy_type = torch.zeros(num_envs, dtype=torch.long, device=device)
    env.policy_type[1::2] = 1  # Odd indices use distilled policy

    def custom_reset_envs_idx(self, env_ids, target_states=None, target_buf=None):
        if len(env_ids) == 0:
            return

        # Find all pairs needing reset and reset both envs together
        pair_indices = env_ids // 2
        unique_pairs = torch.unique(pair_indices)
        paired_env_ids = torch.cat([unique_pairs * 2, unique_pairs * 2 + 1])
        paired_env_ids = torch.unique(paired_env_ids)
        paired_env_ids = paired_env_ids[paired_env_ids < num_envs]

        original_reset_envs_idx(paired_env_ids, target_states, target_buf)

        # Sync motion buffers from even (original) to odd (distilled)
        for pair_idx in unique_pairs:
            even_idx = pair_idx * 2
            odd_idx = pair_idx * 2 + 1
            if odd_idx < num_envs:
                if hasattr(self, "motion_start_times"):
                    self.motion_start_times[odd_idx] = self.motion_start_times[even_idx]
                self.episode_length_buf[odd_idx] = self.episode_length_buf[even_idx]
                if hasattr(self, "episode_motion_length"):
                    self.episode_motion_length[odd_idx] = self.episode_motion_length[
                        even_idx
                    ]
                if hasattr(self, "motion_times"):
                    self.motion_times[odd_idx] = self.motion_times[even_idx]

        self.reset_flag[paired_env_ids] = True
        if len(paired_env_ids) > 0:
            logger.debug(f"Paired reset: {paired_env_ids.tolist()}")

    env.reset_envs_idx = types.MethodType(custom_reset_envs_idx, env)


def sync_paired_commands(env, device):
    """Copy commands from even (original) to odd (distilled) envs."""
    env.commands[1::2] = env.commands[0::2].clone()


def sync_paired_motion_ids(env, device):
    """Copy motion IDs from even to odd envs."""
    env.motion_ids[1::2] = env.motion_ids[0::2].clone()
    env.motion_len[1::2] = env.motion_len[0::2].clone()


def sync_paired_motion_times(env, device):
    """Copy motion timing buffers from even to odd envs."""
    if hasattr(env, "motion_start_times"):
        env.motion_start_times[1::2] = env.motion_start_times[0::2].clone()
    env.episode_length_buf[1::2] = env.episode_length_buf[0::2].clone()
    if hasattr(env, "episode_motion_length"):
        env.episode_motion_length[1::2] = env.episode_motion_length[0::2].clone()
    if hasattr(env, "motion_times"):
        env.motion_times[1::2] = env.motion_times[0::2].clone()


def arrange_paired_environments(env, config, device):
    """Position paired robots side-by-side: original (left) and distilled (right)."""
    num_envs = env.num_envs
    num_pairs = num_envs // 2
    env_spacing = config.get("env_spacing", 3.0)
    pair_offset = config.get("pair_offset", 1.5)

    num_cols = int(np.floor(np.sqrt(num_pairs)))
    num_rows = int(np.ceil(num_pairs / num_cols))
    logger.info(f"Arranging {num_pairs} pairs in a {num_rows}x{num_cols} grid")
    logger.info(f"  Environment spacing: {env_spacing}m, Pair offset: {pair_offset}m")
    logger.info("  LEFT = Original policy, RIGHT = Distilled policy")

    new_origins = torch.zeros(num_envs, 3, device=device)
    for pair_idx in range(num_pairs):
        row = pair_idx // num_cols
        col = pair_idx % num_cols
        base_x = row * env_spacing
        base_y = col * (env_spacing + pair_offset)

        # Original policy (even) on left, distilled (odd) on right
        orig_idx = pair_idx * 2
        new_origins[orig_idx, 0] = base_x
        new_origins[orig_idx, 1] = base_y - pair_offset / 2
        new_origins[orig_idx, 2] = 0.0

        dist_idx = pair_idx * 2 + 1
        new_origins[dist_idx, 0] = base_x
        new_origins[dist_idx, 1] = base_y + pair_offset / 2
        new_origins[dist_idx, 2] = 0.0

    env.env_origins = new_origins
    logger.info("Environment origins rearranged for paired comparison.")


def initialize_motion_pool(env, config, device):
    """Load motion pool and assign same motion ID to each pair."""
    num_envs = config.env.config.num_envs
    num_pairs = num_envs // 2
    motion_pool_size = config.get("motion_pool_size", 1980)
    logger.info(
        f"Loading motion pool with {motion_pool_size} motions for {num_pairs} pairs..."
    )

    original_motion_lib_num_envs = env._motion_lib.num_envs
    env._motion_lib.num_envs = motion_pool_size
    env._motion_lib.load_motions(random_sample=False)
    env._motion_lib.num_envs = original_motion_lib_num_envs

    # Assign same motion to both envs in each pair
    pair_motion_ids = torch.randint(0, motion_pool_size, (num_pairs,), device=device)
    env.motion_ids = torch.zeros(num_envs, dtype=torch.long, device=device)
    env.motion_ids[0::2] = pair_motion_ids
    env.motion_ids[1::2] = pair_motion_ids
    env.motion_len = env._motion_lib.get_motion_length(env.motion_ids)

    return motion_pool_size


def load_distilled_policy(config, device):
    """Load distilled policy networks (phi_nn and K_nn)."""
    distilled_model_dir = Path(config.distilled_model_dir)
    feature_dim = config.get("feature_dim", 1024)
    hidden_size = config.get("hidden_size", 2048)
    actor_obs_dim = config.get("actor_obs_dim", 575)
    action_dim = config.get("action_dim", 29)

    logger.info(f"Loading distilled policy from: {distilled_model_dir}")
    logger.info(
        f"  Model dimensions: actor_obs_dim={actor_obs_dim}, action_dim={action_dim}"
    )
    logger.info(
        f"  Network dimensions: feature_dim={feature_dim}, hidden_size={hidden_size}"
    )

    phi_nn = ObsNet(
        input_size=actor_obs_dim, hidden_size=hidden_size, output_size=feature_dim
    ).to(device)
    K_nn = KNet(input_size=feature_dim, output_size=action_dim).to(device)

    phi_path = distilled_model_dir / "phi_nn.pth"
    K_path = distilled_model_dir / "K_nn.pth"
    if not phi_path.exists():
        logger.error(f"phi_nn.pth not found at: {phi_path}")
        sys.exit(1)
    if not K_path.exists():
        logger.error(f"K_nn.pth not found at: {K_path}")
        sys.exit(1)

    phi_nn.load_state_dict(torch.load(phi_path, map_location=device))
    K_nn.load_state_dict(torch.load(K_path, map_location=device))
    phi_nn.eval()
    K_nn.eval()
    logger.info("Distilled policy loaded successfully.")
    return phi_nn, K_nn


def load_original_policy(config, env, device):
    """Load original policy from checkpoint."""
    logger.info("Loading original policy from checkpoint...")
    algo: BaseAlgo = instantiate(config.algo, env=env, device=device, log_dir=None)
    algo.setup()
    algo.load(config.checkpoint)
    algo._eval_mode()
    logger.info("Original policy loaded successfully.")
    return algo


def setup_simulation(config, checkpoint, device):
    """Initialize env, arrange paired positions, and load both policies."""
    configure_env_settings(config, checkpoint)
    logger.info("Instantiating environment...")
    env = instantiate(config.env, device=device)

    arrange_paired_environments(env, config, device)
    monkey_patch_paired_reset(env, device)
    motion_pool_size = initialize_motion_pool(env, config, device)

    logger.info("Loading policies...")
    algo = load_original_policy(config, env, device)
    phi_nn, K_nn = load_distilled_policy(config, device)

    return env, algo, phi_nn, K_nn, motion_pool_size


def handle_timeouts(env, infos, motion_pool_size, step, device):
    """On timeout, assign new motion to both envs in affected pairs."""
    num_envs = env.num_envs
    time_outs = infos.get(
        "time_outs", torch.zeros(num_envs, dtype=torch.bool, device=device)
    )
    timed_out_env_ids = torch.where(time_outs)[0]
    if len(timed_out_env_ids) == 0:
        return

    pair_indices = timed_out_env_ids // 2
    unique_pairs = torch.unique(pair_indices)
    new_motion_ids = torch.randint(
        0, motion_pool_size, (len(unique_pairs),), device=device
    )

    for i, pair_idx in enumerate(unique_pairs):
        env.motion_ids[pair_idx * 2] = new_motion_ids[i]
        env.motion_ids[pair_idx * 2 + 1] = new_motion_ids[i]

    affected_env_ids = torch.cat([unique_pairs * 2, unique_pairs * 2 + 1])
    env.motion_len[affected_env_ids] = env._motion_lib.get_motion_length(
        env.motion_ids[affected_env_ids]
    )


def draw_policy_markers(env):
    """Draw colored spheres above robots: blue=original, orange=distilled."""
    if not hasattr(env, "simulator") or not hasattr(env.simulator, "gym"):
        return
    if env.simulator.headless:
        return
    if not hasattr(env.simulator, "viewer") or env.simulator.viewer is None:
        return

    gym = env.simulator.gym
    viewer = env.simulator.viewer
    gym.clear_lines(viewer)
    root_states = env.simulator.robot_root_states

    for env_id in range(env.num_envs):
        robot_pos = root_states[env_id, :3]
        color = ORIGINAL_POLICY_COLOR if env_id % 2 == 0 else DISTILLED_POLICY_COLOR

        marker_pos = gymapi.Vec3(
            robot_pos[0].item(),
            robot_pos[1].item(),
            robot_pos[2].item() + MARKER_HEIGHT_OFFSET,
        )
        sphere_geom = gymutil.WireframeSphereGeometry(
            MARKER_RADIUS, 12, 12, None, color=color
        )
        sphere_pose = gymapi.Transform(marker_pos, r=None)
        gymutil.draw_lines(
            sphere_geom, gym, viewer, env.simulator.envs[env_id], sphere_pose
        )


def collect_data(env, algo, phi_nn, K_nn, config, motion_pool_size, device):
    """Run simulation: original policy on even envs, distilled on odd. Track rewards."""
    num_steps = config.get("num_steps", 500)
    num_envs = config.env.config.num_envs
    num_pairs = num_envs // 2

    logger.info(
        f"Collecting data for {num_steps} steps with {num_pairs} paired environments..."
    )
    logger.info("  BLUE marker: Original policy (even indices)")
    logger.info("  ORANGE marker: Distilled policy (odd indices)")

    obs_list = []
    actions_original_list = []
    actions_distilled_list = []
    reset_flags_list = []
    completed_episodes = []

    # Per-pair accumulators
    current_episode_rewards_original = torch.zeros(num_pairs, device=device)
    current_episode_rewards_distilled = torch.zeros(num_pairs, device=device)
    current_episode_steps = torch.zeros(num_pairs, dtype=torch.long, device=device)
    current_motion_ids = torch.zeros(num_pairs, dtype=torch.long, device=device)

    obs_dict = env.reset_all()
    sync_paired_commands(env, device)
    sync_paired_motion_ids(env, device)
    sync_paired_motion_times(env, device)
    current_motion_ids[:] = env.motion_ids[0::2]

    with torch.inference_mode():
        for step in tqdm(range(num_steps), desc="Collecting Data (Paired)"):
            env.reset_flag[:] = False
            sync_paired_commands(env, device)
            sync_paired_motion_times(env, device)

            curr_obs = {k: v.cpu().numpy() for k, v in obs_dict.items()}
            obs_list.append(curr_obs)

            actor_obs = obs_dict["actor_obs"]
            actions = torch.zeros(num_envs, env.config.robot.actions_dim, device=device)

            # Original policy for even indices
            even_indices = torch.arange(0, num_envs, 2, device=device)
            if hasattr(algo, "act_inference"):
                actions[even_indices] = algo.act_inference(actor_obs[even_indices])
            else:
                actions[even_indices] = algo.actor.act_inference(
                    actor_obs[even_indices]
                )

            # Distilled policy for odd indices: u = K(phi(x))
            odd_indices = torch.arange(1, num_envs, 2, device=device)
            phi_x = phi_nn(actor_obs[odd_indices])
            actions[odd_indices] = K_nn(phi_x)

            actions_original_list.append(actions[even_indices].cpu().numpy())
            actions_distilled_list.append(actions[odd_indices].cpu().numpy())

            actor_state = {"actions": actions}
            obs_dict, rewards, dones, infos = env.step(actor_state)

            # Accumulate rewards per pair
            if isinstance(rewards, dict):
                total_rewards = sum(rewards.values())
            else:
                total_rewards = rewards
            rewards_original = total_rewards[0::2]
            rewards_distilled = total_rewards[1::2]
            current_episode_rewards_original += rewards_original
            current_episode_rewards_distilled += rewards_distilled
            current_episode_steps += 1

            # Check for episode completion
            reset_flags_even = env.reset_flag[0::2]
            reset_flags_odd = env.reset_flag[1::2]
            completed_pairs = reset_flags_even | reset_flags_odd

            time_outs = infos.get(
                "time_outs", torch.zeros(num_envs, dtype=torch.bool, device=device)
            )
            timeout_pairs = time_outs[0::2] | time_outs[1::2]
            completed_pairs = completed_pairs | timeout_pairs

            # Record completed episodes
            completed_pair_ids = torch.where(completed_pairs)[0]
            for pair_id in completed_pair_ids:
                pid = pair_id.item()
                completed_episodes.append(
                    {
                        "motion_id": current_motion_ids[pid].item(),
                        "original_reward": current_episode_rewards_original[pid].item(),
                        "distilled_reward": current_episode_rewards_distilled[
                            pid
                        ].item(),
                        "steps": current_episode_steps[pid].item(),
                    }
                )
                current_episode_rewards_original[pid] = 0
                current_episode_rewards_distilled[pid] = 0
                current_episode_steps[pid] = 0

            handle_timeouts(env, infos, motion_pool_size, step, device)
            current_motion_ids[:] = env.motion_ids[0::2]
            draw_policy_markers(env)
            reset_flags_list.append(env.reset_flag.cpu().numpy().copy())

    # Record incomplete episodes
    for pid in range(num_pairs):
        if current_episode_steps[pid] > 0:
            completed_episodes.append(
                {
                    "motion_id": current_motion_ids[pid].item(),
                    "original_reward": current_episode_rewards_original[pid].item(),
                    "distilled_reward": current_episode_rewards_distilled[pid].item(),
                    "steps": current_episode_steps[pid].item(),
                }
            )

    return (
        obs_list,
        actions_original_list,
        actions_distilled_list,
        reset_flags_list,
        completed_episodes,
    )


def print_reward_summary(completed_episodes, output_dir):
    """Print and save reward comparison between policies."""
    if not completed_episodes:
        logger.warning("No completed episodes to summarize.")
        return

    logger.info("")
    logger.info("=" * 80)
    logger.info("  REWARD SUMMARY: Original Policy vs Distilled Policy")
    logger.info("=" * 80)
    logger.info("")
    logger.info(
        f"{'Motion ID':>10} | {'Steps':>6} | {'Original':>12} | {'Distilled':>12} | {'Diff':>10} | {'Winner':>10}"
    )
    logger.info("-" * 80)

    total_original = 0.0
    total_distilled = 0.0
    total_steps = 0
    original_wins = 0
    distilled_wins = 0
    ties = 0

    for ep in completed_episodes:
        motion_id = ep["motion_id"]
        steps = ep["steps"]
        orig_reward = ep["original_reward"]
        dist_reward = ep["distilled_reward"]
        diff = orig_reward - dist_reward

        if abs(diff) < 1e-6:
            winner = "TIE"
            ties += 1
        elif diff > 0:
            winner = "Original"
            original_wins += 1
        else:
            winner = "Distilled"
            distilled_wins += 1

        logger.info(
            f"{motion_id:>10} | {steps:>6} | {orig_reward:>12.2f} | {dist_reward:>12.2f} | {diff:>+10.2f} | {winner:>10}"
        )
        total_original += orig_reward
        total_distilled += dist_reward
        total_steps += steps

    logger.info("-" * 80)
    logger.info(
        f"{'TOTAL':>10} | {total_steps:>6} | {total_original:>12.2f} | {total_distilled:>12.2f} | {total_original - total_distilled:>+10.2f} |"
    )
    logger.info("")
    logger.info(f"Number of episodes: {len(completed_episodes)}")
    logger.info(f"Original policy wins: {original_wins}")
    logger.info(f"Distilled policy wins: {distilled_wins}")
    logger.info(f"Ties: {ties}")
    logger.info("")

    if total_original > total_distilled:
        logger.info(">>> OVERALL WINNER: Original Policy <<<")
    elif total_distilled > total_original:
        logger.info(">>> OVERALL WINNER: Distilled Policy <<<")
    else:
        logger.info(">>> OVERALL: TIE <<<")

    if total_steps > 0:
        logger.info("")
        logger.info(
            f"Average reward per step (Original):  {total_original / total_steps:.4f}"
        )
        logger.info(
            f"Average reward per step (Distilled): {total_distilled / total_steps:.4f}"
        )

    rewards_path = output_dir / "reward_comparison.npz"
    np.savez(
        rewards_path,
        episodes=completed_episodes,
        total_original=total_original,
        total_distilled=total_distilled,
        original_wins=original_wins,
        distilled_wins=distilled_wins,
    )
    logger.info(f"\nReward data saved to: {rewards_path}")


def save_results(
    obs_list,
    actions_original_list,
    actions_distilled_list,
    reset_flags_list,
    output_dir,
):
    """Save observations, actions (both policies), and reset flags to disk."""
    obs_path = output_dir / "observations.npz"
    actions_original_path = output_dir / "actions_original.npy"
    actions_distilled_path = output_dir / "actions_distilled.npy"
    reset_flags_path = output_dir / "reset_flags.npy"

    save_obs = {}
    keys = obs_list[0].keys()
    for k in keys:
        save_obs[k] = np.array([x[k] for x in obs_list])

    logger.info(f"Saving observations to {obs_path}")
    np.savez(obs_path, **save_obs)
    for k, v in save_obs.items():
        logger.info(f"  {k}: {v.shape}")

    save_actions_original = np.array(actions_original_list)
    save_actions_distilled = np.array(actions_distilled_list)

    logger.info(
        f"Saving original policy actions to {actions_original_path} with shape {save_actions_original.shape}"
    )
    np.save(actions_original_path, save_actions_original)
    logger.info(
        f"Saving distilled policy actions to {actions_distilled_path} with shape {save_actions_distilled.shape}"
    )
    np.save(actions_distilled_path, save_actions_distilled)

    save_reset_flags = np.array(reset_flags_list)
    logger.info(
        f"Saving reset flags to {reset_flags_path} with shape {save_reset_flags.shape}"
    )
    logger.info(f"  Total resets: {save_reset_flags.sum()}")
    np.save(reset_flags_path, save_reset_flags)


def create_output_dir():
    """Create timestamped output directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(EVAL_OUTPUT_BASE_DIR) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


@hydra.main(config_path="config", config_name="base_eval", version_base="1.1")
def main(override_config: OmegaConf):
    """Entry point: run paired policy comparison and save results."""
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    output_dir = create_output_dir()
    setup_logging(output_dir)
    os.chdir(hydra.utils.get_original_cwd())

    logger.info("=" * 60)
    logger.info("  PAIRED POLICY COMPARISON EVALUATION")
    logger.info("  Original Policy vs Distilled Linear Policy")
    logger.info("=" * 60)
    logger.info(f"Output directory: {output_dir}")

    config, checkpoint = load_config(override_config)
    config_save_path = output_dir / "config.yaml"
    with open(config_save_path, "w") as f:
        OmegaConf.save(config, f)
    logger.info(f"Config saved to: {config_save_path}")

    env, algo, phi_nn, K_nn, motion_pool_size = setup_simulation(
        config, checkpoint, device
    )

    (
        obs_list,
        actions_original_list,
        actions_distilled_list,
        reset_flags_list,
        completed_episodes,
    ) = collect_data(env, algo, phi_nn, K_nn, config, motion_pool_size, device)

    save_results(
        obs_list,
        actions_original_list,
        actions_distilled_list,
        reset_flags_list,
        output_dir,
    )
    print_reward_summary(completed_episodes, output_dir)

    logger.info("=" * 60)
    logger.info("  EVALUATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
