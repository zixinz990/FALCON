"""
Triplet policy comparison: runs three policies side-by-side.
Env index % 3 == 0: original policy
Env index % 3 == 1: Distilled (NN) policy (u = K_nn(phi(x)))
Env index % 3 == 2: Distilled (matrix) policy (u = K @ phi(x), extracted from K_nn)
"""

import os
import sys
import types
import logging
from pathlib import Path
from datetime import datetime

import h5py
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
from humanoid_linear_distill.utils.networks import (
    TwoLayerNet,
    SingleLayerNet,
    IdentityEncoder,
)

# Visual marker settings for policy distinction
ORIGINAL_POLICY_COLOR = (0.0, 0.5, 1.0)  # Blue
DISTILLED_POLICY_COLOR = (1.0, 0.3, 0.0)  # Orange
MATRIX_POLICY_COLOR = (0.0, 0.8, 0.2)  # Green
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
    """Configure env settings: disable domain rand, set motion file, ensure num_envs divisible by 3."""
    simulator_type = config.simulator["_target_"].split(".")[-1]
    if simulator_type != "IsaacGym":
        logger.error(
            f"This script only supports IsaacGym simulator. Found: {simulator_type}"
        )
        sys.exit(1)

    pre_process_config(config)
    config.headless = True

    # Ensure num_envs is divisible by 3 for triplet comparison
    num_envs = config.get("data_collection_num_envs", 3960)
    remainder = num_envs % 3
    if remainder != 0:
        num_envs = num_envs + (3 - remainder)
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
        f"Number of environments: {config.env.config.num_envs} (triplets: {config.env.config.num_envs // 3})"
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


def monkey_patch_triplet_reset(env, device):
    """Patch reset to sync triplet envs: when one resets, all three reset with same state."""
    original_reset_envs_idx = env.reset_envs_idx
    num_envs = env.num_envs
    env.reset_flag = torch.zeros(num_envs, dtype=torch.bool, device=device)
    env.policy_type = torch.zeros(num_envs, dtype=torch.long, device=device)
    env.policy_type[1::3] = 1  # Distilled (NN) policy
    env.policy_type[2::3] = 2  # Distilled (matrix) policy

    def custom_reset_envs_idx(self, env_ids, target_states=None, target_buf=None):
        if len(env_ids) == 0:
            return

        # Find all triplets needing reset and reset all three envs together
        triplet_indices = env_ids // 3
        unique_triplets = torch.unique(triplet_indices)
        triplet_env_ids = torch.cat(
            [unique_triplets * 3, unique_triplets * 3 + 1, unique_triplets * 3 + 2]
        )
        triplet_env_ids = torch.unique(triplet_env_ids)
        triplet_env_ids = triplet_env_ids[triplet_env_ids < num_envs]

        original_reset_envs_idx(triplet_env_ids, target_states, target_buf)

        # Sync motion buffers from leader (index 0 in triplet) to the other two
        for triplet_idx in unique_triplets:
            leader_idx = triplet_idx * 3
            for offset in [1, 2]:
                follower_idx = triplet_idx * 3 + offset
                if follower_idx < num_envs:
                    if hasattr(self, "motion_start_times"):
                        self.motion_start_times[follower_idx] = self.motion_start_times[
                            leader_idx
                        ]
                    self.episode_length_buf[follower_idx] = self.episode_length_buf[
                        leader_idx
                    ]
                    if hasattr(self, "episode_motion_length"):
                        self.episode_motion_length[follower_idx] = (
                            self.episode_motion_length[leader_idx]
                        )
                    if hasattr(self, "motion_times"):
                        self.motion_times[follower_idx] = self.motion_times[leader_idx]

        self.reset_flag[triplet_env_ids] = True
        if len(triplet_env_ids) > 0:
            logger.debug(f"Triplet reset: {triplet_env_ids.tolist()}")

    env.reset_envs_idx = types.MethodType(custom_reset_envs_idx, env)


def sync_triplet_commands(env, device):
    """Copy commands from leader (index 0 in triplet) to the other two."""
    env.commands[1::3] = env.commands[0::3].clone()
    env.commands[2::3] = env.commands[0::3].clone()


def sync_triplet_motion_ids(env, device):
    """Copy motion IDs from leader to the other two in each triplet."""
    env.motion_ids[1::3] = env.motion_ids[0::3].clone()
    env.motion_ids[2::3] = env.motion_ids[0::3].clone()
    env.motion_len[1::3] = env.motion_len[0::3].clone()
    env.motion_len[2::3] = env.motion_len[0::3].clone()


def sync_triplet_motion_times(env, device):
    """Copy motion timing buffers from leader to the other two in each triplet."""
    if hasattr(env, "motion_start_times"):
        env.motion_start_times[1::3] = env.motion_start_times[0::3].clone()
        env.motion_start_times[2::3] = env.motion_start_times[0::3].clone()
    env.episode_length_buf[1::3] = env.episode_length_buf[0::3].clone()
    env.episode_length_buf[2::3] = env.episode_length_buf[0::3].clone()
    if hasattr(env, "episode_motion_length"):
        env.episode_motion_length[1::3] = env.episode_motion_length[0::3].clone()
        env.episode_motion_length[2::3] = env.episode_motion_length[0::3].clone()
    if hasattr(env, "motion_times"):
        env.motion_times[1::3] = env.motion_times[0::3].clone()
        env.motion_times[2::3] = env.motion_times[0::3].clone()


def arrange_triplet_environments(env, config, device):
    """Position triplet robots side-by-side: original (left), distilled NN (center), matrix K (right)."""
    num_envs = env.num_envs
    num_triplets = num_envs // 3
    env_spacing = config.get("env_spacing", 3.0)
    triplet_offset = config.get("pair_offset", 1.5)

    num_cols = int(np.floor(np.sqrt(num_triplets)))
    num_rows = int(np.ceil(num_triplets / num_cols))
    logger.info(f"Arranging {num_triplets} triplets in a {num_rows}x{num_cols} grid")
    logger.info(
        f"  Environment spacing: {env_spacing}m, Triplet offset: {triplet_offset}m"
    )
    logger.info(
        "  LEFT = Original policy, CENTER = Distilled (NN) policy, RIGHT = Distilled (matrix) policy"
    )

    new_origins = torch.zeros(num_envs, 3, device=device)
    for triplet_idx in range(num_triplets):
        row = triplet_idx // num_cols
        col = triplet_idx % num_cols
        base_x = row * env_spacing
        base_y = col * (env_spacing + 2 * triplet_offset)

        # Original policy (i%3==0) on left
        orig_idx = triplet_idx * 3
        new_origins[orig_idx, 0] = base_x
        new_origins[orig_idx, 1] = base_y - triplet_offset
        new_origins[orig_idx, 2] = 0.0

        # Distilled (NN) policy (i%3==1) in center
        dist_nn_idx = triplet_idx * 3 + 1
        new_origins[dist_nn_idx, 0] = base_x
        new_origins[dist_nn_idx, 1] = base_y
        new_origins[dist_nn_idx, 2] = 0.0

        # Distilled (matrix) policy (i%3==2) on right
        mat_idx = triplet_idx * 3 + 2
        new_origins[mat_idx, 0] = base_x
        new_origins[mat_idx, 1] = base_y + triplet_offset
        new_origins[mat_idx, 2] = 0.0

    env.env_origins = new_origins
    logger.info("Environment origins rearranged for triplet comparison.")


def initialize_motion_pool(env, config, device):
    """Load motion pool and assign same motion ID to each triplet."""
    num_envs = config.env.config.num_envs
    num_triplets = num_envs // 3
    motion_pool_size = config.get("motion_pool_size", 1980)
    logger.info(
        f"Loading motion pool with {motion_pool_size} motions for {num_triplets} triplets..."
    )

    original_motion_lib_num_envs = env._motion_lib.num_envs
    env._motion_lib.num_envs = motion_pool_size
    env._motion_lib.load_motions(random_sample=False)
    env._motion_lib.num_envs = original_motion_lib_num_envs

    # Assign same motion to all three envs in each triplet
    triplet_motion_ids = torch.randint(
        0, motion_pool_size, (num_triplets,), device=device
    )
    env.motion_ids = torch.zeros(num_envs, dtype=torch.long, device=device)
    env.motion_ids[0::3] = triplet_motion_ids
    env.motion_ids[1::3] = triplet_motion_ids
    env.motion_ids[2::3] = triplet_motion_ids
    env.motion_len = env._motion_lib.get_motion_length(env.motion_ids)

    return motion_pool_size


def load_distilled_policy(config, device):
    """Load distilled policy networks (phi_nn and K_nn).

    Auto-detects input dimension from saved phi_nn weights.
    Returns state_type ("actor_obs" or "state_206") indicating which obs to feed.
    Override with config.distilled_state_type = "actor_obs" | "state_206" | "auto".
    """
    distilled_model_dir = Path(config.distilled_model_dir)
    feature_dim = config.get("feature_dim", 1024)
    hidden_size = config.get("hidden_size", 2048)
    action_dim = config.get("action_dim", 29)

    phi_path = distilled_model_dir / "phi_nn.pth"
    K_path = distilled_model_dir / "K_nn.pth"
    if not phi_path.exists():
        logger.error(f"phi_nn.pth not found at: {phi_path}")
        sys.exit(1)
    if not K_path.exists():
        logger.error(f"K_nn.pth not found at: {K_path}")
        sys.exit(1)

    # Auto-detect architecture from saved weights
    phi_state = torch.load(phi_path, map_location=device)
    is_identity = any(k.startswith("extra_net.") for k in phi_state)

    if is_identity:
        input_size = phi_state["extra_net.linear1.weight"].shape[1]
        # Dynamically detect encoder hidden dims and feature dim from state_dict
        extra_linear_indices = sorted(
            int(k.split(".")[1].replace("linear", ""))
            for k in phi_state
            if k.startswith("extra_net.linear") and k.endswith(".weight")
        )
        encoder_hidden_dims = [
            int(phi_state[f"extra_net.linear{idx}.weight"].shape[0])
            for idx in extra_linear_indices[:-1]
        ]
        learned_feature_dim = int(
            phi_state[f"extra_net.linear{extra_linear_indices[-1]}.weight"].shape[0]
        )
        latent_dim = input_size + learned_feature_dim
    else:
        input_size = phi_state["linear1.weight"].shape[1]
        latent_dim = feature_dim

    # Determine state_type: auto-detect from input_size or use explicit override
    distilled_state_type = config.get("distilled_state_type", "auto")
    if distilled_state_type == "auto":
        if input_size == 575:
            state_type = "actor_obs"
        elif input_size == 206:
            state_type = "state_206"
        else:
            state_type = "state_206"
            logger.warning(
                f"Unknown input_size {input_size} from phi_nn weights, defaulting to state_206"
            )
    else:
        state_type = distilled_state_type
        expected_size = 575 if state_type == "actor_obs" else 206
        if input_size != expected_size:
            logger.warning(
                f"distilled_state_type={state_type} expects input_size={expected_size}, "
                f"but phi_nn weights have input_size={input_size}. Proceeding with override."
            )

    logger.info(f"Loading distilled policy from: {distilled_model_dir}")
    logger.info(
        f"  State type: {state_type} (input_size={input_size}), action_dim={action_dim}"
    )
    logger.info(
        f"  Network dimensions: latent_dim={latent_dim}, hidden_size={hidden_size}"
    )

    if is_identity:
        phi_nn = IdentityEncoder(
            input_size, *encoder_hidden_dims, learned_feature_dim
        ).to(device)
    else:
        # TODO: legacy TwoLayerNet now includes BatchNorm — old checkpoints
        #       without bn keys will fail to load here.
        phi_nn = TwoLayerNet(input_size, hidden_size, hidden_size, feature_dim).to(
            device
        )
    K_nn = SingleLayerNet(latent_dim, action_dim, bias=False).to(device)

    phi_nn.load_state_dict(phi_state)
    K_nn.load_state_dict(torch.load(K_path, map_location=device))
    phi_nn.eval()
    K_nn.eval()

    # Extract K matrix from the single linear layer of K_nn
    K_matrix = K_nn.linear1.weight.data.clone()  # (action_dim, latent_dim)
    logger.info(f"Extracted K matrix {tuple(K_matrix.shape)} from K_nn")

    # Load reward_nn if available (optional)
    reward_path = distilled_model_dir / "reward_nn.pth"
    reward_nn = None
    if reward_path.exists():
        from humanoid_linear_distill.export_models import load_and_build

        reward_nn_model, _ = load_and_build(
            str(distilled_model_dir), "reward_nn", device
        )
        reward_nn = reward_nn_model
        reward_nn.eval()
        logger.info(f"  Loaded reward_nn from {reward_path}")
    else:
        logger.info("  reward_nn.pth not found, skipping reward estimation")

    logger.info("Distilled policy loaded successfully.")
    return phi_nn, K_nn, K_matrix, state_type, reward_nn


def load_original_policy(config, env, device):
    """Load original policy from checkpoint."""
    logger.info("Loading original policy from checkpoint...")
    algo: BaseAlgo = instantiate(config.algo, env=env, device=device, log_dir=None)
    algo.setup()
    algo.load(config.checkpoint)
    algo._eval_mode()
    logger.info("Original policy loaded successfully.")
    return algo


def extract_extended_state(env, body_idx):
    """Extract 78-dim extended rigid body state from simulator.

    Layout: base_height(1), torques(29), left_foot_pos(3), right_foot_pos(3),
            left_foot_rot(4), right_foot_rot(4), left_foot_contact_z(1),
            right_foot_contact_z(1), left_foot_vel(3), right_foot_vel(3),
            torso_rot(4), torso_ang_vel(3), pelvis_pos(3), pelvis_rot(4),
            left_hand_vel(3), right_hand_vel(3), left_hand_ang_vel(3),
            right_hand_ang_vel(3)
    """
    sim = env.simulator
    return torch.cat(
        [
            sim.robot_root_states[:, 2:3],
            env.torques,
            sim._rigid_body_pos[:, body_idx["left_foot"], :],
            sim._rigid_body_pos[:, body_idx["right_foot"], :],
            sim._rigid_body_rot[:, body_idx["left_foot"], :],
            sim._rigid_body_rot[:, body_idx["right_foot"], :],
            sim.contact_forces[:, body_idx["left_foot"], 2:3],
            sim.contact_forces[:, body_idx["right_foot"], 2:3],
            sim._rigid_body_vel[:, body_idx["left_foot"], :],
            sim._rigid_body_vel[:, body_idx["right_foot"], :],
            sim._rigid_body_rot[:, body_idx["torso"], :],
            sim._rigid_body_ang_vel[:, body_idx["torso"], :],
            sim.robot_root_states[:, 0:3],
            sim.robot_root_states[:, 3:7],
            sim._rigid_body_vel[:, body_idx["left_hand"], :],
            sim._rigid_body_vel[:, body_idx["right_hand"], :],
            sim._rigid_body_ang_vel[:, body_idx["left_hand"], :],
            sim._rigid_body_ang_vel[:, body_idx["right_hand"], :],
        ],
        dim=1,
    )


def setup_simulation(config, checkpoint, device):
    """Initialize env, arrange triplet positions, and load all three policies."""
    configure_env_settings(config, checkpoint)
    logger.info("Instantiating environment...")
    env = instantiate(config.env, device=device)

    # Look up body indices for extended state extraction
    body_idx = {
        "left_foot": env.body_names.index("left_ankle_roll_link"),
        "right_foot": env.body_names.index("right_ankle_roll_link"),
        "torso": env.body_names.index("torso_link"),
        "left_hand": env.body_names.index("left_rubber_hand"),
        "right_hand": env.body_names.index("right_rubber_hand"),
    }
    logger.info(f"Body indices for extended state: {body_idx}")

    arrange_triplet_environments(env, config, device)
    monkey_patch_triplet_reset(env, device)
    motion_pool_size = initialize_motion_pool(env, config, device)

    logger.info("Loading policies...")
    algo = load_original_policy(config, env, device)
    phi_nn, K_nn, K_matrix, state_type, reward_nn = load_distilled_policy(
        config, device
    )

    return (
        env,
        algo,
        phi_nn,
        K_nn,
        K_matrix,
        motion_pool_size,
        body_idx,
        state_type,
        reward_nn,
    )


def handle_timeouts(env, infos, motion_pool_size, step, device):
    """On timeout, assign new motion to all three envs in affected triplets."""
    num_envs = env.num_envs
    time_outs = infos.get(
        "time_outs", torch.zeros(num_envs, dtype=torch.bool, device=device)
    )
    timed_out_env_ids = torch.where(time_outs)[0]
    if len(timed_out_env_ids) == 0:
        return

    triplet_indices = timed_out_env_ids // 3
    unique_triplets = torch.unique(triplet_indices)
    new_motion_ids = torch.randint(
        0, motion_pool_size, (len(unique_triplets),), device=device
    )

    for i, triplet_idx in enumerate(unique_triplets):
        env.motion_ids[triplet_idx * 3] = new_motion_ids[i]
        env.motion_ids[triplet_idx * 3 + 1] = new_motion_ids[i]
        env.motion_ids[triplet_idx * 3 + 2] = new_motion_ids[i]

    affected_env_ids = torch.cat(
        [unique_triplets * 3, unique_triplets * 3 + 1, unique_triplets * 3 + 2]
    )
    env.motion_len[affected_env_ids] = env._motion_lib.get_motion_length(
        env.motion_ids[affected_env_ids]
    )


def draw_policy_markers(env):
    """Draw colored spheres above robots: blue=Original, orange=Distilled (NN), green=Distilled (matrix)."""
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
        policy_idx = env_id % 3
        if policy_idx == 0:
            color = ORIGINAL_POLICY_COLOR
        elif policy_idx == 1:
            color = DISTILLED_POLICY_COLOR
        else:
            color = MATRIX_POLICY_COLOR

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


# def collect_data(
#     env, algo, phi_nn, K_nn, K_matrix, b_vector, config, motion_pool_size, device
# ):
def collect_data(
    env,
    algo,
    phi_nn,
    K_nn,
    K_matrix,
    config,
    motion_pool_size,
    body_idx,
    state_type,
    reward_nn,
    device,
):
    """Run simulation with three policies per triplet. Track rewards for all three."""
    num_steps = config.get("num_steps", 500)
    num_envs = config.env.config.num_envs
    num_triplets = num_envs // 3

    logger.info(
        f"Collecting data for {num_steps} steps with {num_triplets} triplet environments..."
    )
    logger.info("  BLUE marker: Original policy (i%3==0)")
    logger.info("  ORANGE marker: Distilled (NN) policy (i%3==1)")
    logger.info("  GREEN marker: Distilled (matrix) policy (i%3==2)")

    obs_list = []
    commands_list = []
    base_height_list = []
    base_position_list = []
    base_orientation_list = []
    hand_positions_list = []
    hand_velocities_list = []
    actions_all_list = []
    actions_original_list = []
    actions_distilled_list = []
    actions_matrix_list = []
    reset_flags_list = []
    completed_episodes = []

    # Get hand link indices
    left_hand_idx = env.body_names.index("left_rubber_hand")
    right_hand_idx = env.body_names.index("right_rubber_hand")
    logger.info(f"Hand link indices - Left: {left_hand_idx}, Right: {right_hand_idx}")

    # Per-triplet accumulators
    current_episode_rewards_original = torch.zeros(num_triplets, device=device)
    current_episode_rewards_distilled = torch.zeros(num_triplets, device=device)
    current_episode_rewards_matrix = torch.zeros(num_triplets, device=device)
    current_episode_steps = torch.zeros(num_triplets, dtype=torch.long, device=device)
    current_motion_ids = torch.zeros(num_triplets, dtype=torch.long, device=device)

    # Reward model prediction accumulators (only if reward_nn is available)
    if reward_nn is not None:
        current_episode_reward_pred_original = torch.zeros(num_triplets, device=device)
        current_episode_reward_pred_distilled = torch.zeros(num_triplets, device=device)
        current_episode_reward_pred_matrix = torch.zeros(num_triplets, device=device)
        # Per-step error tracking for MAE/RMSE
        reward_abs_error_sum = torch.zeros(
            3, device=device
        )  # [original, distilled, matrix]
        reward_sq_error_sum = torch.zeros(3, device=device)
        reward_error_count = torch.zeros(3, dtype=torch.long, device=device)

    obs_dict = env.reset_all()
    sync_triplet_commands(env, device)
    sync_triplet_motion_ids(env, device)
    sync_triplet_motion_times(env, device)
    current_motion_ids[:] = env.motion_ids[0::3]

    with torch.inference_mode():
        for step in tqdm(range(num_steps), desc="Collecting Data (Triplet)"):
            env.reset_flag[:] = False
            sync_triplet_commands(env, device)
            sync_triplet_motion_times(env, device)

            curr_obs = {k: v.cpu().numpy() for k, v in obs_dict.items()}
            obs_list.append(curr_obs)

            # Record commands for all robots
            commands_list.append(env.commands.cpu().numpy().copy())

            actor_obs = obs_dict["actor_obs"]
            actions = torch.zeros(num_envs, env.config.robot.actions_dim, device=device)

            # Original policy for i%3==0 (uses 575-dim actor_obs)
            orig_indices = torch.arange(0, num_envs, 3, device=device)
            if hasattr(algo, "act_inference"):
                actions[orig_indices] = algo.act_inference(actor_obs[orig_indices])
            else:
                actions[orig_indices] = algo.actor.act_inference(
                    actor_obs[orig_indices]
                )

            # Build input for distilled policies based on state_type
            if state_type == "actor_obs":
                distilled_input = actor_obs  # 575-dim
            else:
                critic_obs = obs_dict["critic_obs"]
                extended_state = extract_extended_state(env, body_idx)
                distilled_input = torch.cat(
                    [critic_obs, extended_state], dim=1
                )  # 206-dim

            # Distilled (NN) policy for i%3==1: u = K_nn(phi_nn(x))
            dist_nn_indices = torch.arange(1, num_envs, 3, device=device)
            phi_x_nn = phi_nn(distilled_input[dist_nn_indices])
            actions[dist_nn_indices] = K_nn(phi_x_nn)

            # Distilled (matrix) policy for i%3==2: u = K @ phi_nn(x)
            mat_indices = torch.arange(2, num_envs, 3, device=device)
            phi_x_mat = phi_nn(distilled_input[mat_indices])
            actions[mat_indices] = phi_x_mat @ K_matrix.T

            actions_original_list.append(actions[orig_indices].cpu().numpy())
            actions_distilled_list.append(actions[dist_nn_indices].cpu().numpy())
            actions_matrix_list.append(actions[mat_indices].cpu().numpy())
            actions_all_list.append(actions.cpu().numpy().copy())

            actor_state = {"actions": actions}
            obs_dict, rewards, dones, infos = env.step(actor_state)

            # Record base pose (position and orientation) after physics step
            base_height_list.append(
                env.simulator.robot_root_states[:, 2].cpu().numpy().copy()
            )
            base_position_list.append(
                env.simulator.robot_root_states[:, 0:3].cpu().numpy().copy()
            )
            base_orientation_list.append(
                env.simulator.robot_root_states[:, 3:7].cpu().numpy().copy()
            )

            # Record hand positions and velocities
            left_hand_pos = (
                env.simulator._rigid_body_pos[:, left_hand_idx, :].cpu().numpy()
            )
            right_hand_pos = (
                env.simulator._rigid_body_pos[:, right_hand_idx, :].cpu().numpy()
            )
            hand_positions_list.append(
                np.stack([left_hand_pos, right_hand_pos], axis=1).copy()
            )  # (num_envs, 2, 3)

            left_hand_vel = (
                env.simulator._rigid_body_vel[:, left_hand_idx, :].cpu().numpy()
            )
            right_hand_vel = (
                env.simulator._rigid_body_vel[:, right_hand_idx, :].cpu().numpy()
            )
            hand_velocities_list.append(
                np.stack([left_hand_vel, right_hand_vel], axis=1).copy()
            )  # (num_envs, 2, 3)

            # Accumulate rewards per triplet
            if isinstance(rewards, dict):
                total_rewards = sum(rewards.values())
            else:
                total_rewards = rewards
            rewards_original = total_rewards[0::3]
            rewards_distilled = total_rewards[1::3]
            rewards_matrix = total_rewards[2::3]
            current_episode_rewards_original += rewards_original
            current_episode_rewards_distilled += rewards_distilled
            current_episode_rewards_matrix += rewards_matrix
            current_episode_steps += 1

            # Reward model estimation
            if reward_nn is not None:
                z_all = phi_nn(distilled_input)
                predicted_rewards = reward_nn(z_all).squeeze(-1)  # (num_envs,)
                pred_original = predicted_rewards[0::3]
                pred_distilled = predicted_rewards[1::3]
                pred_matrix = predicted_rewards[2::3]
                current_episode_reward_pred_original += pred_original
                current_episode_reward_pred_distilled += pred_distilled
                current_episode_reward_pred_matrix += pred_matrix
                # Per-step error tracking
                reward_abs_error_sum[0] += (
                    (pred_original - rewards_original).abs().sum()
                )
                reward_abs_error_sum[1] += (
                    (pred_distilled - rewards_distilled).abs().sum()
                )
                reward_abs_error_sum[2] += (pred_matrix - rewards_matrix).abs().sum()
                reward_sq_error_sum[0] += (
                    (pred_original - rewards_original) ** 2
                ).sum()
                reward_sq_error_sum[1] += (
                    (pred_distilled - rewards_distilled) ** 2
                ).sum()
                reward_sq_error_sum[2] += ((pred_matrix - rewards_matrix) ** 2).sum()
                reward_error_count += num_triplets

            # Check for episode completion
            reset_flags_0 = env.reset_flag[0::3]
            reset_flags_1 = env.reset_flag[1::3]
            reset_flags_2 = env.reset_flag[2::3]
            completed_triplets = reset_flags_0 | reset_flags_1 | reset_flags_2

            time_outs = infos.get(
                "time_outs", torch.zeros(num_envs, dtype=torch.bool, device=device)
            )
            timeout_triplets = time_outs[0::3] | time_outs[1::3] | time_outs[2::3]
            completed_triplets = completed_triplets | timeout_triplets

            # Record completed episodes
            completed_triplet_ids = torch.where(completed_triplets)[0]
            for triplet_id in completed_triplet_ids:
                tid = triplet_id.item()
                ep_data = {
                    "motion_id": current_motion_ids[tid].item(),
                    "original_reward": current_episode_rewards_original[tid].item(),
                    "distilled_reward": current_episode_rewards_distilled[tid].item(),
                    "matrix_reward": current_episode_rewards_matrix[tid].item(),
                    "steps": current_episode_steps[tid].item(),
                }
                if reward_nn is not None:
                    ep_data["reward_pred_original"] = (
                        current_episode_reward_pred_original[tid].item()
                    )
                    ep_data["reward_pred_distilled"] = (
                        current_episode_reward_pred_distilled[tid].item()
                    )
                    ep_data["reward_pred_matrix"] = current_episode_reward_pred_matrix[
                        tid
                    ].item()
                    current_episode_reward_pred_original[tid] = 0
                    current_episode_reward_pred_distilled[tid] = 0
                    current_episode_reward_pred_matrix[tid] = 0
                completed_episodes.append(ep_data)
                current_episode_rewards_original[tid] = 0
                current_episode_rewards_distilled[tid] = 0
                current_episode_rewards_matrix[tid] = 0
                current_episode_steps[tid] = 0

            handle_timeouts(env, infos, motion_pool_size, step, device)
            current_motion_ids[:] = env.motion_ids[0::3]
            draw_policy_markers(env)
            reset_flags_list.append(env.reset_flag.cpu().numpy().copy())

    # Record incomplete episodes
    for tid in range(num_triplets):
        if current_episode_steps[tid] > 0:
            ep_data = {
                "motion_id": current_motion_ids[tid].item(),
                "original_reward": current_episode_rewards_original[tid].item(),
                "distilled_reward": current_episode_rewards_distilled[tid].item(),
                "matrix_reward": current_episode_rewards_matrix[tid].item(),
                "steps": current_episode_steps[tid].item(),
            }
            if reward_nn is not None:
                ep_data["reward_pred_original"] = current_episode_reward_pred_original[
                    tid
                ].item()
                ep_data["reward_pred_distilled"] = (
                    current_episode_reward_pred_distilled[tid].item()
                )
                ep_data["reward_pred_matrix"] = current_episode_reward_pred_matrix[
                    tid
                ].item()
            completed_episodes.append(ep_data)

    # Compute per-step reward estimation error stats
    reward_estimation_stats = None
    if reward_nn is not None:
        reward_estimation_stats = {
            "mae": (reward_abs_error_sum / reward_error_count.float()).cpu().tolist(),
            "rmse": ((reward_sq_error_sum / reward_error_count.float()).sqrt())
            .cpu()
            .tolist(),
        }

    return (
        obs_list,
        commands_list,
        base_height_list,
        base_position_list,
        base_orientation_list,
        hand_positions_list,
        hand_velocities_list,
        actions_all_list,
        actions_original_list,
        actions_distilled_list,
        actions_matrix_list,
        reset_flags_list,
        completed_episodes,
        reward_estimation_stats,
    )


def print_reward_summary(completed_episodes, output_dir, reward_estimation_stats=None):
    """Print and save reward comparison between all three policies."""
    if not completed_episodes:
        logger.warning("No completed episodes to summarize.")
        return

    logger.info("")
    logger.info("=" * 110)
    logger.info("  REWARD SUMMARY: Original vs Distilled (NN) vs Distilled (matrix)")
    logger.info("=" * 110)
    logger.info("")
    logger.info(
        f"{'Motion ID':>10} | {'Steps':>6} | {'Original':>12} | {'Dist (NN)':>12} | {'Dist (mat)':>12} | {'O-D(NN)':>10} | {'O-D(mat)':>10} | {'D-D Diff':>10}"
    )
    logger.info("-" * 110)

    total_original = 0.0
    total_distilled = 0.0
    total_matrix = 0.0
    total_steps = 0
    original_wins = 0
    distilled_wins = 0
    matrix_wins = 0
    ties = 0

    for ep in completed_episodes:
        motion_id = ep["motion_id"]
        steps = ep["steps"]
        orig_reward = ep["original_reward"]
        dist_reward = ep["distilled_reward"]
        mat_reward = ep["matrix_reward"]
        diff_od = orig_reward - dist_reward
        diff_om = orig_reward - mat_reward
        diff_dm = dist_reward - mat_reward

        best = max(orig_reward, dist_reward, mat_reward)
        num_best = [orig_reward, dist_reward, mat_reward].count(best)
        if (
            num_best > 1
            and abs(orig_reward - dist_reward) < 1e-6
            and abs(orig_reward - mat_reward) < 1e-6
        ):
            winner = "TIE"
            ties += 1
        elif orig_reward == best:
            winner = "Original"
            original_wins += 1
        elif dist_reward == best:
            winner = "Distilled (NN)"
            distilled_wins += 1
        else:
            winner = "Distilled (matrix)"
            matrix_wins += 1

        logger.info(
            f"{motion_id:>10} | {steps:>6} | {orig_reward:>12.2f} | {dist_reward:>12.2f} | {mat_reward:>12.2f} | {diff_od:>+10.2f} | {diff_om:>+10.2f} | {diff_dm:>+10.2f}"
        )
        total_original += orig_reward
        total_distilled += dist_reward
        total_matrix += mat_reward
        total_steps += steps

    logger.info("-" * 110)
    logger.info(
        f"{'TOTAL':>10} | {total_steps:>6} | {total_original:>12.2f} | {total_distilled:>12.2f} | {total_matrix:>12.2f} | {total_original - total_distilled:>+10.2f} | {total_original - total_matrix:>+10.2f} | {total_distilled - total_matrix:>+10.2f}"
    )
    logger.info("")
    logger.info(f"Number of episodes: {len(completed_episodes)}")
    logger.info(f"Original policy wins:         {original_wins}")
    logger.info(f"Distilled (NN) policy wins:   {distilled_wins}")
    logger.info(f"Distilled (matrix) policy wins: {matrix_wins}")
    logger.info(f"Ties: {ties}")
    logger.info("")

    # Distilled NN vs Matrix K agreement check
    logger.info("--- Distilled (NN) vs Distilled (matrix) Agreement ---")
    max_diff_dm = 0.0
    for ep in completed_episodes:
        d = abs(ep["distilled_reward"] - ep["matrix_reward"])
        if d > max_diff_dm:
            max_diff_dm = d
    logger.info(
        f"Max absolute reward difference (Distilled (NN) - Distilled (matrix)): {max_diff_dm:.6f}"
    )
    if max_diff_dm < 1e-3:
        logger.info(
            "Distilled (NN) and Distilled (matrix) policies produce nearly identical rewards (as expected)."
        )
    logger.info("")

    if total_steps > 0:
        logger.info(
            f"Average reward per step (Original):     {total_original / total_steps:.4f}"
        )
        logger.info(
            f"Average reward per step (Distilled (NN)):     {total_distilled / total_steps:.4f}"
        )
        logger.info(
            f"Average reward per step (Distilled (matrix)): {total_matrix / total_steps:.4f}"
        )

    # Reward model estimation summary
    has_reward_pred = (
        completed_episodes and "reward_pred_original" in completed_episodes[0]
    )
    if has_reward_pred and reward_estimation_stats is not None:
        logger.info("")
        logger.info("=" * 110)
        logger.info("  REWARD MODEL ESTIMATION")
        logger.info("=" * 110)
        logger.info("")

        # Per-step error stats
        mae = reward_estimation_stats["mae"]
        rmse = reward_estimation_stats["rmse"]
        logger.info("Per-step error (across all episodes):")
        logger.info(
            f"  MAE  — Original: {mae[0]:.6f}  Distilled: {mae[1]:.6f}  Matrix K: {mae[2]:.6f}"
        )
        logger.info(
            f"  RMSE — Original: {rmse[0]:.6f}  Distilled: {rmse[1]:.6f}  Matrix K: {rmse[2]:.6f}"
        )
        logger.info("")

        # Per-episode actual vs predicted
        logger.info(
            f"{'Motion ID':>10} | {'Steps':>6} | "
            f"{'Orig Actual':>12} {'Orig Pred':>12} {'Orig Err%':>10} | "
            f"{'D(NN) Actual':>12} {'D(NN) Pred':>12} {'D(NN) Err%':>10} | "
            f"{'D(mat) Actual':>13} {'D(mat) Pred':>12} {'D(mat) Err%':>11}"
        )
        logger.info("-" * 145)

        total_pred_original = 0.0
        total_pred_distilled = 0.0
        total_pred_matrix = 0.0
        for ep in completed_episodes:
            mid = ep["motion_id"]
            steps = ep["steps"]
            o_act, o_pred = ep["original_reward"], ep["reward_pred_original"]
            d_act, d_pred = ep["distilled_reward"], ep["reward_pred_distilled"]
            m_act, m_pred = ep["matrix_reward"], ep["reward_pred_matrix"]
            o_pct = ((o_pred - o_act) / abs(o_act) * 100) if abs(o_act) > 1e-6 else 0.0
            d_pct = ((d_pred - d_act) / abs(d_act) * 100) if abs(d_act) > 1e-6 else 0.0
            m_pct = ((m_pred - m_act) / abs(m_act) * 100) if abs(m_act) > 1e-6 else 0.0
            total_pred_original += o_pred
            total_pred_distilled += d_pred
            total_pred_matrix += m_pred
            logger.info(
                f"{mid:>10} | {steps:>6} | "
                f"{o_act:>12.2f} {o_pred:>12.2f} {o_pct:>+9.1f}% | "
                f"{d_act:>12.2f} {d_pred:>12.2f} {d_pct:>+9.1f}% | "
                f"{m_act:>12.2f} {m_pred:>12.2f} {m_pct:>+9.1f}%"
            )

        logger.info("-" * 145)
        o_pct_tot = (
            ((total_pred_original - total_original) / abs(total_original) * 100)
            if abs(total_original) > 1e-6
            else 0.0
        )
        d_pct_tot = (
            ((total_pred_distilled - total_distilled) / abs(total_distilled) * 100)
            if abs(total_distilled) > 1e-6
            else 0.0
        )
        m_pct_tot = (
            ((total_pred_matrix - total_matrix) / abs(total_matrix) * 100)
            if abs(total_matrix) > 1e-6
            else 0.0
        )
        logger.info(
            f"{'TOTAL':>10} | {total_steps:>6} | "
            f"{total_original:>12.2f} {total_pred_original:>12.2f} {o_pct_tot:>+9.1f}% | "
            f"{total_distilled:>12.2f} {total_pred_distilled:>12.2f} {d_pct_tot:>+9.1f}% | "
            f"{total_matrix:>12.2f} {total_pred_matrix:>12.2f} {m_pct_tot:>+9.1f}%"
        )
        logger.info("")

    rewards_path = output_dir / "reward_comparison.npz"
    np.savez(
        rewards_path,
        episodes=completed_episodes,
        total_original=total_original,
        total_distilled=total_distilled,
        total_matrix=total_matrix,
        original_wins=original_wins,
        distilled_wins=distilled_wins,
        matrix_wins=matrix_wins,
    )
    logger.info(f"\nReward data saved to: {rewards_path}")


def save_results(
    obs_list,
    actions_original_list,
    actions_distilled_list,
    actions_matrix_list,
    reset_flags_list,
    output_dir,
):
    """Save observations, actions (all three policies), and reset flags to disk."""
    obs_path = output_dir / "observations.npz"
    actions_original_path = output_dir / "actions_original.npy"
    actions_distilled_path = output_dir / "actions_distilled.npy"
    actions_matrix_path = output_dir / "actions_matrix.npy"
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
    save_actions_matrix = np.array(actions_matrix_list)

    logger.info(
        f"Saving original policy actions to {actions_original_path} with shape {save_actions_original.shape}"
    )
    np.save(actions_original_path, save_actions_original)
    logger.info(
        f"Saving distilled NN policy actions to {actions_distilled_path} with shape {save_actions_distilled.shape}"
    )
    np.save(actions_distilled_path, save_actions_distilled)
    logger.info(
        f"Saving matrix K policy actions to {actions_matrix_path} with shape {save_actions_matrix.shape}"
    )
    np.save(actions_matrix_path, save_actions_matrix)

    save_reset_flags = np.array(reset_flags_list)
    logger.info(
        f"Saving reset flags to {reset_flags_path} with shape {save_reset_flags.shape}"
    )
    logger.info(f"  Total resets: {save_reset_flags.sum()}")
    np.save(reset_flags_path, save_reset_flags)


def save_results_h5(
    obs_list,
    commands_list,
    base_height_list,
    base_position_list,
    base_orientation_list,
    hand_positions_list,
    hand_velocities_list,
    actions_all_list,
    actions_original_list,
    actions_distilled_list,
    actions_matrix_list,
    reset_flags_list,
    output_dir,
    config,
):
    """Save all experiment data (commands, observations, actions) to HDF5 format."""
    h5_path = output_dir / "experiment_data.h5"
    logger.info(f"Saving experiment data to HDF5: {h5_path}")

    # Convert lists to numpy arrays
    commands_array = np.array(commands_list)
    base_height_array = np.array(base_height_list)
    base_position_array = np.array(base_position_list)
    base_orientation_array = np.array(base_orientation_list)
    hand_positions_array = np.array(hand_positions_list)
    hand_velocities_array = np.array(hand_velocities_list)
    actions_all_array = np.array(actions_all_list)
    actions_original_array = np.array(actions_original_list)
    actions_distilled_array = np.array(actions_distilled_list)
    actions_matrix_array = np.array(actions_matrix_list)
    reset_flags_array = np.array(reset_flags_list)

    # Process observations
    obs_arrays = {}
    obs_keys = obs_list[0].keys()
    for k in obs_keys:
        obs_arrays[k] = np.array([x[k] for x in obs_list])

    # Extract metadata
    num_steps = len(obs_list)
    num_envs = commands_array.shape[1] if len(commands_array.shape) > 1 else 0
    num_triplets = num_envs // 3
    dt = config.get("dt", 0.02)

    with h5py.File(h5_path, "w") as f:
        # Metadata group
        metadata = f.create_group("metadata")
        metadata.attrs["num_envs"] = num_envs
        metadata.attrs["num_steps"] = num_steps
        metadata.attrs["num_triplets"] = num_triplets
        metadata.attrs["dt"] = dt

        # Commands dataset: (num_steps, num_envs, cmd_dim)
        f.create_dataset("commands", data=commands_array, compression="gzip")
        logger.info(f"  commands: {commands_array.shape}")

        # Base height dataset: (num_steps, num_envs)
        f.create_dataset("base_height", data=base_height_array, compression="gzip")
        logger.info(f"  base_height: {base_height_array.shape}")

        # Base pose group: position (XYZ) and orientation (quaternion xyzw)
        base_pose_group = f.create_group("base_pose")
        base_pose_group.create_dataset(
            "position", data=base_position_array, compression="gzip"
        )
        base_pose_group.create_dataset(
            "orientation", data=base_orientation_array, compression="gzip"
        )
        logger.info(f"  base_pose/position: {base_position_array.shape}")
        logger.info(f"  base_pose/orientation: {base_orientation_array.shape}")

        # Hand tracking group
        hand_group = f.create_group("hand_tracking")
        hand_group.create_dataset(
            "positions", data=hand_positions_array, compression="gzip"
        )
        hand_group.create_dataset(
            "velocities", data=hand_velocities_array, compression="gzip"
        )
        logger.info(f"  hand_tracking/positions: {hand_positions_array.shape}")
        logger.info(f"  hand_tracking/velocities: {hand_velocities_array.shape}")

        # Observations group
        obs_group = f.create_group("observations")
        for k, v in obs_arrays.items():
            obs_group.create_dataset(k, data=v, compression="gzip")
            logger.info(f"  observations/{k}: {v.shape}")

        # Actions group
        actions_group = f.create_group("actions")
        actions_group.create_dataset("all", data=actions_all_array, compression="gzip")
        actions_group.create_dataset(
            "original", data=actions_original_array, compression="gzip"
        )
        actions_group.create_dataset(
            "distilled", data=actions_distilled_array, compression="gzip"
        )
        actions_group.create_dataset(
            "matrix", data=actions_matrix_array, compression="gzip"
        )
        logger.info(f"  actions/all: {actions_all_array.shape}")
        logger.info(f"  actions/original: {actions_original_array.shape}")
        logger.info(f"  actions/distilled: {actions_distilled_array.shape}")
        logger.info(f"  actions/matrix: {actions_matrix_array.shape}")

        # Episode info group
        episode_group = f.create_group("episode_info")
        episode_group.create_dataset(
            "reset_flags", data=reset_flags_array, compression="gzip"
        )
        logger.info(f"  episode_info/reset_flags: {reset_flags_array.shape}")

    logger.info(f"HDF5 file saved successfully: {h5_path}")
    return h5_path


def create_output_dir():
    """Create timestamped output directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(EVAL_OUTPUT_BASE_DIR) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


@hydra.main(config_path="config", config_name="base_eval", version_base="1.1")
def main(override_config: OmegaConf):
    """Entry point: run triplet policy comparison and save results."""
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    output_dir = create_output_dir()
    setup_logging(output_dir)
    os.chdir(hydra.utils.get_original_cwd())

    logger.info("=" * 60)
    logger.info("  TRIPLET POLICY COMPARISON EVALUATION")
    logger.info("  Original vs Distilled (NN) vs Distilled (matrix) Policy")
    logger.info("=" * 60)
    logger.info(f"Output directory: {output_dir}")

    config, checkpoint = load_config(override_config)
    config_save_path = output_dir / "config.yaml"
    with open(config_save_path, "w") as f:
        OmegaConf.save(config, f)
    logger.info(f"Config saved to: {config_save_path}")

    (
        env,
        algo,
        phi_nn,
        K_nn,
        K_matrix,
        motion_pool_size,
        body_idx,
        state_type,
        reward_nn,
    ) = setup_simulation(config, checkpoint, device)

    (
        obs_list,
        commands_list,
        base_height_list,
        base_position_list,
        base_orientation_list,
        hand_positions_list,
        hand_velocities_list,
        actions_all_list,
        actions_original_list,
        actions_distilled_list,
        actions_matrix_list,
        reset_flags_list,
        completed_episodes,
        reward_estimation_stats,
    ) = collect_data(
        env,
        algo,
        phi_nn,
        K_nn,
        K_matrix,
        config,
        motion_pool_size,
        body_idx,
        state_type,
        reward_nn,
        device,
    )

    save_results(
        obs_list,
        actions_original_list,
        actions_distilled_list,
        actions_matrix_list,
        reset_flags_list,
        output_dir,
    )

    # Save all data to HDF5 format
    save_results_h5(
        obs_list,
        commands_list,
        base_height_list,
        base_position_list,
        base_orientation_list,
        hand_positions_list,
        hand_velocities_list,
        actions_all_list,
        actions_original_list,
        actions_distilled_list,
        actions_matrix_list,
        reset_flags_list,
        output_dir,
        config,
    )

    print_reward_summary(completed_episodes, output_dir, reward_estimation_stats)

    logger.info("=" * 60)
    logger.info("  EVALUATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
