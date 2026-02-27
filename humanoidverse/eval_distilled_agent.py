"""
Triplet policy comparison: runs three policies side-by-side.
Env index % 3 == 0: original policy
Env index % 3 == 1: Distilled (NN) policy (u = K_nn(phi(x)))
Env index % 3 == 2: Distilled (matrix) policy (u = K @ phi(x), extracted from K_nn)
"""

# =============================================================================
# Imports
# =============================================================================

import os
import sys
import types
import logging
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

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
from humanoid_linear_distill.utils.networks import SingleLayerNet
from humanoid_linear_distill.export_models import (
    load_and_build,
    get_input_dim,
    get_output_dim,
    describe_architecture,
)

# =============================================================================
# Constants
# =============================================================================

ORIGINAL_POLICY_COLOR = (0.0, 0.5, 1.0)  # Blue
DISTILLED_POLICY_COLOR = (1.0, 0.3, 0.0)  # Orange
MATRIX_POLICY_COLOR = (0.0, 0.8, 0.2)  # Green
MARKER_HEIGHT_OFFSET = 1.5  # meters above robot
MARKER_RADIUS = 0.15

EVAL_OUTPUT_BASE_DIR = "/home/zixin/Dev/FALCON/logs_eval/linear_distill"

# =============================================================================
# Dataclasses
# =============================================================================


@dataclass
class TrajectoryData:
    """Per-step trajectory state recorded from the simulation."""

    obs: List[Dict[str, np.ndarray]] = field(default_factory=list)
    commands: List[np.ndarray] = field(default_factory=list)
    base_height: List[np.ndarray] = field(default_factory=list)
    base_position: List[np.ndarray] = field(default_factory=list)
    base_orientation: List[np.ndarray] = field(default_factory=list)


@dataclass
class HandTrackingData:
    """Per-step hand positions and velocities for all envs."""

    positions: List[np.ndarray] = field(default_factory=list)
    velocities: List[np.ndarray] = field(default_factory=list)


@dataclass
class ActionData:
    """Per-step actions for each of the three policy types."""

    all: List[np.ndarray] = field(default_factory=list)
    original: List[np.ndarray] = field(default_factory=list)
    distilled: List[np.ndarray] = field(default_factory=list)
    matrix: List[np.ndarray] = field(default_factory=list)


@dataclass
class EpisodeData:
    """Episode-level tracking: reset flags and completed episode records."""

    reset_flags: List[np.ndarray] = field(default_factory=list)
    completed_episodes: List[Dict] = field(default_factory=list)
    reward_estimation_stats: Optional[Dict] = None


@dataclass
class CollectionResult:
    """Complete output of collect_data(), grouping all recorded data."""

    trajectory: TrajectoryData
    hands: HandTrackingData
    actions: ActionData
    episodes: EpisodeData


@dataclass
class RewardSummaryStats:
    """Aggregated reward statistics across all completed episodes."""

    total_original: float = 0.0
    total_distilled: float = 0.0
    total_matrix: float = 0.0
    total_steps: int = 0
    original_wins: int = 0
    distilled_wins: int = 0
    matrix_wins: int = 0
    ties: int = 0


# =============================================================================
# Logging and Configuration
# =============================================================================


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


# =============================================================================
# Policy Loading
# =============================================================================


def load_distilled_policy(config, device):
    """Load distilled policy networks (phi_nn and K_nn).

    Auto-detects network architectures from saved state_dict keys using
    load_and_build() — no config-based dimension assumptions needed.
    Returns state_type ("actor_obs" or "state_206") indicating which obs to feed.
    Override with config.distilled_state_type = "actor_obs" | "state_206" | "auto".
    """
    distilled_model_dir = Path(config.distilled_model_dir)

    phi_path = distilled_model_dir / "phi_nn.pth"
    K_path = distilled_model_dir / "K_nn.pth"
    if not phi_path.exists():
        logger.error(f"phi_nn.pth not found at: {phi_path}")
        sys.exit(1)
    if not K_path.exists():
        logger.error(f"K_nn.pth not found at: {K_path}")
        sys.exit(1)

    # Auto-detect phi_nn architecture from state_dict
    phi_nn, phi_sd = load_and_build(str(distilled_model_dir), "phi_nn", device)
    phi_nn = phi_nn.to(device)
    phi_nn.eval()
    input_size = get_input_dim(phi_sd)
    latent_dim = get_output_dim(phi_sd)

    # Load K_nn (always SingleLayerNet — need .linear1.weight for K_matrix)
    K_sd = torch.load(K_path, map_location=device, weights_only=True)
    action_dim = K_sd["linear1.weight"].shape[0]
    K_nn = SingleLayerNet(latent_dim, action_dim, bias=False).to(device)
    K_nn.load_state_dict(K_sd)
    K_nn.eval()

    # Extract K matrix from the single linear layer of K_nn
    K_matrix = K_nn.linear1.weight.data.clone()  # (action_dim, latent_dim)

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
    logger.info(f"  latent_dim={latent_dim}")
    logger.info(f"  phi_nn architecture: {describe_architecture(phi_sd)}")
    logger.info(f"  K matrix shape: {tuple(K_matrix.shape)}")

    # Load reward_nn if available (optional)
    reward_path = distilled_model_dir / "reward_nn.pth"
    reward_nn = None
    if reward_path.exists():
        reward_nn, reward_sd = load_and_build(
            str(distilled_model_dir), "reward_nn", device
        )
        reward_nn = reward_nn.to(device)
        reward_nn.eval()
        logger.info(f"  reward_nn architecture: {describe_architecture(reward_sd)}")
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


# =============================================================================
# Triplet Environment Wrapper
# =============================================================================


class TripletEnvWrapper:
    """Wraps a HumanoidVerse env to enforce synchronized triplet behavior.

    Each group of 3 consecutive environment indices forms a "triplet":
      - index % 3 == 0: leader (original policy)
      - index % 3 == 1: follower (distilled NN policy)
      - index % 3 == 2: follower (distilled matrix policy)

    The wrapper ensures that:
      1. Resets are synchronized: if any env in a triplet resets, all three reset.
      2. Commands, motion IDs, and timing buffers are copied from leader to followers.
      3. Environment origins are arranged so triplets are visually grouped.
    """

    def __init__(self, env, config, device):
        self.env = env
        self.device = device
        self.num_envs = env.num_envs
        self.num_triplets = self.num_envs // 3

        env.reset_flag = torch.zeros(self.num_envs, dtype=torch.bool, device=device)
        env.policy_type = torch.zeros(self.num_envs, dtype=torch.long, device=device)
        env.policy_type[1::3] = 1  # Distilled (NN) policy
        env.policy_type[2::3] = 2  # Distilled (matrix) policy

        self._patch_reset()
        self.arrange_origins(config)

    def _patch_reset(self):
        """Monkey-patch env.reset_envs_idx so resets are triplet-synchronized."""
        original_reset_envs_idx = self.env.reset_envs_idx
        num_envs = self.num_envs
        env = self.env

        def custom_reset_envs_idx(
            self_env, env_ids, target_states=None, target_buf=None
        ):
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
                        if hasattr(self_env, "motion_start_times"):
                            self_env.motion_start_times[follower_idx] = (
                                self_env.motion_start_times[leader_idx]
                            )
                        self_env.episode_length_buf[follower_idx] = (
                            self_env.episode_length_buf[leader_idx]
                        )
                        if hasattr(self_env, "episode_motion_length"):
                            self_env.episode_motion_length[follower_idx] = (
                                self_env.episode_motion_length[leader_idx]
                            )
                        if hasattr(self_env, "motion_times"):
                            self_env.motion_times[follower_idx] = self_env.motion_times[
                                leader_idx
                            ]
                        self_env.motion_ids[follower_idx] = self_env.motion_ids[
                            leader_idx
                        ]
                        self_env.motion_len[follower_idx] = self_env.motion_len[
                            leader_idx
                        ]

            self_env.reset_flag[triplet_env_ids] = True
            if len(triplet_env_ids) > 0:
                logger.debug(f"Triplet reset: {triplet_env_ids.tolist()}")

        env.reset_envs_idx = types.MethodType(custom_reset_envs_idx, env)

    def arrange_origins(self, config):
        """Position triplet robots side-by-side: original (left), distilled NN (center), matrix K (right)."""
        env = self.env
        num_triplets = self.num_triplets
        env_spacing = config.get("env_spacing", 3.0)
        triplet_offset = config.get("pair_offset", 1.5)

        num_cols = int(np.floor(np.sqrt(num_triplets)))
        num_rows = int(np.ceil(num_triplets / num_cols))
        logger.info(
            f"Arranging {num_triplets} triplets in a {num_rows}x{num_cols} grid"
        )
        logger.info(
            f"  Environment spacing: {env_spacing}m, Triplet offset: {triplet_offset}m"
        )
        logger.info(
            "  LEFT = Original policy, CENTER = Distilled (NN) policy, RIGHT = Distilled (matrix) policy"
        )

        new_origins = torch.zeros(self.num_envs, 3, device=self.device)
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

    def sync_commands(self):
        """Copy commands from leader (index 0 in triplet) to the other two."""
        self.env.commands[1::3] = self.env.commands[0::3].clone()
        self.env.commands[2::3] = self.env.commands[0::3].clone()

    def sync_motion_ids(self):
        """Copy motion IDs from leader to the other two in each triplet."""
        self.env.motion_ids[1::3] = self.env.motion_ids[0::3].clone()
        self.env.motion_ids[2::3] = self.env.motion_ids[0::3].clone()
        self.env.motion_len[1::3] = self.env.motion_len[0::3].clone()
        self.env.motion_len[2::3] = self.env.motion_len[0::3].clone()

    def sync_motion_times(self):
        """Copy motion timing buffers from leader to the other two in each triplet."""
        env = self.env
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

    def sync_all(self):
        """Convenience: sync commands + motion times. Called at top of each step."""
        self.sync_commands()
        self.sync_motion_times()

    def handle_timeouts(self, infos, motion_pool_size, step):
        """On timeout, assign new motion to all three envs in affected triplets."""
        env = self.env
        time_outs = infos.get(
            "time_outs",
            torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
        )
        timed_out_env_ids = torch.where(time_outs)[0]
        if len(timed_out_env_ids) == 0:
            return

        triplet_indices = timed_out_env_ids // 3
        unique_triplets = torch.unique(triplet_indices)
        new_motion_ids = torch.randint(
            0, motion_pool_size, (len(unique_triplets),), device=self.device
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


# =============================================================================
# State Extraction and Recording
# =============================================================================


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


def record_step_state(env, left_hand_idx, right_hand_idx):
    """Capture base pose and hand kinematics from simulator after physics step.

    Returns:
        base_height, base_position, base_orientation, hand_positions, hand_velocities
    """
    sim = env.simulator
    base_height = sim.robot_root_states[:, 2].cpu().numpy().copy()
    base_position = sim.robot_root_states[:, 0:3].cpu().numpy().copy()
    base_orientation = sim.robot_root_states[:, 3:7].cpu().numpy().copy()

    left_hand_pos = sim._rigid_body_pos[:, left_hand_idx, :].cpu().numpy()
    right_hand_pos = sim._rigid_body_pos[:, right_hand_idx, :].cpu().numpy()
    hand_positions = np.stack([left_hand_pos, right_hand_pos], axis=1).copy()

    left_hand_vel = sim._rigid_body_vel[:, left_hand_idx, :].cpu().numpy()
    right_hand_vel = sim._rigid_body_vel[:, right_hand_idx, :].cpu().numpy()
    hand_velocities = np.stack([left_hand_vel, right_hand_vel], axis=1).copy()

    return base_height, base_position, base_orientation, hand_positions, hand_velocities


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


# =============================================================================
# Action Inference
# =============================================================================


def _build_distilled_input(obs_dict, state_type, env, body_idx):
    """Build input tensor for distilled policies based on state_type.

    Returns actor_obs (575-dim) if state_type == "actor_obs",
    otherwise cat(critic_obs, extended_state) for 206-dim.
    """
    if state_type == "actor_obs":
        return obs_dict["actor_obs"]
    critic_obs = obs_dict["critic_obs"]
    extended_state = extract_extended_state(env, body_idx)
    return torch.cat([critic_obs, extended_state], dim=1)


def compute_triplet_actions(
    actor_obs, distilled_input, algo, phi_nn, K_nn, K_matrix, num_envs, device
):
    """Compute actions for all three policy types.

    Returns:
        actions: (num_envs, action_dim) combined action tensor.
    """
    actions = torch.zeros(num_envs, K_matrix.shape[0], device=device)

    # Original policy for i%3==0 (uses 575-dim actor_obs)
    orig_indices = torch.arange(0, num_envs, 3, device=device)
    if hasattr(algo, "act_inference"):
        actions[orig_indices] = algo.act_inference(actor_obs[orig_indices])
    else:
        actions[orig_indices] = algo.actor.act_inference(actor_obs[orig_indices])

    # Distilled (NN) policy for i%3==1: u = K_nn(phi_nn(x))
    dist_nn_indices = torch.arange(1, num_envs, 3, device=device)
    phi_x_nn = phi_nn(distilled_input[dist_nn_indices])
    actions[dist_nn_indices] = K_nn(phi_x_nn)

    # Distilled (matrix) policy for i%3==2: u = K @ phi_nn(x)
    mat_indices = torch.arange(2, num_envs, 3, device=device)
    phi_x_mat = phi_nn(distilled_input[mat_indices])
    actions[mat_indices] = phi_x_mat @ K_matrix.T

    return actions


# =============================================================================
# Reward Tracking
# =============================================================================


class EpisodeRewardTracker:
    """Tracks per-triplet cumulative rewards and emits completed episode records.

    Handles both the ground-truth rewards from the simulator and the optional
    reward_nn model predictions.
    """

    def __init__(self, num_triplets, reward_nn, device):
        self.num_triplets = num_triplets
        self.reward_nn = reward_nn
        self.device = device
        self.completed_episodes = []

        self.current_rewards_original = torch.zeros(num_triplets, device=device)
        self.current_rewards_distilled = torch.zeros(num_triplets, device=device)
        self.current_rewards_matrix = torch.zeros(num_triplets, device=device)
        self.current_steps = torch.zeros(num_triplets, dtype=torch.long, device=device)
        self.current_motion_ids = torch.zeros(
            num_triplets, dtype=torch.long, device=device
        )

        if reward_nn is not None:
            self.current_pred_original = torch.zeros(num_triplets, device=device)
            self.current_pred_distilled = torch.zeros(num_triplets, device=device)
            self.current_pred_matrix = torch.zeros(num_triplets, device=device)
            self.reward_abs_error_sum = torch.zeros(3, device=device)
            self.reward_sq_error_sum = torch.zeros(3, device=device)
            self.reward_error_count = torch.zeros(3, dtype=torch.long, device=device)

    def accumulate(self, rewards, distilled_input, phi_nn):
        """Add one step's rewards to the running totals for each triplet."""
        if isinstance(rewards, dict):
            total_rewards = sum(rewards.values())
        else:
            total_rewards = rewards

        rewards_original = total_rewards[0::3]
        rewards_distilled = total_rewards[1::3]
        rewards_matrix = total_rewards[2::3]
        self.current_rewards_original += rewards_original
        self.current_rewards_distilled += rewards_distilled
        self.current_rewards_matrix += rewards_matrix
        self.current_steps += 1

        if self.reward_nn is not None:
            z_all = phi_nn(distilled_input)
            predicted_rewards = self.reward_nn(z_all).squeeze(-1)
            pred_original = predicted_rewards[0::3]
            pred_distilled = predicted_rewards[1::3]
            pred_matrix = predicted_rewards[2::3]
            self.current_pred_original += pred_original
            self.current_pred_distilled += pred_distilled
            self.current_pred_matrix += pred_matrix
            # Per-step error tracking
            self.reward_abs_error_sum[0] += (
                (pred_original - rewards_original).abs().sum()
            )
            self.reward_abs_error_sum[1] += (
                (pred_distilled - rewards_distilled).abs().sum()
            )
            self.reward_abs_error_sum[2] += (pred_matrix - rewards_matrix).abs().sum()
            self.reward_sq_error_sum[0] += (
                (pred_original - rewards_original) ** 2
            ).sum()
            self.reward_sq_error_sum[1] += (
                (pred_distilled - rewards_distilled) ** 2
            ).sum()
            self.reward_sq_error_sum[2] += ((pred_matrix - rewards_matrix) ** 2).sum()
            self.reward_error_count += self.num_triplets

    def finalize_episodes(self, completed_triplet_ids):
        """Record completed episodes and reset their accumulators."""
        for triplet_id in completed_triplet_ids:
            tid = triplet_id.item()
            ep_data = {
                "motion_id": self.current_motion_ids[tid].item(),
                "original_reward": self.current_rewards_original[tid].item(),
                "distilled_reward": self.current_rewards_distilled[tid].item(),
                "matrix_reward": self.current_rewards_matrix[tid].item(),
                "steps": self.current_steps[tid].item(),
            }
            if self.reward_nn is not None:
                ep_data["reward_pred_original"] = self.current_pred_original[tid].item()
                ep_data["reward_pred_distilled"] = self.current_pred_distilled[
                    tid
                ].item()
                ep_data["reward_pred_matrix"] = self.current_pred_matrix[tid].item()
                self.current_pred_original[tid] = 0
                self.current_pred_distilled[tid] = 0
                self.current_pred_matrix[tid] = 0
            self.completed_episodes.append(ep_data)
            self.current_rewards_original[tid] = 0
            self.current_rewards_distilled[tid] = 0
            self.current_rewards_matrix[tid] = 0
            self.current_steps[tid] = 0

    def flush_incomplete(self, min_steps=10):
        """Record in-progress episodes at end of collection, skipping very short ones."""
        for tid in range(self.num_triplets):
            if self.current_steps[tid] >= min_steps:
                ep_data = {
                    "motion_id": self.current_motion_ids[tid].item(),
                    "original_reward": self.current_rewards_original[tid].item(),
                    "distilled_reward": self.current_rewards_distilled[tid].item(),
                    "matrix_reward": self.current_rewards_matrix[tid].item(),
                    "steps": self.current_steps[tid].item(),
                }
                if self.reward_nn is not None:
                    ep_data["reward_pred_original"] = self.current_pred_original[
                        tid
                    ].item()
                    ep_data["reward_pred_distilled"] = self.current_pred_distilled[
                        tid
                    ].item()
                    ep_data["reward_pred_matrix"] = self.current_pred_matrix[tid].item()
                self.completed_episodes.append(ep_data)

    def get_estimation_stats(self):
        """Compute per-step MAE/RMSE for reward model estimation."""
        if self.reward_nn is None:
            return None
        count = self.reward_error_count.float().clamp(min=1.0)
        return {
            "mae": (self.reward_abs_error_sum / count).cpu().tolist(),
            "rmse": ((self.reward_sq_error_sum / count).sqrt()).cpu().tolist(),
        }


def _detect_completed_triplets(env, infos, num_envs, device):
    """Identify which triplets completed this step (reset or timeout).

    Returns:
        Tensor of triplet indices that completed.
    """
    reset_flags_0 = env.reset_flag[0::3]
    reset_flags_1 = env.reset_flag[1::3]
    reset_flags_2 = env.reset_flag[2::3]
    completed = reset_flags_0 | reset_flags_1 | reset_flags_2

    time_outs = infos.get(
        "time_outs", torch.zeros(num_envs, dtype=torch.bool, device=device)
    )
    timeout_triplets = time_outs[0::3] | time_outs[1::3] | time_outs[2::3]
    completed = completed | timeout_triplets

    return torch.where(completed)[0]


# =============================================================================
# Simulation Setup and Data Collection
# =============================================================================


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

    triplet_wrapper = TripletEnvWrapper(env, config, device)
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
        triplet_wrapper,
    )


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
    triplet,
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

    trajectory = TrajectoryData()
    hands = HandTrackingData()
    actions = ActionData()
    reset_flags_list = []
    tracker = EpisodeRewardTracker(num_triplets, reward_nn, device)

    left_hand_idx = env.body_names.index("left_rubber_hand")
    right_hand_idx = env.body_names.index("right_rubber_hand")
    logger.info(f"Hand link indices - Left: {left_hand_idx}, Right: {right_hand_idx}")

    obs_dict = env.reset_all()
    triplet.sync_commands()
    triplet.sync_motion_ids()
    triplet.sync_motion_times()
    tracker.current_motion_ids[:] = env.motion_ids[0::3]

    with torch.inference_mode():
        for step in tqdm(range(num_steps), desc="Collecting Data (Triplet)"):
            env.reset_flag[:] = False
            triplet.sync_all()

            # Record pre-step observations and commands
            trajectory.obs.append({k: v.cpu().numpy() for k, v in obs_dict.items()})
            trajectory.commands.append(env.commands.cpu().numpy().copy())

            # Compute actions for all three policies
            actor_obs = obs_dict["actor_obs"]
            distilled_input = _build_distilled_input(
                obs_dict, state_type, env, body_idx
            )
            step_actions = compute_triplet_actions(
                actor_obs,
                distilled_input,
                algo,
                phi_nn,
                K_nn,
                K_matrix,
                num_envs,
                device,
            )

            # Record per-policy actions
            actions.original.append(step_actions[0::3].cpu().numpy())
            actions.distilled.append(step_actions[1::3].cpu().numpy())
            actions.matrix.append(step_actions[2::3].cpu().numpy())
            actions.all.append(step_actions.cpu().numpy().copy())

            # Step simulation
            actor_state = {"actions": step_actions}
            obs_dict, rewards, dones, infos = env.step(actor_state)

            # Record post-step state
            bh, bp, bo, hp, hv = record_step_state(env, left_hand_idx, right_hand_idx)
            trajectory.base_height.append(bh)
            trajectory.base_position.append(bp)
            trajectory.base_orientation.append(bo)
            hands.positions.append(hp)
            hands.velocities.append(hv)

            # Reward tracking
            tracker.accumulate(rewards, distilled_input, phi_nn)

            # Episode completion
            completed_triplet_ids = _detect_completed_triplets(
                env, infos, num_envs, device
            )
            tracker.finalize_episodes(completed_triplet_ids)

            # Handle timeouts and bookkeeping
            triplet.handle_timeouts(infos, motion_pool_size, step)
            tracker.current_motion_ids[:] = env.motion_ids[0::3]
            draw_policy_markers(env)
            reset_flags_list.append(env.reset_flag.cpu().numpy().copy())

    tracker.flush_incomplete()

    return CollectionResult(
        trajectory=trajectory,
        hands=hands,
        actions=actions,
        episodes=EpisodeData(
            reset_flags=reset_flags_list,
            completed_episodes=tracker.completed_episodes,
            reward_estimation_stats=tracker.get_estimation_stats(),
        ),
    )


# =============================================================================
# Reporting
# =============================================================================


def _print_episode_reward_table(completed_episodes):
    """Print per-episode reward comparison table. Returns aggregated stats."""
    logger.info("")
    logger.info("=" * 110)
    logger.info("  REWARD SUMMARY: Original vs Distilled (NN) vs Distilled (matrix)")
    logger.info("=" * 110)
    logger.info("")
    logger.info(
        f"{'Motion ID':>10} | {'Steps':>6} | {'Original':>12} | {'Dist (NN)':>12} | {'Dist (mat)':>12} | {'O-D(NN)':>10} | {'O-D(mat)':>10} | {'D-D Diff':>10}"
    )
    logger.info("-" * 110)

    stats = RewardSummaryStats()

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
        tol = max(1e-6, abs(best) * 1e-4)
        is_best = [abs(r - best) < tol for r in [orig_reward, dist_reward, mat_reward]]
        if sum(is_best) > 1:
            stats.ties += 1
        elif is_best[0]:
            stats.original_wins += 1
        elif is_best[1]:
            stats.distilled_wins += 1
        else:
            stats.matrix_wins += 1

        logger.info(
            f"{motion_id:>10} | {steps:>6} | {orig_reward:>12.2f} | {dist_reward:>12.2f} | {mat_reward:>12.2f} | {diff_od:>+10.2f} | {diff_om:>+10.2f} | {diff_dm:>+10.2f}"
        )
        stats.total_original += orig_reward
        stats.total_distilled += dist_reward
        stats.total_matrix += mat_reward
        stats.total_steps += steps

    logger.info("-" * 110)
    logger.info(
        f"{'TOTAL':>10} | {stats.total_steps:>6} | {stats.total_original:>12.2f} | {stats.total_distilled:>12.2f} | {stats.total_matrix:>12.2f} | {stats.total_original - stats.total_distilled:>+10.2f} | {stats.total_original - stats.total_matrix:>+10.2f} | {stats.total_distilled - stats.total_matrix:>+10.2f}"
    )

    return stats


def _print_reward_statistics(stats, completed_episodes):
    """Print win counts, NN-vs-matrix agreement, and per-step averages."""
    logger.info("")
    logger.info(f"Number of episodes: {len(completed_episodes)}")
    logger.info(f"Original policy wins:         {stats.original_wins}")
    logger.info(f"Distilled (NN) policy wins:   {stats.distilled_wins}")
    logger.info(f"Distilled (matrix) policy wins: {stats.matrix_wins}")
    logger.info(f"Ties: {stats.ties}")
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

    if stats.total_steps > 0:
        logger.info(
            f"Average reward per step (Original):     {stats.total_original / stats.total_steps:.4f}"
        )
        logger.info(
            f"Average reward per step (Distilled (NN)):     {stats.total_distilled / stats.total_steps:.4f}"
        )
        logger.info(
            f"Average reward per step (Distilled (matrix)): {stats.total_matrix / stats.total_steps:.4f}"
        )


def _print_reward_model_estimation(completed_episodes, stats, reward_estimation_stats):
    """Print reward model (reward_nn) estimation accuracy table."""
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
        ((total_pred_original - stats.total_original) / abs(stats.total_original) * 100)
        if abs(stats.total_original) > 1e-6
        else 0.0
    )
    d_pct_tot = (
        (
            (total_pred_distilled - stats.total_distilled)
            / abs(stats.total_distilled)
            * 100
        )
        if abs(stats.total_distilled) > 1e-6
        else 0.0
    )
    m_pct_tot = (
        ((total_pred_matrix - stats.total_matrix) / abs(stats.total_matrix) * 100)
        if abs(stats.total_matrix) > 1e-6
        else 0.0
    )
    logger.info(
        f"{'TOTAL':>10} | {stats.total_steps:>6} | "
        f"{stats.total_original:>12.2f} {total_pred_original:>12.2f} {o_pct_tot:>+9.1f}% | "
        f"{stats.total_distilled:>12.2f} {total_pred_distilled:>12.2f} {d_pct_tot:>+9.1f}% | "
        f"{stats.total_matrix:>12.2f} {total_pred_matrix:>12.2f} {m_pct_tot:>+9.1f}%"
    )
    logger.info("")


def _save_reward_npz(stats, completed_episodes, output_dir):
    """Save reward comparison data to .npz file."""
    rewards_path = output_dir / "reward_comparison.npz"
    np.savez(
        rewards_path,
        episodes=completed_episodes,
        total_original=stats.total_original,
        total_distilled=stats.total_distilled,
        total_matrix=stats.total_matrix,
        original_wins=stats.original_wins,
        distilled_wins=stats.distilled_wins,
        matrix_wins=stats.matrix_wins,
    )
    logger.info(f"\nReward data saved to: {rewards_path}")


def print_reward_summary(completed_episodes, output_dir, reward_estimation_stats=None):
    """Print and save reward comparison between all three policies."""
    if not completed_episodes:
        logger.warning("No completed episodes to summarize.")
        return

    stats = _print_episode_reward_table(completed_episodes)
    _print_reward_statistics(stats, completed_episodes)

    has_reward_pred = "reward_pred_original" in completed_episodes[0]
    if has_reward_pred and reward_estimation_stats is not None:
        _print_reward_model_estimation(
            completed_episodes, stats, reward_estimation_stats
        )

    _save_reward_npz(stats, completed_episodes, output_dir)


# =============================================================================
# Output / Persistence
# =============================================================================


def save_results(result, output_dir):
    """Save observations, actions (all three policies), and reset flags to disk."""
    obs_path = output_dir / "observations.npz"
    actions_original_path = output_dir / "actions_original.npy"
    actions_distilled_path = output_dir / "actions_distilled.npy"
    actions_matrix_path = output_dir / "actions_matrix.npy"
    reset_flags_path = output_dir / "reset_flags.npy"

    save_obs = {}
    keys = result.trajectory.obs[0].keys()
    for k in keys:
        save_obs[k] = np.array([x[k] for x in result.trajectory.obs])

    logger.info(f"Saving observations to {obs_path}")
    np.savez(obs_path, **save_obs)
    for k, v in save_obs.items():
        logger.info(f"  {k}: {v.shape}")

    save_actions_original = np.array(result.actions.original)
    save_actions_distilled = np.array(result.actions.distilled)
    save_actions_matrix = np.array(result.actions.matrix)

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

    save_reset_flags = np.array(result.episodes.reset_flags)
    logger.info(
        f"Saving reset flags to {reset_flags_path} with shape {save_reset_flags.shape}"
    )
    logger.info(f"  Total resets: {save_reset_flags.sum()}")
    np.save(reset_flags_path, save_reset_flags)


def save_results_h5(result, output_dir, config):
    """Save all experiment data (commands, observations, actions) to HDF5 format."""
    h5_path = output_dir / "experiment_data.h5"
    logger.info(f"Saving experiment data to HDF5: {h5_path}")

    # Convert lists to numpy arrays
    commands_array = np.array(result.trajectory.commands)
    base_height_array = np.array(result.trajectory.base_height)
    base_position_array = np.array(result.trajectory.base_position)
    base_orientation_array = np.array(result.trajectory.base_orientation)
    hand_positions_array = np.array(result.hands.positions)
    hand_velocities_array = np.array(result.hands.velocities)
    actions_all_array = np.array(result.actions.all)
    actions_original_array = np.array(result.actions.original)
    actions_distilled_array = np.array(result.actions.distilled)
    actions_matrix_array = np.array(result.actions.matrix)
    reset_flags_array = np.array(result.episodes.reset_flags)

    # Process observations
    obs_arrays = {}
    obs_keys = result.trajectory.obs[0].keys()
    for k in obs_keys:
        obs_arrays[k] = np.array([x[k] for x in result.trajectory.obs])

    # Extract metadata
    num_steps = len(result.trajectory.obs)
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


# =============================================================================
# Entry Point
# =============================================================================


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
        triplet_wrapper,
    ) = setup_simulation(config, checkpoint, device)

    result = collect_data(
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
        triplet_wrapper,
        device,
    )

    save_results(result, output_dir)
    save_results_h5(result, output_dir, config)
    print_reward_summary(
        result.episodes.completed_episodes,
        output_dir,
        result.episodes.reward_estimation_stats,
    )

    logger.info("=" * 60)
    logger.info("  EVALUATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
