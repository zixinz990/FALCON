import os
import sys
import types
import logging
from pathlib import Path

import isaacgym
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


def setup_logging(output_dir):
    """
    Configures the logging system for both Hydra and console output.
    Sets up file logging to the output directory and console logging based on environment variables.
    """
    if HydraConfig.initialized():
        hydra_log_path = os.path.join(output_dir, "collect_data.log")
        logger.remove()
        logger.add(hydra_log_path, level="DEBUG")

    console_log_level = os.environ.get("LOGURU_LEVEL", "INFO").upper()
    logger.add(sys.stdout, level=console_log_level, colorize=True)
    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger().addHandler(HydraLoggerBridge())


def load_config(override_config):
    """
    Loads and merges the training configuration with the override configuration.
    Validates the existence of the checkpoint and its associated config file.
    """
    if override_config.checkpoint is None:
        logger.error("Please provide a checkpoint path via checkpoint=/path/to/ckpt.pt")
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
    """
    Configures environment-specific settings such as simulator type, dataset paths, and termination conditions.
    Also sets up the directory for saving rendering data.
    """
    simulator_type = config.simulator["_target_"].split(".")[-1]
    if simulator_type != "IsaacGym":
        logger.error(
            f"This script only supports IsaacGym simulator. Found: {simulator_type}"
        )
        sys.exit(1)

    pre_process_config(config)

    config.headless = True
    config.num_envs = config.get("data_collection_num_envs", 3960)
    config.env.config.locomotion_command_resampling_time = 4.0
    config.domain_rand.push_robots = True

    # Modify training curriculum
    config.rewards.command_height_scale_initial_scale = 1.0
    config.rewards.force_scale_initial_scale = 1.0

    logger.info(f"Number of environments: {config.env.config.num_envs}")

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


def monkey_patch_reset(env, device):
    """
    Patches the environment's reset callback to modify reset behavior.
    Ensures that only environments with failures are reset, while timeouts are handled separately.
    Also tracks which environments have been reset due to failures.
    """
    original_reset_robot_states = env._reset_robot_states_callback

    # Initialize reset flag buffer - True indicates the environment was reset due to failure
    env.reset_flag = torch.zeros(env.num_envs, dtype=torch.bool, device=device)

    def custom_reset_robot_states(self, env_ids, target_states=None):
        # Reset_flag is cleared at the start of each step in collect_data()
        is_timeout = self.time_out_buf[env_ids]
        failure_ids = env_ids[~is_timeout]
        # is_timeout_ids = env_ids[is_timeout]

        if len(failure_ids) > 0:
            original_reset_robot_states(failure_ids, target_states)
            logger.info(f"Reset {failure_ids.tolist()} due to failure")
            self.reset_flag[failure_ids] = True

        # if len(is_timeout_ids) > 0:
        #     original_reset_robot_states(is_timeout_ids, target_states)
        #     logger.info(f"Reset {is_timeout_ids.tolist()} due to timeout")
        #     self.reset_flag[is_timeout_ids] = True

        # original_reset_robot_states(env_ids, target_states)
        # logger.info(f"Reset {env_ids.tolist()}")
        # self.reset_flag[env_ids] = True

    env._reset_robot_states_callback = types.MethodType(custom_reset_robot_states, env)


def initialize_motion_pool(env, config, device):
    """
    Initializes a pool of motions larger than the number of environments.
    This enables independent motion switching for each environment upon completion or timeout.
    """
    num_envs = config.env.config.num_envs
    motion_pool_size = config.get(
        "motion_pool_size", 1980
    )  # 1980 is the size of CMU dataset
    logger.info(
        f"Loading motion pool with {motion_pool_size} motions for {num_envs} environments..."
    )

    original_motion_lib_num_envs = env._motion_lib.num_envs
    env._motion_lib.num_envs = motion_pool_size
    env._motion_lib.load_motions(random_sample=False)
    env._motion_lib.num_envs = original_motion_lib_num_envs

    env.motion_ids = torch.randint(0, motion_pool_size, (num_envs,), device=device)
    env.motion_len = env._motion_lib.get_motion_length(env.motion_ids)
    # logger.info(f"Initial motion assignments: {env.motion_ids.tolist()}")
    return motion_pool_size


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
    """
    Sets up the entire simulation environment, including the environment instance, agent, and motion pool.
    Loads the agent's checkpoint and sets it to evaluation mode.
    """
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

    monkey_patch_reset(env, device)
    motion_pool_size = initialize_motion_pool(env, config, device)

    logger.info("Instantiating agent...")
    algo: BaseAlgo = instantiate(config.algo, env=env, device=device, log_dir=None)
    algo.setup()
    algo.load(config.checkpoint)
    algo._eval_mode()

    return env, algo, motion_pool_size, body_idx


def handle_timeouts(env, infos, motion_pool_size, step, device):
    """
    Checks for environment timeouts and switches them to new random motions from the pool.
    This ensures continuous data collection with varied motions.
    """
    num_envs = env.num_envs
    time_outs = infos.get(
        "time_outs", torch.zeros(num_envs, dtype=torch.bool, device=device)
    )
    timed_out_env_ids = torch.where(time_outs)[0]

    if len(timed_out_env_ids) > 0:
        new_motion_ids = torch.randint(
            0, motion_pool_size, (len(timed_out_env_ids),), device=device
        )
        env.motion_ids[timed_out_env_ids] = new_motion_ids
        env.motion_len[timed_out_env_ids] = env._motion_lib.get_motion_length(
            new_motion_ids
        )
        # logger.info(
        #     f"Step {step}: Env {timed_out_env_ids.tolist()} switched to motions {new_motion_ids.tolist()}"
        # )


def collect_data(env, algo, config, motion_pool_size, body_idx, device):
    """
    Runs the main data collection loop.
    Iterates through steps, collects observations and actions, steps the environment, and handles timeouts.
    Also tracks reset flags indicating which environments were reset due to failures.
    Additionally constructs 206-dim state vectors (critic_obs 128 + extended rigid body 78).
    """
    num_steps = config.get("num_steps", 500)
    num_envs = config.env.config.num_envs
    logger.info(
        f"Collecting data for {num_steps} steps with {num_envs} environments..."
    )

    obs_list = []
    actions_list = []
    state_206_list = []
    reset_flags_list = []
    rewards_list = []
    reward_keys = None
    obs_dict = env.reset_all()

    with torch.inference_mode():
        for step in tqdm(range(num_steps), desc="Collecting Data"):
            # Clear reset_flag at the START of each step
            # Ensures reset_flag is True only for envs reset during THIS step
            env.reset_flag[:] = False

            curr_obs = {k: v.cpu().numpy() for k, v in obs_dict.items()}
            obs_list.append(curr_obs)

            # Construct 206-dim state: critic_obs (128) + extended rigid body (78)
            # For fields shared between critic_obs and actor_obs, use the
            # actor_obs values so that observation noise is consistent with
            # what the actor policy received.
            #
            # Fields are in ALPHABETICAL order (sorted() in
            # legged_robot_base_ma.py _post_config_observation_callback).
            #
            # Critic obs (128, alphabetical):
            #   actions[0:29], base_ang_vel[29:32], base_lin_vel[32:35],
            #   base_orientation[35:39], cmd_ang_vel[39:40], cmd_base_height[40:41],
            #   cmd_lin_vel[41:43], cmd_stand[43:44], cmd_waist_dofs[44:47],
            #   dof_pos[47:76], dof_vel[76:105], left_ee_force[105:108],
            #   proj_gravity[108:111], ref_upper_dof_pos[111:125],
            #   right_ee_force[125:128]
            #
            # Actor obs last frame (115, alphabetical, at offset 460):
            #   actions[460:489], base_ang_vel[489:492], cmd_ang_vel[492:493],
            #   cmd_base_height[493:494], cmd_lin_vel[494:496], cmd_stand[496:497],
            #   cmd_waist_dofs[497:500], dof_pos[500:529], dof_vel[529:558],
            #   proj_gravity[558:561], ref_upper_dof_pos[561:575]
            critic_obs = obs_dict["critic_obs"].clone()  # (num_envs, 128)
            actor_obs = obs_dict["actor_obs"]  # (num_envs, 575)

            # Overwrite shared fields with actor_obs (last frame) values
            critic_obs[:, 0:29] = actor_obs[:, 460:489]  # actions
            critic_obs[:, 29:32] = actor_obs[:, 489:492]  # base_ang_vel
            critic_obs[:, 39:40] = actor_obs[:, 492:493]  # command_ang_vel
            critic_obs[:, 40:41] = actor_obs[:, 493:494]  # command_base_height
            critic_obs[:, 41:43] = actor_obs[:, 494:496]  # command_lin_vel
            critic_obs[:, 43:44] = actor_obs[:, 496:497]  # command_stand
            critic_obs[:, 44:47] = actor_obs[:, 497:500]  # command_waist_dofs
            critic_obs[:, 47:76] = actor_obs[:, 500:529]  # dof_pos
            critic_obs[:, 76:105] = actor_obs[:, 529:558]  # dof_vel
            critic_obs[:, 108:111] = actor_obs[:, 558:561]  # projected_gravity
            critic_obs[:, 111:125] = actor_obs[:, 561:575]  # ref_upper_dof_pos
            # Critic-only fields kept: base_lin_vel[32:35], base_orientation[35:39],
            #   left_ee_force[105:108], right_ee_force[125:128]

            extended_state = extract_extended_state(env, body_idx)  # (num_envs, 78)
            state_206 = torch.cat([critic_obs, extended_state], dim=1)
            state_206_list.append(state_206.cpu().numpy())

            if hasattr(algo, "act_inference"):
                actions = algo.act_inference(obs_dict["actor_obs"])
            else:
                actions = algo.actor.act_inference(obs_dict["actor_obs"])

            actions_list.append(actions.cpu().numpy())

            actor_state = {"actions": actions}
            obs_dict, rewards, _, infos = env.step(actor_state)

            total_reward = sum(rewards.values())
            rewards_list.append(total_reward.cpu().numpy())
            if reward_keys is None:
                reward_keys = sorted(rewards.keys())

            handle_timeouts(env, infos, motion_pool_size, step, device)

            # Collect reset flags after step (indicates if env was reset due to failure)
            reset_flags_list.append(env.reset_flag.cpu().numpy().copy())

    return (
        obs_list,
        actions_list,
        state_206_list,
        reset_flags_list,
        rewards_list,
        reward_keys,
    )


def save_results(
    obs_list,
    actions_list,
    state_206_list,
    reset_flags_list,
    rewards_list,
    reward_keys,
    output_dir,
):
    """
    Saves the collected observations, actions, 206-dim state, reset flags, and rewards to disk.
    Reorganizes observations from list of dicts to dict of arrays.
    Reset flags indicate which environments were reset due to failures (discontinuous data).
    """
    obs_path = output_dir / "observations.npz"
    actions_path = output_dir / "actions.npy"
    state_206_path = output_dir / "state_206.npy"
    reset_flags_path = output_dir / "reset_flags.npy"
    rewards_path = output_dir / "rewards.npy"
    reward_keys_path = output_dir / "reward_keys.txt"

    save_obs = {}
    keys = obs_list[0].keys()
    for k in keys:
        save_obs[k] = np.array([x[k] for x in obs_list])

    logger.info(f"Saving observations to {obs_path}")
    np.savez(obs_path, **save_obs)
    for k, v in save_obs.items():
        logger.info(f"  {k}: {v.shape}")

    save_actions = np.array(actions_list)
    logger.info(f"Saving actions to {actions_path} with shape {save_actions.shape}")
    np.save(actions_path, save_actions)

    save_state_206 = np.array(state_206_list)
    logger.info(
        f"Saving 206-dim state to {state_206_path} with shape {save_state_206.shape}"
    )
    np.save(state_206_path, save_state_206)

    save_reset_flags = np.array(reset_flags_list)
    logger.info(
        f"Saving reset flags to {reset_flags_path} with shape {save_reset_flags.shape}"
    )
    logger.info(f"  Total resets due to failures: {save_reset_flags.sum()}")
    # logger.info(f"  Total resets due to timeouts: {save_reset_flags.sum()}")
    np.save(reset_flags_path, save_reset_flags)

    save_rewards = np.array(rewards_list)
    logger.info(f"Saving rewards to {rewards_path} with shape {save_rewards.shape}")
    np.save(rewards_path, save_rewards)

    with open(reward_keys_path, "w") as f:
        f.write("\n".join(reward_keys))
    logger.info(f"Saving reward keys to {reward_keys_path}: {reward_keys}")


@hydra.main(config_path="config", config_name="base_eval", version_base="1.1")
def main(override_config: OmegaConf):
    """
    Main entry point for the data collection script.
    Orchestrates the entire process: logging setup, config loading, simulation setup,
    data collection, and saving results.
    """
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    output_dir = Path(HydraConfig.get().runtime.output_dir)
    setup_logging(output_dir)
    os.chdir(hydra.utils.get_original_cwd())

    config, checkpoint = load_config(override_config)

    env, algo, motion_pool_size, body_idx = setup_simulation(config, checkpoint, device)

    (
        obs_list,
        actions_list,
        state_206_list,
        reset_flags_list,
        rewards_list,
        reward_keys,
    ) = collect_data(env, algo, config, motion_pool_size, body_idx, device)

    save_results(
        obs_list,
        actions_list,
        state_206_list,
        reset_flags_list,
        rewards_list,
        reward_keys,
        output_dir,
    )

    logger.info("Done.")


if __name__ == "__main__":
    main()
