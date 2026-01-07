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


def setup_simulation(config, checkpoint, device):
    """
    Sets up the entire simulation environment, including the environment instance, agent, and motion pool.
    Loads the agent's checkpoint and sets it to evaluation mode.
    """
    configure_env_settings(config, checkpoint)

    logger.info("Instantiating environment...")
    env = instantiate(config.env, device=device)

    monkey_patch_reset(env, device)
    motion_pool_size = initialize_motion_pool(env, config, device)

    logger.info("Instantiating agent...")
    algo: BaseAlgo = instantiate(config.algo, env=env, device=device, log_dir=None)
    algo.setup()
    algo.load(config.checkpoint)
    algo._eval_mode()

    return env, algo, motion_pool_size


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


def collect_data(env, algo, config, motion_pool_size, device):
    """
    Runs the main data collection loop.
    Iterates through steps, collects observations and actions, steps the environment, and handles timeouts.
    Also tracks reset flags indicating which environments were reset due to failures.
    """
    num_steps = config.get("num_steps", 500)
    num_envs = config.env.config.num_envs
    logger.info(
        f"Collecting data for {num_steps} steps with {num_envs} environments..."
    )

    obs_list = []
    actions_list = []
    reset_flags_list = []
    obs_dict = env.reset_all()

    with torch.inference_mode():
        for step in tqdm(range(num_steps), desc="Collecting Data"):
            # Clear reset_flag at the START of each step
            # Ensures reset_flag is True only for envs reset during THIS step
            env.reset_flag[:] = False

            curr_obs = {k: v.cpu().numpy() for k, v in obs_dict.items()}
            obs_list.append(curr_obs)

            if hasattr(algo, "act_inference"):
                actions = algo.act_inference(obs_dict["actor_obs"])
            else:
                actions = algo.actor.act_inference(obs_dict["actor_obs"])

            actions_list.append(actions.cpu().numpy())

            actor_state = {"actions": actions}
            obs_dict, _, _, infos = env.step(actor_state)

            handle_timeouts(env, infos, motion_pool_size, step, device)

            # Collect reset flags after step (indicates if env was reset due to failure)
            reset_flags_list.append(env.reset_flag.cpu().numpy().copy())

    return obs_list, actions_list, reset_flags_list


def save_results(obs_list, actions_list, reset_flags_list, output_dir):
    """
    Saves the collected observations, actions, and reset flags to disk.
    Reorganizes observations from list of dicts to dict of arrays.
    Reset flags indicate which environments were reset due to failures (discontinuous data).
    """
    obs_path = output_dir / "observations.npz"
    actions_path = output_dir / "actions.npy"
    reset_flags_path = output_dir / "reset_flags.npy"

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

    save_reset_flags = np.array(reset_flags_list)
    logger.info(
        f"Saving reset flags to {reset_flags_path} with shape {save_reset_flags.shape}"
    )
    logger.info(f"  Total resets due to failures: {save_reset_flags.sum()}")
    # logger.info(f"  Total resets due to timeouts: {save_reset_flags.sum()}")
    np.save(reset_flags_path, save_reset_flags)


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

    env, algo, motion_pool_size = setup_simulation(config, checkpoint, device)

    obs_list, actions_list, reset_flags_list = collect_data(
        env, algo, config, motion_pool_size, device
    )

    save_results(obs_list, actions_list, reset_flags_list, output_dir)

    logger.info("Done.")


if __name__ == "__main__":
    main()
