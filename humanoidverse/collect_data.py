import os
import sys
from pathlib import Path
import hydra
from omegaconf import OmegaConf
import logging
from loguru import logger
import numpy as np
from hydra.core.hydra_config import HydraConfig

# Helper to add project root to path if needed, though hydra handles this usually
# sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from utils.config_utils import *


@hydra.main(config_path="config", config_name="base_eval", version_base="1.1")
def main(override_config: OmegaConf):
    # Logging setup
    # logging to hydra log file
    if HydraConfig.initialized():
        hydra_log_path = os.path.join(
            HydraConfig.get().runtime.output_dir, "collect_data.log"
        )
        logger.remove()
        logger.add(hydra_log_path, level="DEBUG")

    # Get log level from LOGURU_LEVEL environment variable or use INFO as default
    console_log_level = os.environ.get("LOGURU_LEVEL", "INFO").upper()
    logger.add(sys.stdout, level=console_log_level, colorize=True)

    logging.basicConfig(level=logging.DEBUG)
    # We delay importing HydraLoggerBridge until after potential isaacgym import just in case,
    # though it should be fine.

    os.chdir(hydra.utils.get_original_cwd())

    # Load Checkpoint Config
    if override_config.checkpoint is None:
        logger.error("Please provide a checkpoint path via checkpoint=/path/to/ckpt.pt")
        return

    checkpoint = Path(override_config.checkpoint)
    config_path = checkpoint.parent / "config.yaml"
    if not config_path.exists():
        config_path = checkpoint.parent.parent / "config.yaml"

    if not config_path.exists():
        logger.error(f"Could not find config path: {config_path}")
        return

    logger.info(f"Loading training config file from {config_path}")
    with open(config_path) as file:
        train_config = OmegaConf.load(file)

    # Merge configs
    # We merge override_config (CLI args) into train_config.
    # We do NOT merge train_config.eval_overrides (which usually sets num_envs=1)
    # because we want to keep training parallel configuration.
    config = OmegaConf.merge(train_config, override_config)

    # Only IsaacGym
    simulator_type = config.simulator["_target_"].split(".")[-1]
    if simulator_type != "IsaacGym":
        logger.error(
            f"This script only supports IsaacGym simulator. Found: {simulator_type}"
        )
        return

    config.headless = False

    # Import IsaacGym and Torch
    import isaacgym
    import torch
    from hydra.utils import instantiate
    from humanoidverse.utils.logging import HydraLoggerBridge
    from humanoidverse.utils.helpers import pre_process_config
    from humanoidverse.agents.base_algo.base_algo import BaseAlgo

    logging.getLogger().addHandler(HydraLoggerBridge())

    pre_process_config(config)
    config.env.config.num_envs = 10

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Parallel simulation configuration same as training
    # This is handled by using `train_config` which contains `num_envs` from training.
    logger.info(f"Number of environments: {config.env.config.num_envs}")

    # Setup paths
    ckpt_num = config.checkpoint.split("/")[-1].split("_")[-1].split(".")[0]
    # We might need save_rendering_dir for some env internal logic, even if we don't render video explicitly
    config.env.config.save_rendering_dir = str(
        checkpoint.parent / "renderings_data" / f"ckpt_{ckpt_num}"
    )

    # Instantiate Environment
    logger.info("Instantiating environment...")
    env = instantiate(config.env, device=device)

    # Instantiate Agent
    logger.info("Instantiating agent...")
    algo: BaseAlgo = instantiate(config.algo, env=env, device=device, log_dir=None)
    algo.setup()

    # Load Checkpoint
    algo.load(config.checkpoint)

    # No exploration
    algo._eval_mode()

    # Data Collection Loop
    logger.info("Starting data collection...")
    obs_list = []
    actions_list = []
    obs_dict = env.reset_all()

    # Run one full iteration
    num_steps = config.get("num_steps", 250)
    logger.info(f"Collecting data for {num_steps} steps...")

    from tqdm import tqdm

    # Record all observations and actions
    with torch.inference_mode():
        for step in tqdm(range(num_steps), desc="Collecting Data"):
            # Collect current observation (before action)
            # Transfer to CPU/Numpy to save memory
            curr_obs = {k: v.cpu().numpy() for k, v in obs_dict.items()}
            obs_list.append(curr_obs)

            # Get Action (No exploration)
            # act_inference returns the mean of the distribution
            if hasattr(algo, "act_inference"):
                actions = algo.act_inference(obs_dict["actor_obs"])
            else:
                actions = algo.actor.act_inference(obs_dict["actor_obs"])

            # Store action
            curr_actions = actions.cpu().numpy()
            actions_list.append(curr_actions)

            # Step environment
            actor_state = {"actions": actions}
            obs_dict, rewards, dones, infos = env.step(actor_state)

    logger.info("Data collection finished.")

    # Save to files
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    obs_path = output_dir / "observations.npz"
    actions_path = output_dir / "actions.npy"

    # Reorganize observations: List of Dicts -> Dict of Arrays
    # shape: (T, num_envs, ...)
    save_obs = {}
    keys = obs_list[0].keys()
    for k in keys:
        save_obs[k] = np.array([x[k] for x in obs_list])

    save_actions = np.array(actions_list)

    logger.info(f"Saving observations to {obs_path}")
    logger.info("Observation shapes:")
    for k, v in save_obs.items():
        logger.info(f"  {k}: {v.shape}")

    np.savez(obs_path, **save_obs)

    logger.info(f"Saving actions to {actions_path} with shape {save_actions.shape}")
    np.save(actions_path, save_actions)

    logger.info("Done.")


if __name__ == "__main__":
    main()
