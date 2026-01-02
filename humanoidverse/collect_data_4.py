import os
import sys
from pathlib import Path

# Important: Import isaacgym before torch
import isaacgym
import torch
import numpy as np

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import OmegaConf

from humanoidverse.utils.logging import HydraLoggerBridge
import logging
from loguru import logger
from humanoidverse.utils.helpers import pre_process_config
from humanoidverse.utils.common import seeding


@hydra.main(config_path="config", config_name="base_eval")
def main(override_config: OmegaConf):
    # logging to hydra log file
    hydra_log_path = os.path.join(
        HydraConfig.get().runtime.output_dir, "collect_data.log"
    )
    logger.remove()
    logger.add(hydra_log_path, level="DEBUG")

    # Get log level from LOGURU_LEVEL environment variable or use INFO as default
    console_log_level = os.environ.get("LOGURU_LEVEL", "INFO").upper()
    logger.add(sys.stdout, level=console_log_level, colorize=True)

    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger().addHandler(HydraLoggerBridge())

    os.chdir(hydra.utils.get_original_cwd())

    if override_config.checkpoint is not None:
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

        # Do NOT apply eval_overrides to keep training configuration (parallel envs, randomization)
        # But we need to force visualization
        train_config.headless = False

        # Merge override_config (cli args) on top of train_config
        config = OmegaConf.merge(train_config, override_config)
    else:
        logger.error(
            "Please provide a checkpoint using checkpoint=/path/to/checkpoint.pt"
        )
        return

    simulator_type = config.simulator["_target_"].split(".")[-1]
    if simulator_type != "IsaacGym":
        logger.error(
            f"Simulator {simulator_type} not supported for this script. Only IsaacGym is supported."
        )
        return

    pre_process_config(config)

    # Set seed if present in config to ensure same randomization as training
    if config.seed is not None:
        seeding(config.seed, torch_deterministic=config.torch_deterministic)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Instantiate environment with training config
    logger.info("Instantiating environment...")
    env = instantiate(config.env, device=device)

    # Instantiate algorithm
    logger.info("Instantiating algorithm...")
    algo = instantiate(config.algo, env=env, device=device, log_dir=None)
    algo.setup()

    # Load checkpoint
    algo.load(config.checkpoint)

    # Set to eval mode and get inference policy
    algo._eval_mode()
    policy = algo.actor.act_inference

    logger.info("Starting data collection...")
    obs_dict = env.reset_all()

    observations = []
    actions_list = []

    # Run for one full iteration (num_steps_per_env)
    num_steps = algo.num_steps_per_env
    logger.info(f"Collecting data for {num_steps} steps (one iteration)...")

    with torch.no_grad():
        for i in range(num_steps):
            # Get actor observations
            actor_obs = obs_dict["actor_obs"]

            # Get actions (deterministic/inference)
            actions = policy(actor_obs)

            # Store data
            observations.append(actor_obs.cpu().numpy())
            actions_list.append(actions.cpu().numpy())

            # Step environment
            obs_dict, rewards, dones, infos = env.step({"actions": actions})

            if (i + 1) % 10 == 0:
                logger.info(f"Step {i + 1}/{num_steps}")

    # Convert to numpy arrays
    observations = np.array(observations)
    actions_np = np.array(actions_list)

    logger.info(f"Collected observations shape: {observations.shape}")
    logger.info(f"Collected actions shape: {actions_np.shape}")

    # Save to files
    output_dir = HydraConfig.get().runtime.output_dir
    obs_path = os.path.join(output_dir, "observations.npy")
    act_path = os.path.join(output_dir, "actions.npy")

    np.save(obs_path, observations)
    np.save(act_path, actions_np)

    logger.info(f"Saved observations to {obs_path}")
    logger.info(f"Saved actions to {act_path}")


if __name__ == "__main__":
    main()
