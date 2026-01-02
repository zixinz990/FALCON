import os
import sys
from pathlib import Path
import isaacgym
import torch
import numpy as np
import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf
from loguru import logger
from humanoidverse.utils.config_utils import *
from humanoidverse.agents.base_algo.base_algo import BaseAlgo


@hydra.main(config_path="config", config_name="base_eval", version_base="1.1")
def main(override_config: OmegaConf):
    # Ensure we are in the original working directory to resolve relative paths correctly
    os.chdir(hydra.utils.get_original_cwd())

    # Check for checkpoint argument
    if override_config.checkpoint is None:
        logger.error(
            "Please provide a checkpoint path using 'checkpoint=/path/to/checkpoint.pt'"
        )
        sys.exit(1)

    checkpoint_path = Path(override_config.checkpoint)
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found at {checkpoint_path}")
        sys.exit(1)

    # Load training configuration from the checkpoint directory
    config_path = checkpoint_path.parent / "config.yaml"
    if not config_path.exists():
        # Try looking one level up (standard structure: experiment_dir/config.yaml, checkpoint in experiment_dir/checkpoints/...)
        config_path = checkpoint_path.parent.parent / "config.yaml"

    if not config_path.exists():
        logger.error(
            f"Could not find config.yaml at {checkpoint_path.parent} or {checkpoint_path.parent.parent}"
        )
        sys.exit(1)

    logger.info(f"Loading training config from {config_path}")
    with open(config_path) as file:
        train_config = OmegaConf.load(file)

    # Merge configurations
    # We prioritize training config for environment setup, but override specific settings for data collection
    config = OmegaConf.merge(train_config, override_config)

    # Force IsaacGym
    simulator_type = config.simulator["_target_"].split(".")[-1]
    if simulator_type != "IsaacGym":
        logger.error(
            f"This script only supports IsaacGym, but config specifies {simulator_type}"
        )
        sys.exit(1)

    # Force visualization (headless=False) as per requirements
    config.headless = False

    # Force test mode to False/True?
    # Usually training config has test=False. We want training configuration (parallel, randomization), so we leave it as is.
    # But we want to ensure we don't explore. That is handled by using the inference policy.

    # Device setup
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Instantiate Environment
    logger.info("Instantiating environment...")
    try:
        env = instantiate(config.env, device=device)
    except Exception as e:
        logger.error(f"Failed to instantiate environment: {e}")
        # Sometimes headless=False fails on servers without display. User asked for it though.
        raise e

    # Instantiate Algorithm
    logger.info("Instantiating algorithm...")
    # We use a dummy log_dir since we aren't training
    algo: BaseAlgo = instantiate(config.algo, env=env, device=device, log_dir=None)
    algo.setup()

    # Load Checkpoint
    logger.info(f"Loading checkpoint weights from {checkpoint_path}")
    algo.load(str(checkpoint_path))

    # Data Collection Loop
    logger.info("Starting data collection...")

    all_obs = []
    all_actions = []

    # Reset environment to get initial observation
    obs_dict = env.reset_all()

    # Determine number of steps
    num_steps = algo.num_steps_per_env
    logger.info(f"Collecting data for {num_steps} steps (one full iteration)...")

    # Ensure inference mode
    algo.actor.eval()
    algo.critic.eval()

    with torch.inference_mode():
        for step in range(num_steps):
            # 1. Store current observation (actor_obs)
            # We move to CPU and numpy immediately to save GPU memory
            # obs_dict["actor_obs"] shape: (num_envs, obs_dim)
            current_obs = obs_dict["actor_obs"].cpu().numpy()
            all_obs.append(current_obs)

            # 2. Get action from policy (Inference/Deterministic)
            # algo.actor.act_inference usually returns the mean action (deterministic)
            actions = algo.actor.act_inference(obs_dict["actor_obs"])

            # 3. Store action
            # actions shape: (num_envs, action_dim)
            current_actions = actions.cpu().numpy()
            all_actions.append(current_actions)

            # 4. Step environment
            # Prepare actor_state as expected by env.step
            actor_state = {"actions": actions}
            obs_dict, rewards, dones, infos = env.step(actor_state)

            if (step + 1) % 10 == 0:
                print(f"Step {step + 1}/{num_steps}", end="\r")

    print("\nData collection completed.")

    # Save to files
    output_dir = Path("collected_data")
    output_dir.mkdir(exist_ok=True)

    obs_file = output_dir / "observations.npy"
    act_file = output_dir / "actions.npy"

    logger.info(f"Saving observations to {obs_file}...")
    np.save(obs_file, np.array(all_obs))

    logger.info(f"Saving actions to {act_file}...")
    np.save(act_file, np.array(all_actions))

    logger.info("Done.")


if __name__ == "__main__":
    main()
