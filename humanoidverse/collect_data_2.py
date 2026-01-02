import os
import sys
from pathlib import Path
import numpy as np
import hydra
from hydra.utils import instantiate
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from humanoidverse.utils.logging import HydraLoggerBridge
import logging
from loguru import logger

# import torch # Moved inside main to ensure it is imported after isaacgym

# Ensure isaacgym is imported before torch (if it was imported here, but we do it conditionally below)
# However, train_agent.py imports torch after isaacgym.
# Here torch is imported at top level. Let's follow eval_agent.py pattern.


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

    # Config loading logic
    if override_config.checkpoint is not None:
        has_config = True
        checkpoint = Path(override_config.checkpoint)
        config_path = checkpoint.parent / "config.yaml"
        if not config_path.exists():
            config_path = checkpoint.parent.parent / "config.yaml"
            if not config_path.exists():
                has_config = False
                logger.error(f"Could not find config path: {config_path}")

        if has_config:
            logger.info(f"Loading training config file from {config_path}")
            with open(config_path) as file:
                train_config = OmegaConf.load(file)

            # Note: We do NOT apply eval_overrides from train_config because they might set num_envs to 1.
            # We want to keep the training parallel configuration.

            # Merge with override_config (command line args)
            config = OmegaConf.merge(train_config, override_config)
        else:
            config = override_config
    else:
        logger.error(
            "Please provide a checkpoint using checkpoint=path/to/checkpoint.pt"
        )
        return

    # Enforce requirements:
    # 1. IsaacGym only
    simulator_type = config.simulator["_target_"].split(".")[-1]
    if simulator_type != "IsaacGym":
        logger.error(
            f"Simulator type is {simulator_type}, but only IsaacGym is supported for this script."
        )
        return

    import isaacgym  # noqa: F401
    import torch
    from humanoidverse.agents.base_algo.base_algo import BaseAlgo
    from humanoidverse.utils.helpers import pre_process_config

    # 4. Open visualization (headless=False)
    config.headless = False

    # Ensure parallel configuration matches training (num_envs should be preserved from train_config)
    logger.info(f"Number of environments: {config.num_envs}")

    pre_process_config(config)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Instantiate environment
    # config.env.config.save_rendering_dir might be needed if env tries to use it
    ckpt_num = config.checkpoint.split("/")[-1].split("_")[-1].split(".")[0]
    config.env.config.save_rendering_dir = str(
        checkpoint.parent / "renderings_collect" / f"ckpt_{ckpt_num}"
    )

    logger.info("Instantiating environment...")
    env = instantiate(config.env, device=device)

    # Instantiate algorithm
    logger.info("Instantiating algorithm...")
    algo: BaseAlgo = instantiate(config.algo, env=env, device=device, log_dir=None)
    algo.setup()

    # 2. Load checkpoint
    algo.load(config.checkpoint)

    # 5. No learning or updating
    # 3. Run one full iteration
    # We use algo.num_steps_per_env to define "one full iteration" of data collection.
    num_steps = algo.num_steps_per_env
    logger.info(f"Collecting data for {num_steps} steps...")

    obs_list = []
    action_list = []

    # Set eval mode to disable exploration (if applicable in eval mode)
    # Usually eval mode uses mean action.
    algo._eval_mode()

    # Get the inference policy (deterministic)
    if hasattr(algo, "actor") and hasattr(algo.actor, "act_inference"):
        get_action = algo.actor.act_inference
    else:
        logger.warning(
            "Could not find act_inference method. Using standard act method which might include exploration noise if not handled by _eval_mode."
        )
        get_action = algo.actor.act

    with torch.no_grad():
        obs_dict = env.reset_all()

        for step in range(num_steps):
            # Record observation
            # We are interested in actor observations which the policy sees
            actor_obs = obs_dict["actor_obs"]

            # Store observation (CPU, numpy)
            obs_list.append(actor_obs.cpu().numpy())

            # Get action
            actions = get_action(actor_obs)

            # Store action
            action_list.append(actions.cpu().numpy())

            # Step environment
            actor_state = {"actions": actions}
            # Note: env.step expects a dictionary usually, or handled by algo wrapper.
            # In PPO.py: obs_dict, rewards, dones, infos = self.env.step(actor_state)
            # But wait, PPO.py's _rollout_step calls:
            # obs_dict, rewards, dones, infos = self.env.step(actor_state)
            # Let's check env.step signature in base_task.py or how PPO uses it.
            # PPO.py line 276: obs_dict, rewards, dones, infos = self.env.step(actor_state)

            obs_dict, rewards, dones, infos = env.step(actor_state)

            # Handle resets if any (though for one iteration with many envs, we just record what happens)
            # If done, the obs_dict usually contains the reset observation.

            if step % 10 == 0:
                print(f"Step {step}/{num_steps}", end="\r")

    print(f"Step {num_steps}/{num_steps}")
    logger.info("Data collection finished.")

    # Convert to numpy arrays
    obs_array = np.array(obs_list)
    action_array = np.array(action_list)

    logger.info(f"Observations shape: {obs_array.shape}")
    logger.info(f"Actions shape: {action_array.shape}")

    # Save to files
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    obs_file = output_dir / "observations.npy"
    action_file = output_dir / "actions.npy"

    np.save(obs_file, obs_array)
    np.save(action_file, action_array)

    logger.info(f"Saved observations to {obs_file}")
    logger.info(f"Saved actions to {action_file}")


if __name__ == "__main__":
    main()
