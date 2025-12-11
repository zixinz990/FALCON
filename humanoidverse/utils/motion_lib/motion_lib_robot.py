from humanoidverse.utils.motion_lib.motion_lib_base import MotionLibBase
from humanoidverse.utils.motion_lib.torch_humanoid_batch import Humanoid_Batch
import hydra
import torch
from loguru import logger
from omegaconf import DictConfig


class MotionLibRobot(MotionLibBase):
    def __init__(self, motion_lib_cfg, num_envs, device):
        super().__init__(
            motion_lib_cfg=motion_lib_cfg, num_envs=num_envs, device=device
        )
        self.mesh_parsers = Humanoid_Batch(motion_lib_cfg)
        # return


@hydra.main(version_base=None, config_path="../../config", config_name="base")
def main(config: DictConfig):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_envs = 512
    motionlib = MotionLibRobot(config.robot.motion, num_envs, device)
    motions = motionlib.load_motions()
    logger.info(f"Loaded {len(motions)} motions")
    motion_times = 0
    import ipdb

    ipdb.set_trace()
    for motion in motions:
        # logger.info(f"Motion DoF Pos Length: {motion['dof_pos'].shape}")
        # logger.info(f"First Frame Pos: {motion['dof_pos'][0, 12:]}")
        motion_times += motion["dof_pos"].shape[0] / motion["fps"]
    logger.info(f"Average Motion Length: {motion_times / len(motions)}")


if __name__ == "__main__":
    main()
