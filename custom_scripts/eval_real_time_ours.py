import time
import logging
from pprint import pformat
from dataclasses import asdict

from termcolor import colored
import torch

from piper_sdk import C_PiperInterface
from cam_utils import RealSenseCamera
from robot_utils import read_end_pose_msg

from lerobot.common.policies.factory import make_policy

from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata

from lerobot.common.utils.random_utils import set_seed
from lerobot.common.utils.wandb_utils import WandBLogger
from lerobot.common.utils.utils import (
    get_safe_torch_device,
    init_logging,
    format_big_number,
)


def setZeroConfiguration(piper):
    piper.JointCtrl(0, 0, 0, 0, 0, 0)
    piper.GripperCtrl(0,0, 0x01, 0)
    piper.MotionCtrl_2(0x01, 0x01, 20, 0x00)
    time.sleep(5)


def init_devices(cfg):
    fps = cfg.fps

    piper = C_PiperInterface("can0")
    piper.ConnectPort()
    piper.EnableArm(7)
    setZeroConfiguration(piper)

    wrist_rs_cam = RealSenseCamera('wrist', fps)
    exo_rs_cam = RealSenseCamera('exo', fps)
    table_rs_cam = RealSenseCamera('table', fps)

    return piper, wrist_rs_cam, exo_rs_cam, table_rs_cam


def eval_main(cfg):
    logging.info(pformat(asdict(cfg)))
    piper, wrist_rs_cam, exo_rs_cam, table_rs_cam = init_devices(cfg)

    if cfg.wandb.enable and cfg.wandb.project:
        wandb_logger = WandBLogger(cfg)
    else:
        wandb_logger = None
        logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

    device = get_safe_torch_device(cfg.policy.device, log=True)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    set_seed(cfg.seed)

    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")

    logging.info("Creating dataset")
    train_dataset_meta = LeRobotDatasetMetadata(
        cfg.train_dataset.repo_id, cfg.train_dataset.root, revision=cfg.train_dataset.revision
    )

    logging.info("Making policy.")

    policy = make_policy(
        cfg=cfg.policy,
        ds_meta=train_dataset_meta,
    )

    step = 0
    num_total_params = sum(p.numel() for p in policy.parameters())

    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
    if cfg.env is not None:
        logging.info(f"{cfg.env.task=}")
    logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")

    wrist_rs_cam.start_recording()
    exo_rs_cam.start_recording()
    table_rs_cam.start_recording()

    policy.eval()

    logging.info("Start offline evaluation on a fixed dataset")
    while True:
        pass
        # create batch
        batch = {}
        batch['observation.state'] = read_end_pose_msg(piper)
        batch['observation.images.exo'] = exo_rs_cam.image
        batch['observation.images.wrist'] = wrist_rs_cam.image
        # batch['observation.images.table'] = table_rs_cam.image

        for item in batch.items():



        # infer data

        # actuate robot



if __name__ == "__main__":
    init_logging()
    eval_main()