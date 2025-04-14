import time
import logging
from pprint import pformat
from dataclasses import asdict

import matplotlib.pyplot as plt
from termcolor import colored
import torch
import numpy as np
from huggingface_hub import login

from piper_sdk import C_PiperInterface

from custom_scripts.common.constants import GRIPPER_EFFORT
from custom_scripts.common.robot_devices.cam_utils import RealSenseCamera
from custom_scripts.common.robot_devices.robot_utils import read_end_pose_msg, set_zero_configuration, ctrl_end_pose
from custom_scripts.common.utils.utils import (
    load_buffer,
    get_current_action,
    random_piper_action,
    random_piper_image,
    plot_trajectory,
    pretty_plot,
)
from custom_scripts.configs.eval_real_time_ours import EvalRealTimeOursPipelineConfig

from lerobot.configs import parser

from lerobot.common.policies.factory import make_policy
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata

from lerobot.common.utils.random_utils import set_seed
from lerobot.common.utils.wandb_utils import WandBLogger
from lerobot.common.utils.utils import (
    get_safe_torch_device,
    init_logging,
    format_big_number,
)

def create_batch(piper, wrist_rs_cam, exo_rs_cam, use_devices, task):
    if use_devices:
        return {
            'observation.state': read_end_pose_msg(piper),
            'observation.images.exo': exo_rs_cam.image,
            'observation.images.wrist': wrist_rs_cam.image,
            'task': [task],
        }
    else:
        return {
            'observation.state': random_piper_action(),
            'observation.images.exo': random_piper_image(),
            'observation.images.wrist': random_piper_image(),
            'task': [task],
        }


def init_devices(cfg):
    fps = cfg.fps

    piper = C_PiperInterface("can0")
    piper.ConnectPort()
    piper.EnableArm(7)
    set_zero_configuration(piper)

    wrist_rs_cam = RealSenseCamera('wrist', fps)
    exo_rs_cam = RealSenseCamera('exo', fps)
    # table_rs_cam = RealSenseCamera('table', fps)

    return piper, wrist_rs_cam, exo_rs_cam
    # return piper, wrist_rs_cam, exo_rs_cam, table_rs_cam


@parser.wrap()
def eval_main(cfg: EvalRealTimeOursPipelineConfig):
    logging.info(pformat(asdict(cfg)))
    if cfg.use_devices:
        piper, wrist_rs_cam, exo_rs_cam = init_devices(cfg)
        # piper, wrist_rs_cam, exo_rs_cam, table_rs_cam = init_devices(cfg)
    else:
        piper = None
        wrist_rs_cam = None
        exo_rs_cam = None
        table_rs_cam = None

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

    if cfg.use_devices:
        wrist_rs_cam.start_recording()
        exo_rs_cam.start_recording()
        # table_rs_cam.start_recording()
        logging.info("Devices started recording")

    policy.eval()

    logging.info("Start offline evaluation on a fixed dataset")

    buffer = [[] for _ in range(policy.config.n_action_steps)]
    action_pred_list = []

    fig_2d, ax_2d = plt.subplots(4, 2, figsize=[25, 15])
    fig_3d, ax_3d = plt.subplots(subplot_kw={'projection': '3d'}, figsize=[25, 15])

    while True:
        t0 = time.time()

        # create batch
        batch = create_batch(piper, wrist_rs_cam, exo_rs_cam, cfg.use_devices, cfg.task)

        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device, non_blocking=True)

        # infer data
        action_pred = policy.select_action(batch).squeeze()
        if cfg.temporal_ensemble:
            action_pred_queue = policy._action_queue.copy()
            action_pred_queue.extendleft(action_pred.unsqueeze(0))
            policy.reset()

            buffer = load_buffer(buffer, action_pred_queue)
            buffer, action_pred = get_current_action(buffer)
            buffer.append([])

        # actuate robot
        end_pose_data = action_pred[:6].cpu().to(dtype=int).tolist()
        gripper_data = [action_pred[6].cpu().to(dtype=int), GRIPPER_EFFORT]
        ctrl_end_pose(piper, end_pose_data, gripper_data)

        # log data
        action_pred_list.append(action_pred.cpu() if isinstance(action_pred, torch.Tensor) else action_pred)

        step += 1
        time.sleep(max(0, 1 / cfg.fps - (time.time() - t0)))

        if step > cfg.max_steps:
            break
        pass

    plot_trajectory(ax_2d, action_pred_list)
    pretty_plot(ax_2d)

    plot_trajectory(ax_3d, action_pred_list, projection='3d')
    pretty_plot(ax_3d)

    fig_2d.show()
    fig_3d.show()


if __name__ == "__main__":
    init_logging()
    eval_main()