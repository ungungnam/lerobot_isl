import logging
import time
from contextlib import nullcontext
from dataclasses import asdict
from pprint import pformat
from typing import Any
from collections import deque

import numpy as np
import torch
from termcolor import colored
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils import load_buffer, get_current_action

from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.common.datasets.factory import make_dataset
from lerobot.common.datasets.sampler import EpisodeAwareSampler
from lerobot.common.datasets.utils import cycle

from lerobot.common.policies.factory import make_policy
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.policies.utils import get_device_from_parameters

from lerobot.common.utils.wandb_utils import WandBLogger
from lerobot.common.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.common.utils.random_utils import set_seed
from lerobot.common.utils.utils import (
    get_safe_torch_device,
    init_logging,
    format_big_number,
)

from lerobot.configs import parser
from lerobot.configs.eval_ours import EvalOursPipelineConfig


def evaluate_policy(
    eval_metrics: MetricsTracker,
    policy: PreTrainedPolicy,
    batch: Any,
    use_amp: bool = False,
    lock=None,
) -> tuple[MetricsTracker, dict]:
    start_time = time.perf_counter()
    device = get_device_from_parameters(policy)
    policy.eval()
    with torch.autocast(device_type=device.type) if use_amp else nullcontext():
        loss, output_dict = policy.forward(batch)
        # TODO(rcadene): policy.unnormalize_outputs(out_dict)

    eval_metrics.loss = loss.item()
    eval_metrics.update_s = time.perf_counter() - start_time
    return eval_metrics, output_dict


@parser.wrap()
def eval_main(cfg: EvalOursPipelineConfig):
    logging.info(pformat(asdict(cfg)))

    if cfg.wandb.enable and cfg.wandb.project:
        wandb_logger = WandBLogger(cfg)
    else:
        wandb_logger = None
        logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

    # Check device is available
    device = get_safe_torch_device(cfg.policy.device, log=True)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    set_seed(cfg.seed)

    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")

    logging.info("Creating dataset")
    train_dataset_meta = LeRobotDatasetMetadata(
        cfg.train_dataset.repo_id, cfg.train_dataset.root, revision=cfg.train_dataset.revision
    )
    dataset = make_dataset(cfg)

    logging.info("Making policy.")

    policy = make_policy(
        cfg=cfg.policy,
        ds_meta=train_dataset_meta,
    )

    step = 0  # number of policy updates (forward + backward + optim)
    num_total_params = sum(p.numel() for p in policy.parameters())

    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
    if cfg.env is not None:
        logging.info(f"{cfg.env.task=}")
    logging.info(f"{dataset.num_frames=} ({format_big_number(dataset.num_frames)})")
    logging.info(f"{dataset.num_episodes=}")
    logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")

    # create dataloader for offline training
    if hasattr(cfg.policy, "drop_n_last_frames"):
        shuffle = False
        sampler = EpisodeAwareSampler(
            dataset.episode_data_index,
            drop_n_last_frames=cfg.policy.drop_n_last_frames,
            shuffle=False,
        )
    else:
        shuffle = False
        sampler = None

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        pin_memory=device.type != "cpu",
        drop_last=False,
    )
    dl_iter = cycle(dataloader)

    policy.eval()

    eval_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }

    eval_tracker = MetricsTracker(
        cfg.batch_size, dataset.num_frames, dataset.num_episodes, eval_metrics, initial_step=step
    )

    logging.info("Start offline evaluation on a fixed dataset")
    for episode_num in range(0, len(dataset.episodes)):
        start_frame = dataset.episode_data_index['from'][episode_num]
        end_frame = dataset.episode_data_index['to'][episode_num]

        buffer = [[] for _ in range(policy.config.n_action_steps)]
        fig_2d, ax_2d = plt.subplots(4,2,figsize=[25, 15])
        fig_3d, ax_3d = plt.subplots(subplot_kw={'projection': '3d'}, figsize=[25, 15])

        action_pred_list = []
        action_ans_list = []

        inference_time_list = []

        for frame_num in tqdm(range(end_frame - start_frame)):
            t0 = time.time()
            start_time = time.perf_counter()
            batch = next(dl_iter)
            eval_tracker.dataloading_s = time.perf_counter() - start_time

            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device, non_blocking=True)

            # Check Loss
            eval_tracker, output_dict = evaluate_policy(
                eval_tracker,
                policy,
                batch,
                use_amp = cfg.policy.use_amp,
            )

            # Plot Trajectory
            action_pred = policy.select_action(batch).squeeze()
            action_ans =  batch['action'].squeeze()[0]
            # TODO: Implement ACT
            if cfg.temporal_ensemble:
                action_pred_queue = policy._action_queue.copy()
                action_pred_queue.extendleft(action_pred.unsqueeze(0))
                policy.reset()

                buffer = load_buffer(buffer, action_pred_queue)
                buffer, action_pred = get_current_action(buffer)
                buffer.append([])

            # action_pred_list.append(action_pred.cpu().numpy())
            # action_ans_list.append(action_ans.cpu().numpy())
            action_pred_list.append(action_pred.cpu() if isinstance(action_pred, torch.Tensor) else action_pred)
            action_ans_list.append(action_ans.cpu() if isinstance(action_ans, torch.Tensor) else action_ans)

            # Step Logger
            step += 1
            eval_tracker.step()
            is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0

            if is_log_step:
                logging.info(eval_tracker)
                if wandb_logger:
                    wandb_log_dict = eval_tracker.to_dict()
                    if output_dict:
                        wandb_log_dict.update(output_dict)
                    wandb_logger.log_dict(wandb_log_dict, step)
                eval_tracker.reset_averages()

            t1 = time.time()
            inference_time_list.append(t1 - t0)

        plot_trajectory(ax_2d, action_pred_list)
        plot_trajectory(ax_2d, action_ans_list, mode='ans')
        pretty_plot(ax_2d)

        plot_trajectory(ax_3d, action_pred_list, projection='3d')
        plot_trajectory(ax_3d, action_ans_list, projection='3d', mode='ans')
        pretty_plot(ax_3d)

        fig_2d.show()
        fig_3d.show()

    logging.info("End of eval")


if __name__ == "__main__":
    init_logging()
    eval_main()
