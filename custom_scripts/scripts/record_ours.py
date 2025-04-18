import logging
import time
from tqdm import tqdm
from pprint import pformat
from dataclasses import asdict

from custom_scripts.common.robot_devices.robot_utils import read_end_pose_msg
from custom_scripts.configs.record_ours import RecordOursPipelineConfig
from custom_scripts.common.utils.utils import init_devices
from custom_scripts.common.dataset.piper_dataset import PiperDataset
from custom_scripts.common.robot_devices.robot_utils import read_end_pose_ctrl

from lerobot.configs import parser

@parser.wrap
def record_episodes(cfg = RecordOursPipelineConfig):
    logging.info(pformat(asdict(cfg)))
    if cfg.use_devices:
        piper, cam = init_devices(cfg)

        wrist_rs_cam = cam['wrist_rs_cam']
        exo_rs_cam = cam['exo_rs_cam']
        table_rs_cam = cam['table_rs_cam']
    else:
        piper = None
        wrist_rs_cam = None
        exo_rs_cam = None
        table_rs_cam = None

    if cfg.use_devices:
        wrist_rs_cam.start_recording()
        exo_rs_cam.start_recording()
        table_rs_cam.start_recording()
        logging.info("Devices started recording")

    piper_dataset = PiperDataset(root=cfg.dataset_path, episode_num=cfg.episode_num, episode_len=cfg.episode_len, create_video=cfg.create_video, fps=cfg.fps)

    task = cfg.task
    fps = cfg.fps

    for i in tqdm(range(piper_dataset.num_frames)):
        t0 = time.time()
        frame = {
            'timestamp': time.time(),
            'frame_id': i,
            'action': read_end_pose_ctrl(piper),
            'observation.state': read_end_pose_msg(piper),
            'observation.images.wrist': wrist_rs_cam.image,
            'observation.images.exo': exo_rs_cam.image,
            'observation.images.table': table_rs_cam.image,
            'task': task,
        }
        piper_dataset.add_frame(frame)
        t_act = time.time()-t0
        time.sleep(max(0,1/fps - t_act))

    return piper_dataset

if __name__ == "__main__":
    dataset = record_episodes()
    dataset.save_episode()