import logging
import time
import numpy as np
import torch
from PIL import Image
from custom_scripts.common.robot_devices.robot_utils import init_robot
from custom_scripts.common.robot_devices.cam_utils import RealSenseCamera

def load_buffer(buffer, action_pred_queue):
    for item in buffer:
        item.append(action_pred_queue.popleft().squeeze())
    return buffer

def get_current_action(buffer, m=1.0):
    current_action_stack = torch.stack(buffer.pop(0), dim=0)
    indices = torch.arange(current_action_stack.shape[0])
    weights = torch.exp(-m*indices).cuda()
    weighted_actions = current_action_stack * weights[:, None]  # 가중치 적용
    current_action = weighted_actions.sum(dim=0) / weights.sum()
    return buffer, current_action

def random_piper_action():
    (x, y, z) = torch.rand(3, dtype=torch.float32) * 600000
    (rx, ry, rz) = torch.rand(3, dtype=torch.float32) * 180000
    gripper = torch.rand(1, dtype=torch.float32) * 100000
    return torch.tensor([x,y,z,rx,ry,rz,gripper]).reshape(1,7)

def random_piper_image():
    return torch.rand(1, 3, 480, 640, dtype=torch.float32)

def random_piper_image_openvla(width=224, height=224):
    random_array = np.random.randint(0, 256, size=(height, width, 3), dtype=np.uint8)
    random_image = Image.fromarray(random_array, mode="RGB")
    return random_image

def plot_trajectory(ax, action_list, projection='2d', mode='pred'):
    # action_list = np.array(action_list)
    action_list = torch.stack(action_list, dim=0)
    color = 'r' if mode == 'pred' else 'b'
    if projection == '2d':
        ax[0,0].plot(action_list[:,0], color=color)
        ax[1,0].plot(action_list[:,1], color=color)
        ax[2,0].plot(action_list[:,2], color=color)
        ax[0,1].plot(action_list[:,3], color=color)
        ax[1,1].plot(action_list[:,4], color=color)
        ax[2,1].plot(action_list[:,5], color=color)
        ax[3,0].plot(action_list[:,6], color=color)
    elif projection == '3d':
        ax.plot(action_list[:,0], action_list[:,1], action_list[:,2], color=color)
    else:
        raise ValueError('projection must be \"2d\" or \"3d\"')

def pretty_plot(ax):
    # for ax in ax:
    pass

def log_time():
    return time.perf_counter()

def init_devices(cfg, is_recording=False):
    fps = cfg.fps
    cam_list = cfg.cam_list
    cam = {
        'wrist_rs_cam': None,
        'exo_rs_cam': None,
        'table_rs_cam': None,
    }
    piper = init_robot(is_recording=is_recording)
    if 'wrist' in cam_list:
        cam['wrist_rs_cam'] = RealSenseCamera('wrist', fps)
    if 'exo' in cam_list:
        cam['exo_rs_cam'] = RealSenseCamera('exo', fps)
    if 'table' in cam_list:
        cam['table_rs_cam'] = RealSenseCamera('table', fps)
    return piper, cam

def get_task_index(task):
    task_list = ['Align the cups']
    if task not in task_list:
        logging.info("TASK NOT IN TASK LIST")
        return -1
    return task_list.index(task)
