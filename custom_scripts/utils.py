import torch

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
    return torch.cat((x, y, z, rx, ry, rz, gripper), dim=0)

def random_piper_image():
    return torch.rand(3, 480, 640, dtype=torch.float32) * 255
