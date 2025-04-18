import torch

WRIST_CAM_SN = "f1150781"
EXO_CAM_SN = "f1371608"
TABLE_CAM_SN = "f1371426"

GRIPPER_EFFORT = 500


def deg2rad(deg):
    return deg * torch.pi / 180


def rad2deg(rad):
    return rad * 180 / torch.pi