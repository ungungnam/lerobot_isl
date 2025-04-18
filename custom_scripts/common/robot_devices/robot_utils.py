import torch
import time
from piper_sdk import C_PiperInterface, C_PiperForwardKinematics

from custom_scripts.common.utils.utils import deg2rad


def init_robot():
    piper = C_PiperInterface("can0")
    piper.ConnectPort()
    piper.EnableArm(7)
    set_zero_configuration(piper)

    return piper

def readJointCtrl(piper):
    joints = piper.GetArmJointCtrl().joint_ctrl
    joint_data = torch.tensor([joints.joint_1, joints.joint_2, joints.joint_3, joints.joint_4, joints.joint_5, joints.joint_6])
    return joint_data


def readGripperCtrl(piper):
    grippers = piper.GetArmGripperCtrl().gripper_ctrl
    gripper_data = torch.tensor([grippers.grippers_angle, grippers.grippers_effort])
    return gripper_data


def readJointMsg(piper):
    joints = piper.GetArmJointMsgs().joint_state
    joint_data = torch.tensor([joints.joint_1, joints.joint_2, joints.joint_3, joints.joint_4, joints.joint_5, joints.joint_6])
    return joint_data


def readEndPoseMsg(piper):
    end_pose = piper.GetArmEndPoseMsgs().end_pose
    end_pose_data = torch.tensor(
        [end_pose.X_axis, end_pose.Y_axis, end_pose.Z_axis, end_pose.RX_axis, end_pose.RY_axis, end_pose.RZ_axis])
    return end_pose_data


def readGripperMsg(piper):
    grippers = piper.GetArmGripperMsgs().gripper_state
    gripper_data = torch.tensor([grippers.grippers_angle, grippers.grippers_effort])
    return gripper_data

def read_end_pose_ctrl(piper):
    joints = piper.GetArmJointCtrl().joint_ctrl
    grippers = piper.GetArmGripperCtrl().gripper_ctrl
    joints = deg2rad(0.001 * joints)
    end_pose = 1000 * torch.tensor(C_PiperForwardKinematics.CalFK(joints)[-1]).astype(int)

    end_pose_data =torch.tensor([end_pose.X_axis, end_pose.Y_axis, end_pose.Z_axis, end_pose.RX_axis, end_pose.RY_axis, end_pose.RZ_axis, grippers.grippers_angle])
    end_pose_data = end_pose_data.unsqueeze(0)
    return end_pose_data


def read_end_pose_msg(piper):
    end_pose = piper.GetArmEndPoseMsgs().end_pose
    grippers = piper.GetArmGripperMsgs().gripper_state
    end_pose_data =torch.tensor([end_pose.X_axis, end_pose.Y_axis, end_pose.Z_axis, end_pose.RX_axis, end_pose.RY_axis, end_pose.RZ_axis, grippers.grippers_angle])
    end_pose_data = end_pose_data.unsqueeze(0)
    return end_pose_data


def set_zero_configuration(piper):
    piper.JointCtrl(0, 0, 0, 0, 0, 0)
    piper.GripperCtrl(0,0, 0x01, 0)
    piper.MotionCtrl_2(0x01, 0x01, 20, 0x00)
    time.sleep(5)


def ctrl_end_pose(piper, end_pose_data, gripper_data):
    gripper_angle, gripper_effort = gripper_data[:]
    # gripper_effort = gripper_effort if gripper_effort > 0 else 0

    piper.MotionCtrl_2(0x01, 0x00, 20, 0x00)
    piper.EndPoseCtrl(*end_pose_data)
    piper.GripperCtrl(abs(gripper_angle), gripper_effort, 0x01, 0)