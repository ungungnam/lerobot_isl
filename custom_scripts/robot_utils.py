import torch
import time

def read_end_pose_msg(piper):
    end_pose = piper.GetArmEndPoseMsgs().end_pose
    grippers = piper.GetArmGripperMsgs().gripper_state
    end_pose_data =torch.Tensor([end_pose.X_axis, end_pose.Y_axis, end_pose.Z_axis, end_pose.RX_axis, end_pose.RY_axis, end_pose.RZ_axis, grippers.grippers_angle])
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
