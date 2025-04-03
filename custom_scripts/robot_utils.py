import torch

def read_end_pose_msg(piper):
    end_pose = piper.GetArmEndPoseMsgs().end_pose
    grippers = piper.GetArmGripperMsgs().gripper_state
    end_pose_data =torch.Tensor([end_pose.X_axis, end_pose.Y_axis, end_pose.Z_axis, end_pose.RX_axis, end_pose.RY_axis, end_pose.RZ_axis, grippers.grippers_angle])
    return end_pose_data