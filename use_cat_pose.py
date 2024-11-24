import os
import numpy as np

def get_pose_intrinsic():
    root_path = './scannetpp-scan110/'
    pose_file = 'camera_poses_all.npy'
    intrinsics_file = 'camera_intrinsics_fix.npy'

    pose_path = os.path.join(root_path, pose_file)
    intrinsics_path = os.path.join(root_path, intrinsics_file)

    pose = np.load(pose_path)                   # [100, 4, 4]
    intrinsics = np.load(intrinsics_path)       # [1, 4, 4], no need to process
    # print(pose.shape)
    # print(intrinsics.shape)
    return pose, intrinsics


