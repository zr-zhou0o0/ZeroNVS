'''
NOTE: in our data, Replica use OPENCV coords (raw data is in OPENGL coords, but we convert it to OPENCV coords)
so in cameras.npz, camera parameters are in OPENCV coords
'''

import os
import cv2
import numpy as np

# K is intrinsic matrix of camera
# RT is pose matrix of camera, with rotation and translation. it is NOT extrinsic matrix
# P is projection matrix
def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv2.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    # import pdb; pdb.set_trace()

    K = K/K[2,2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3,3] = (t[:3] / t[3])[:,0]

    return intrinsics, pose


root_path = 'data/replica/scan6'
cam_file = os.path.join(root_path, 'cameras.npz')
camera_dict = np.load(cam_file)
n_images = len(camera_dict.files) // 2
scale_mats = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(n_images)]
world_mats = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(n_images)]

# load camera intrinsics and poses (in OPENCV coords)
# NOTE: camera intrinsics is the camera to image transformation
#       camera pose is the camera to world transformation
#       camera extrinsics is the world to camera transformation
intrinsics_all = []
pose_all = []
for scale_mat, world_mat in zip(scale_mats, world_mats):
    P = world_mat @ scale_mat # matrix multiplication, not element-wise multiplication
    # why world @ scale = projection? what is projection?
    P = P[:3, :4]
    intrinsics, pose = load_K_Rt_from_P(None, P)

    # because we do resize and center crop 384x384 when using omnidata model, we need to adjust the camera intrinsic accordingly
    scale = 384 / 680
    offset = (1200 - 680 ) * 0.5
    intrinsics[0, 2] -= offset
    intrinsics[:2, :] *= scale

    intrinsics_all.append(intrinsics)
    pose_all.append(pose)

print('camera params loaded')

# load image
input_camera_spacing = 10                                   # we will use every 10th image as input, the rest as novel view
img_root_path = os.path.join(root_path, 'image')
for idx in range(n_images):
    img_path = os.path.join(img_root_path, f'{idx:06d}_rgb.png')
    img = cv2.imread(img_path)

    camera_intrinsic = intrinsics_all[idx]
    camera_pose = pose_all[idx]

    if idx % input_camera_spacing == 0:
        print(f'we will use this image for input {idx}')

    # print(f'image {idx}: image shape: {img.shape}, camera intrinsic: {camera_intrinsic}, camera pose: {camera_pose}')
