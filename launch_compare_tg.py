'''
Compare SDS and without SDS. Use a single target camera pose in dataset as condition.
'''

import os
from PIL import Image

import cv2
import numpy as np
import torch

from ldm.models.diffusion import options
options.LDM_DISTILLATION_ONLY = True

from threestudio.models.guidance import zero123_guidance
from omegaconf import OmegaConf 

from test_camera import test_camera, get_defaults


def compute_m_nd(nerf_pose, dataset_pose):
    # dataset=A, nerf=B
    # T_B = T * T_A 
    # T = T_B * T_A_inv

    # Compute T_A_inv
    dataset_r = dataset_pose[:3, :3]
    dataset_t = dataset_pose[:3, 3]
    dataset_r_inv = dataset_r.T
    dataset_t_inv = -dataset_r_inv @ dataset_t
    dataset_pose_inv = np.eye(4)
    dataset_pose_inv[:3, :3] = dataset_r_inv
    dataset_pose_inv[:3, 3] = dataset_t_inv
    
    M_nd = nerf_pose @ dataset_pose_inv
    return M_nd


def compute_tg_pose_nerf():

    cond_pose = None
    target_pose = None

    nerf_default_pose, _, _ = get_defaults()
    M_nd = compute_m_nd(nerf_default_pose, cond_pose)
    target_pose_nerf = M_nd @ target_pose


if __name__ == '__main__':
    compute_tg_pose_nerf()
