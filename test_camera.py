#copy from threestudio/data/uncond.py

import bisect
import math
import random
from dataclasses import dataclass, field

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, IterableDataset

import threestudio
from threestudio import register
from threestudio.utils.base import Updateable
from threestudio.utils.config import parse_structured
from threestudio.utils.misc import get_device
from threestudio.utils.ops import (
    get_mvp_matrix,
    get_projection_matrix,
    get_ray_directions,
    get_rays,
)
from threestudio.utils.typing import *


def test_camera():
    # add outside
    split = 'val'
    n_views = 240
    eval_elevation_deg = 31.0
    eval_camera_distance = 1.0
    eval_batch_size = 1
    eval_fovy_deg = 48.9183
    # add end


    azimuth_deg: Float[Tensor, "B"]
    if split == "val": # XXX HERE!!! VAL CAMERA AZIMUTH! cfg=cfg, split=val
        # make sure the first and last view are not the same
        azimuth_deg = torch.linspace(0, 360.0, n_views + 1)[: n_views] # so the camera view is default(azimuth)?
    else:
        azimuth_deg = torch.linspace(0, 360.0, n_views)
    elevation_deg: Float[Tensor, "B"] = torch.full_like(
        azimuth_deg, eval_elevation_deg
    )
    camera_distances: Float[Tensor, "B"] = torch.full_like(
        elevation_deg, eval_camera_distance
    )

    elevation = elevation_deg * math.pi / 180
    azimuth = azimuth_deg * math.pi / 180

    # convert spherical coordinates to cartesian coordinates
    # right hand coordinate system, x back, y right, z up
    # elevation in (-90, 90), azimuth from +x to +y in (-180, 180)
    camera_positions: Float[Tensor, "B 3"] = torch.stack(
        [
            camera_distances * torch.cos(elevation) * torch.cos(azimuth),
            camera_distances * torch.cos(elevation) * torch.sin(azimuth),
            camera_distances * torch.sin(elevation),
        ],
        dim=-1,
    )

    # default scene center at origin
    center: Float[Tensor, "B 3"] = torch.zeros_like(camera_positions)
    # default camera up direction as +z
    up: Float[Tensor, "B 3"] = torch.as_tensor([0, 0, 1], dtype=torch.float32)[
        None, :
    ].repeat(eval_batch_size, 1)

    fovy_deg: Float[Tensor, "B"] = torch.full_like(
        elevation_deg, eval_fovy_deg
    )
    fov_deg = fovy_deg
    fovy = fovy_deg * math.pi / 180
    light_positions: Float[Tensor, "B 3"] = camera_positions

    lookat: Float[Tensor, "B 3"] = F.normalize(center - camera_positions, dim=-1)
    right: Float[Tensor, "B 3"] = F.normalize(torch.cross(lookat, up), dim=-1)
    up = F.normalize(torch.cross(right, lookat), dim=-1)
    c2w3x4: Float[Tensor, "B 3 4"] = torch.cat(
        [torch.stack([right, up, -lookat], dim=-1), camera_positions[:, :, None]],
        dim=-1,
    )
    c2w: Float[Tensor, "B 4 4"] = torch.cat( # type hint, B is batchsize
        [c2w3x4, torch.zeros_like(c2w3x4[:, :1])], dim=1
    )
    c2w[:, 3, 3] = 1.0


    print(c2w[0])
    print(c2w[20]) # # 20 = 30degree
    print(c2w[60])
    print(c2w[120])
    print(c2w[180])
    print(c2w[200]) # -20 = -30degree


    return c2w


'''
split = 'val'
n_views = 240
eval_elevation_deg = 31.0
eval_camera_distance = 1.0
eval_batch_size = 1
eval_fovy_deg = 48.9183

print(c2w[0])
print(c2w[20]) # # 20 = 30degree
print(c2w[60])
print(c2w[120])
print(c2w[180])
print(c2w[200]) # -20 = -30degree

tensor([[ 0.0000, -0.5150,  0.8572,  0.8572],
        [ 1.0000,  0.0000, -0.0000,  0.0000],
        [-0.0000,  0.8572,  0.5150,  0.5150],
        [ 0.0000,  0.0000,  0.0000,  1.0000]])
tensor([[-0.5000, -0.4460,  0.7423,  0.7423],
        [ 0.8660, -0.2575,  0.4286,  0.4286],
        [ 0.0000,  0.8572,  0.5150,  0.5150],
        [ 0.0000,  0.0000,  0.0000,  1.0000]])
tensor([[-1.0000e+00,  2.2513e-08, -3.7468e-08, -3.7468e-08],
        [-4.3711e-08, -5.1504e-01,  8.5717e-01,  8.5717e-01],
        [ 0.0000e+00,  8.5717e-01,  5.1504e-01,  5.1504e-01],
        [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]])
tensor([[ 8.7423e-08,  5.1504e-01, -8.5717e-01, -8.5717e-01],
        [-1.0000e+00,  4.5026e-08, -7.4936e-08, -7.4936e-08],
        [ 0.0000e+00,  8.5717e-01,  5.1504e-01,  5.1504e-01],
        [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]])
tensor([[ 1.0000e+00, -6.1418e-09,  1.0222e-08,  1.0222e-08],
        [ 1.1925e-08,  5.1504e-01, -8.5717e-01, -8.5717e-01],
        [-0.0000e+00,  8.5717e-01,  5.1504e-01,  5.1504e-01],
        [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]])
tensor([[ 0.8660, -0.2575,  0.4286,  0.4286],
        [ 0.5000,  0.4460, -0.7423, -0.7423],
        [-0.0000,  0.8572,  0.5150,  0.5150],
        [ 0.0000,  0.0000,  0.0000,  1.0000]])
'''