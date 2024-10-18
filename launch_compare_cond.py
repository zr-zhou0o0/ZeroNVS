'''
Compare SDS and without SDS. Use cond image and camera pose in Zero123 as condition.
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

# The convention is opengl (x axis right, y axis up, z axis facing towards the viewer) in camera-to-world format.

# 'gl11' CORRECT
def opencv_to_opengl(pose):
    S = np.diag([1, 1, -1])
    R = pose[:3,:3]

    R_prime = R @ S

    S2 = np.diag([1, -1, 1]) 
    R_prime = S2 @ R_prime

    pose_gl = pose.copy()
    pose_gl[:3,:3] = R_prime
    return pose_gl


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
    euler_angle = out[6]

    K = K/K[2,2]
    intrinsics = np.eye(4) 
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3,3] = (t[:3] / t[3])[:,0] 

    return intrinsics, pose, euler_angle


# OpenCV
def get_camera_poses(cam_file_path):
    camera_dict = np.load(cam_file_path)
    n_images = len(camera_dict.files) // 2
    scale_mats = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(n_images)]
    world_mats = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(n_images)]

    intrinsics_all = []
    pose_all = []
    # forward_vectors_world_all = []  # 用于存储世界坐标系下的前向向量
    camera_positions_world_all = []  # 用于存储世界坐标系下的相机位置
    euler_angles_all = []  # 用于存储前向向量的欧拉角
    euler_angles_returned_all = []
    save_data = {}
    idx = 0
    for scale_mat, world_mat in zip(scale_mats, world_mats):
        P = world_mat @ scale_mat 
        P = P[:3, :4]
        intrinsics, pose, euler_angle_returned = load_K_Rt_from_P(None, P)

        pose = opencv_to_opengl(pose) # 'gl2': here to convert to opengl

        # because we do resize and center crop 384x384 when using omnidata model, we need to adjust the camera intrinsic accordingly
        scale = 384 / 680 # raw is 680 and resized image is 384
        offset = (1200 - 680 ) * 0.5 # photocentre, width from 1200 to 680
        intrinsics[0, 2] -= offset
        intrinsics[:2, :] *= scale

        intrinsics_all.append(intrinsics)
        pose_all.append(pose)

        # 将内参和位姿矩阵分别以 'intrinsics_x' 和 'pose_x' 命名保存
        save_data[f'intrinsics_{idx}'] = intrinsics
        save_data[f'pose_{idx}'] = pose


        # 计算相机前向向量在世界坐标系下的方向
        # cam_to_world = np.eye(4)  # 扩展 3x4 矩阵为 4x4 齐次坐标矩阵
        # # cam_to_world[:3, :4] = pose
        cam_to_world = pose

        # 相机前向向量 (0, 0, 1) 在相机坐标系下
        # forward_vector_cam = np.array([0, 0, -1, 0])  # 使用齐次坐标
        forward_vector_cam = np.array([0, 0, 1, 0])  # 使用齐次坐标
        up_vector_cam = np.array([0, 1, 0, 0])  # 使用齐次坐标

        # # 通过位姿矩阵将前向向量转换到世界坐标系
        forward_vector_world = cam_to_world @ forward_vector_cam
        up_vector_world = cam_to_world @ up_vector_cam
        # forward_vectors_world_all.append(forward_vector_world[:3])  # 存储前向向量

        # # 保存前向向量
        # save_data[f'forward_vector_{idx}'] = forward_vector_world[:3]  # 保存转换后的前向向量


        # # 计算前向向量的欧拉角，并保存
        # roll_deg, pitch_deg, yaw_deg = vector_to_euler(forward_vector_world[:3], up_vector_world[:3]) # xyz
        # euler_angles_all.append([roll_deg, pitch_deg, yaw_deg])
        # # 保存欧拉角
        # save_data[f'forward_vector_euler_compute{idx}'] = [roll_deg, pitch_deg, yaw_deg]


        # 保存opencv decompose 返回的欧拉角
        euler_angles_returned_all.append(euler_angle_returned)
        save_data[f'forward_vector_euler_returned{idx}'] = euler_angle_returned



        # 相机位置向量 (0, 0, 0) 在相机坐标系下，用齐次坐标表示
        position_cam = np.array([0, 0, 0, 1])  # 使用齐次坐标，最后一位是 1 表示位置向量

        # 通过位姿矩阵将相机位置转换到世界坐标系
        position_world = pose @ position_cam
        camera_positions_world_all.append(position_world[:3])  # 存储相机位置向量

        # 保存相机位置
        save_data[f'camera_position_{idx}'] = position_world[:3]  # 保存转换后的相机位置
        idx += 1


    save_path = os.path.join("data/image_output", "20_poses_position_euler2.npz")
    np.savez(save_path, **save_data)
    # np.savez(save_path, intrinsics_all=intrinsics_all, pose_all=pose_all)

    print('camera params loaded') 
    return intrinsics_all, pose_all, euler_angles_returned_all


def get_cond_camera_poses(pose_all):
    # suppose every 10th image are conditions, the rest are novel view
    cond_poses = []
    cond_idx = []
    input_camera_spacing = 6                                  
    for idx, pose in enumerate(pose_all):
        if idx % input_camera_spacing == 0:
            cond_poses.append(pose)
            cond_idx.append(idx)
    return cond_poses, cond_idx
    

def get_target_camera_poses(pose_all):
    target_poses = []
    target_idx = []
    input_camera_spacing = 6                                
    for idx, pose in enumerate(pose_all):
        if idx % input_camera_spacing != 0:
            target_poses.append(pose)
            target_idx.append(idx)
    return target_poses, target_idx


def extract_pose_components(pose):
    translation = pose[:3, 3]
    rotation = pose[:3, :3]
    return translation, rotation


def calculate_distance(input_pose, candidate_pose, alpha=1.0, beta=1.0, target_euler=None, euler=None):
    input_translation, input_rotation = extract_pose_components(input_pose)
    candidate_translation, candidate_rotation = extract_pose_components(candidate_pose)
    
    position_distance = np.linalg.norm(input_translation - candidate_translation)
    
    return alpha * position_distance + beta * np.abs(target_euler[1] - euler[1])


def find_nearest_cond(target_pose, cond_poses, cond_idx, alpha=1.0, beta=1.0, target_euler=None, euler_all=None): # cond euler
    '''
    cond poses: a list
    cond idx: a list, the raw idx of each cond pose
    return the nearest cond pose of target pose, and the raw idx of it
    '''
    nearest_cond_pose = None
    nearest_cond_idx = None
    min_distance = float('inf')

    for pose, euler, idx in zip(cond_poses, euler_all, cond_idx):
        distance = calculate_distance(target_pose, pose, alpha, beta, target_euler, euler)
        if distance < min_distance:
            min_distance = distance
            nearest_cond_pose = pose
            nearest_cond_idx = idx

    return nearest_cond_pose, nearest_cond_idx


def launch():
    
    # SET THIS
    cond_img_path = 'data/image_test/000063_rgb.png'
    sds_img_path = 'data/sds_outputs/save2/it4000-val'
    output_path = 'data/concat_output/test1'
    cam_file = 'data/cameras.npz'
    target_space = 10


    intrinsic_all, pose_all, euler_returned_all = get_camera_poses(cam_file)
    intrinsic = intrinsic_all[0]
    # cond_poses, cond_idx = get_cond_camera_poses(pose_all)
    # target_poses, target_idx = get_target_camera_poses(pose_all)
    # cond_euler = [euler_returned_all[i] for i in cond_idx]
    f_x = intrinsic[0, 0] 
    image_width = 256
    fov_horizontal = 2 * np.arctan(image_width / (2 * f_x)) * 180 / np.pi  # rad -> degree
    fov_tensor = torch.from_numpy(np.array([fov_horizontal])).cuda().to(torch.float32)


    cond_image_pil = Image.open(cond_img_path).convert("RGB")
    cond_image_pil = cond_image_pil.resize((256, 256))
    cond_image = torch.from_numpy(np.array(cond_image_pil)).cuda() / 255.
    

    guidance_cfg = dict(
                pretrained_model_name_or_path= "zeronvs.ckpt", 
                pretrained_config= "zeronvs_config.yaml", 
                guidance_scale= 7.5,
                cond_image_path =cond_img_path,
                min_step_percent=[0,.75,.02,1000],
                max_step_percent=[1000, 0.98, 0.025, 2500],
                vram_O=False 
            )
    guidance = zero123_guidance.Zero123Guidance(OmegaConf.create(guidance_cfg))
    c_crossattn, c_concat = guidance.get_img_embeds(cond_image.permute((2, 0, 1))[None])


    if not os.path.exists(output_path):
        os.makedirs(output_path)


    target_poses_all = test_camera()
    for i, tp in enumerate(target_poses_all):
        if i % target_space == 0:
            
            # cond_camera, _, _ = get_defaults()
            cond_camera = target_poses_all[0]
            target_camera = tp
            print('cond', cond_camera)

            target_camera = target_camera.numpy()
            cond_camera = cond_camera.numpy()
            
            target_camera = torch.from_numpy(target_camera[None]).cuda().to(torch.float32)
            cond_camera = torch.from_numpy(cond_camera[None]).cuda().to(torch.float32)
            # 添加一个维度以匹配形状要求（从 [3] 到 [1, 3]）
            # target_camera = target_camera.unsqueeze(0).cuda().to(torch.float32)
            # cond_camera = cond_camera.unsqueeze(0).cuda().to(torch.float32)

            camera_batch = {
                "target_cam2world": target_camera, 
                "cond_cam2world": cond_camera,
                # "fov_deg": torch.from_numpy(np.array([45.0])).cuda().to(torch.float32) 
                "fov_deg": fov_tensor # XXX horizontal field of view, degree
            }

            guidance.cfg.precomputed_scale=.7
            cond = guidance.get_cond_from_known_camera(
                camera_batch,
                c_crossattn=c_crossattn,
                c_concat=c_concat,
                # precomputed_scale=.7,
            )

            print("------camerabatch--------")
            print(camera_batch["cond_cam2world"]) 
            print(camera_batch["target_cam2world"]) 

            novel_view = guidance.gen_from_cond(cond) 
            novel_view_pil = Image.fromarray(np.clip(novel_view[0]*255, 0, 255).astype(np.uint8))

            # target_path = os.path.join(img_path, f'{idx:06d}_rgb.png')
            # print("target_path=",target_path)
            # target_pil = Image.open(target_path).convert("RGB").resize((256, 256))

            sds_novel_path = os.path.join(sds_img_path, f'{i}.png')
            sds_image_pil = Image.open(sds_novel_path).convert("RGB")

            width, height = sds_image_pil.size
            split_width = width // 4
            split_sds_image = sds_image_pil.crop((0, 0, split_width, height)).resize((256, 256)) # (left, upper, right, lower)

            # concatenate images
            cond_image_array = np.array(cond_image_pil)
            novel_image_array = np.array(novel_view_pil)
            sds_image_array = np.array(split_sds_image)

            # min_height = min(cond_image_array.shape[0], target_image_array.shape[0], novel_image_array.shape[0])
            # cond_image_array = cond_image_array[:min_height]
            # target_image_array = target_image_array[:min_height]
            # novel_image_array = novel_image_array[:min_height]

            concatenated = np.hstack((cond_image_array, novel_image_array, sds_image_array))
            concatenated_image = Image.fromarray(concatenated)
            if not os.path.exists(os.path.join(output_path, 'concat')):
                os.makedirs(os.path.join(output_path, 'concat'))
            concatenated_save_path = os.path.join(output_path, f'concat/{i}_concatenated.png')
            concatenated_image.save(concatenated_save_path)


            # cond_save_path = os.path.join(output_path, f'{i}_{nearest_cond_idx:06d}_cond.png')
            # novel_save_path = os.path.join(output_path, f'{i}_{idx:06d}_novel.png')
            # cond_image_pil.save(cond_save_path)
            # novel_view_pil.save(novel_save_path)

        

if __name__ == '__main__':
    launch()