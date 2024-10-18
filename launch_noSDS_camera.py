# coordinate system used for cond_camera and target_camera: opengl
# find the nearest cond camera pose for each novel camera pose
'''
Input: 
novel camera poses (get from scale mat @ world mat): a list
cond camera poses (get from scale mat @ world mat): a list
cond images directory, match the idx of cond camera poses

Output:
novel images

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

# The convention is opengl (x axis right, y axis up, z axis facing towards the viewer) in camera-to-world format.
# suppose in opencv, x axis right, y axis up, z axis facing towards the object

# def opencv_to_opengl(pose):
#     flip_yz = np.array([
#         [1, 0,  0, 0],
#         [0, -1, 0, 0],
#         [0, 0, -1, 0],
#         [0, 0,  0, 1]
#     ])
#     pose_opengl = flip_yz @ pose
#     return pose_opengl


# BUG which one?
# 'gl3': R @ S, i.e.inverse column 2 and column 3
# def opencv_to_opengl(pose):
#     # 复制一份输入的矩阵，避免修改原始矩阵
#     pose_gl = pose.copy()
    
#     # 将第二列 (索引 1) 和第三列 (索引 2) 取反
#     pose_gl[:, 1] *= -1  # 第二列取反
#     pose_gl[:, 2] *= -1  # 第三列取反
    
#     return pose_gl


# SRS, 'gl4'
# def opencv_to_opengl(pose):
#     # 定义反射矩阵 S
#     S = np.diag([1, -1, -1])

#     R = pose[:3,:3]

#     # 计算新的旋转矩阵 R'
#     R_prime = S @ R @ S

#     pose_gl = pose.copy()
#     pose_gl[:3,:3] = R_prime
#     return pose_gl


# SR, 'gl5'
# the right one...?
# def opencv_to_opengl(pose):
#     # 定义反射矩阵 S
#     S = np.diag([1, -1, -1])

#     R = pose[:3,:3]

#     # 计算新的旋转矩阵 R'
#     R_prime = S @ R

#     pose_gl = pose.copy()
#     pose_gl[:3,:3] = R_prime
#     return pose_gl


# 'gl6'
# def opencv_to_opengl(pose):
#     # 定义反射矩阵 S
#     S = np.diag([-1, 1, -1])

#     R = pose[:3,:3]

#     # 计算新的旋转矩阵 R'
#     R_prime = R @ S

#     pose_gl = pose.copy()
#     pose_gl[:3,:3] = R_prime
#     return pose_gl


#'gl7'
# def opencv_to_opengl(pose):
#     # 定义反射矩阵 S
#     S = np.diag([-1, 1, 1])

#     R = pose[:3,:3]

#     # 计算新的旋转矩阵 R'
#     R_prime = R @ S

#     pose_gl = pose.copy()
#     pose_gl[:3,:3] = R_prime
#     return pose_gl


# 'gl8'
# def opencv_to_opengl(pose):
#     # 定义反射矩阵 S
#     S = np.diag([-1, 1, 1])

#     R = pose[:3,:3]

#     # 计算新的旋转矩阵 R'
#     R_prime = S @ R

#     pose_gl = pose.copy()
#     pose_gl[:3,:3] = R_prime
#     return pose_gl


# 'gl9'
# def opencv_to_opengl(pose):
#     # 定义反射矩阵 S
#     S = np.diag([1, 1, -1])

#     R = pose[:3,:3]
#     # R = np.transpose(R)

#     # 计算新的旋转矩阵 R'
#     R_prime = R @ S

#     pose_gl = pose.copy()
#     pose_gl[:3,:3] = R_prime
#     return pose_gl


# 'gl10'
# def opencv_to_opengl(pose):
#     # 定义反射矩阵 S
#     S = np.diag([1, 1, -1])

#     R = pose[:3,:3]
#     # R = np.transpose(R)

#     # 计算新的旋转矩阵 R'
#     R_prime = S @ R @ S

#     pose_gl = pose.copy()
#     pose_gl[:3,:3] = R_prime
#     return pose_gl


# 'gl11' CORRECT
def opencv_to_opengl(pose):
    # 定义反射矩阵 S
    S = np.diag([1, 1, -1])

    R = pose[:3,:3]
    # R = np.transpose(R)

    # 计算新的旋转矩阵 R'
    R_prime = R @ S

    # R_prime[1,:] = -R_prime[1,:]
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
    pose[:3, :3] = R.transpose() # is it R(pose) or R(extrinsic)? take as extrinsic rotation. TODO confirm it.
    pose[:3,3] = (t[:3] / t[3])[:,0] 

    return intrinsics, pose, euler_angle


# what decompose get? -> opencv_decomposeprojection document.
# def load_K_Rt_from_P(filename, P=None):
#     if P is None:
#         lines = open(filename).read().splitlines()
#         if len(lines) == 4:
#             lines = lines[1:]
#         lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
#         P = np.asarray(lines).astype(np.float32).squeeze()

#     out = cv2.decomposeProjectionMatrix(P)
#     K = out[0]
#     R = out[1]
#     t = out[2]
#     euler_angle = out[6]

#     K = K/K[2,2]
#     intrinsics = np.eye(4) 
#     intrinsics[:3, :3] = K

#     pose = np.eye(4, dtype=np.float32)
#     pose[:3, :3] = R.transpose()
#     # pose[:3,3] = (t[:3] / t[3])[:,0] 
#     t = (t[:3] / t[3])[:, 0]
#     pose[:3, 3] = - R.transpose() @ t  # Corrected translation

#     return intrinsics, pose, euler_angle





# BUG need forward vector and up vector to decide euler
def vector_to_euler(forward_vector, up_vector):
    """
    从 forward_vector 和 up_vector 计算欧拉角 (roll, pitch, yaw)，单位为度数。
    """
    # 将输入向量转换为 NumPy 数组
    forward = np.array(forward_vector)
    up = np.array(up_vector)

    # 规范化输入向量
    forward /= np.linalg.norm(forward)
    up /= np.linalg.norm(up)

    # 计算右向量（right vector）
    right = np.cross(up, forward)
    right /= np.linalg.norm(right)

    # 重新计算正交的上向量，确保与 forward 和 right 正交
    up = np.cross(forward, right)

    # 构建旋转矩阵
    rotation_matrix = np.stack((right, up, forward), axis=1)

    # 提取欧拉角（假设旋转顺序为 ZYX，即 yaw-pitch-roll）
    sy = np.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)

    singular = sy < 1e-6

    if not singular:
        yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        pitch = np.arctan2(-rotation_matrix[2, 0], sy)
        roll = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
    else:
        # 当 sy 约等于零时，出现万向节锁，此时 yaw 和 roll 互相关联
        yaw = np.arctan2(-rotation_matrix[0, 1], rotation_matrix[1, 1])
        pitch = np.arctan2(-rotation_matrix[2, 0], sy)
        roll = 0

    # 将弧度转换为度数
    roll_deg = np.degrees(roll)
    pitch_deg = np.degrees(pitch)
    yaw_deg = np.degrees(yaw)

    return roll_deg, pitch_deg, yaw_deg




# def vector_to_euler(forward_vector):
#     """
#     将方向向量转换为x y z夹角
#     """
#     x, y, z = forward_vector
    
#     xtheta = np.degrees(np.arctan2(y, x))
#     ytheta = np.degrees(np.arctan2(x, z))
#     ztheta = np.degrees(np.arctan2(y, x))
    
#     return xtheta, ytheta, ztheta


# OpenCV
# TODO model translation and euler, use these to choose the nearest camera
# if translation and euler is correct, then can use it; if not, the pose(either opencv or opengl) can't be used. except euler is wrong but the rotation matrix is correct (will it?)
# so translation is correct.
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

        # pose = opencv_to_opengl(pose) # 'gl2': here to convert to opengl

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


        # XXX will the two euler angle be the same?
        # # 计算前向向量的欧拉角，并保存
        roll_deg, pitch_deg, yaw_deg = vector_to_euler(forward_vector_world[:3], up_vector_world[:3]) # xyz
        euler_angles_all.append([roll_deg, pitch_deg, yaw_deg])

        # # 保存欧拉角
        save_data[f'forward_vector_euler_compute{idx}'] = [roll_deg, pitch_deg, yaw_deg]


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


# TODO decide alpha and beta
# def calculate_distance(input_pose, candidate_pose, alpha=1.0, beta=1.0):
#     input_translation, input_rotation = extract_pose_components(input_pose)
#     candidate_translation, candidate_rotation = extract_pose_components(candidate_pose)
    
#     # 欧氏距离计算平移差异
#     position_distance = np.linalg.norm(input_translation - candidate_translation)
    
#     # 旋转差异（两个旋转矩阵之间的角度）
#     rotation_diff = np.dot(input_rotation.T, candidate_rotation)
#     angle_distance = np.arccos(np.clip((np.trace(rotation_diff) - 1) / 2, -1.0, 1.0))
    
#     # 返回加权距离
#     return alpha * position_distance + beta * angle_distance



# this one work well.
# def calculate_distance(input_pose, candidate_pose, alpha=1.0, beta=1.0, target_euler=None, euler=None):
#     return np.abs(target_euler[1] - euler[1])


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




# here it is simply the smallest translation 
# when the cond rotations and novel rotation's differences are between some range, then choose the cond with smallest translation?
# def find_nearest_cond(target_pose, cond_poses, cond_idx):
#     '''
#     cond poses: a list
#     cond idx: a list, the raw idx of each cond pose
#     return the nearest cond pose of target pose, and the raw idx of it
#     opencv coord
#     '''
#     target_translation = target_pose[:3, 3]
#     cond_translations = np.array([pose[:3, 3] for pose in cond_poses])
#     distances = np.linalg.norm(cond_translations - target_translation, axis=1)
#     nearest_idx = np.argmin(distances)

#     nearest_cond_pose = cond_poses[nearest_idx]
#     nearest_cond_idx = cond_idx[nearest_idx]

#     return nearest_cond_pose, nearest_cond_idx



# TODO find the nearest camera pose
# TODO how to prove the correctness of camera pose? opencv or opengl?
# 如果计算不对pose，就没法知道nearest选的对不对
# 如果nearest选的不对，也没法判断pose算的对不对
def launch():
    
    img_path = 'data/image_test'
    cam_file = 'data/cameras.npz'
    output_path = 'data/image_output/35_gl11_50t_1r_new_camera'
    intrinsic_all, pose_all, euler_returned_all = get_camera_poses(cam_file)
    intrinsic = intrinsic_all[0]
    cond_poses, cond_idx = get_cond_camera_poses(pose_all)
    target_poses, target_idx = get_target_camera_poses(pose_all)
    cond_euler = [euler_returned_all[i] for i in cond_idx]

    f_x = intrinsic[0, 0]  # intrinsic matrix (K) 的 [0, 0] 位置是 f_x
    image_width = 256
    fov_horizontal = 2 * np.arctan(image_width / (2 * f_x)) * 180 / np.pi  # rad -> degree
    fov_tensor = torch.from_numpy(np.array([fov_horizontal])).cuda().to(torch.float32)

    for cond_i, cond_idx_value in enumerate(cond_idx):
        cond_img_path = os.path.join(img_path, f'{cond_idx_value:06d}_rgb.png')
        cond_image_pil = Image.open(cond_img_path).convert("RGB")
        cond_image_pil = cond_image_pil.resize((256, 256))
        if not os.path.exists(os.path.join(output_path, 'cond')):
            os.makedirs(os.path.join(output_path, 'cond'))
        cond_image_output_path = os.path.join(output_path, 'cond')
        cond_save_path = os.path.join(cond_image_output_path, f'{cond_i}_{cond_idx_value:06d}_cond.png')
        cond_image_pil.save(cond_save_path)
    print(f"Saved all condition images to: {cond_image_output_path}")

    for i, idx in enumerate(target_idx):
        if i % 3 == 0:
            target_pose = pose_all[idx]
            target_euler = euler_returned_all[idx]
            nearest_cond_pose, nearest_cond_idx = find_nearest_cond(target_pose, cond_poses, cond_idx, alpha=50, beta=1, target_euler=target_euler, euler_all=cond_euler) # XXX
            # nearest_cond_pose, nearest_cond_idx = find_nearest_cond(target_pose, cond_poses, cond_idx)

            # target_pose_gl = opencv_to_opengl(target_pose)
            # nearest_cond_pose_gl = opencv_to_opengl(nearest_cond_pose)
            target_pose_gl = target_pose
            nearest_cond_pose_gl = nearest_cond_pose

            cond_img_path = os.path.join(img_path, f'{nearest_cond_idx:06d}_rgb.png')
            # image_path = "data/image_test/000000_rgb.png" # 384x384, Expected size 32 but got size 48
            

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
        

            cond_image_pil = Image.open(cond_img_path).convert("RGB")
            cond_image_pil = cond_image_pil.resize((256, 256)) # XXX
            cond_image = torch.from_numpy(np.array(cond_image_pil)).cuda() / 255.

            c_crossattn, c_concat = guidance.get_img_embeds(
                cond_image.permute((2, 0, 1))[None]) # change (H, W, C) to (C, H, W)
            
            cond_camera = nearest_cond_pose_gl
            target_camera = target_pose_gl
            target_camera = torch.from_numpy(target_camera[None]).cuda().to(torch.float32)
            cond_camera = torch.from_numpy(cond_camera[None]).cuda().to(torch.float32)
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
            print("------crossattn----------")
            print(c_crossattn.shape) # (1,1,768)
            print("--------concat------------")
            print(c_concat.shape) # (1,4,48,48) Expected size 32 but got size 48 for tensor number 1 in the list.

            novel_view = guidance.gen_from_cond(cond) 
            novel_view_pil = Image.fromarray(np.clip(novel_view[0]*255, 0, 255).astype(np.uint8))
            # display(cond_image_pil)
            # display(novel_view_pil)

            target_path = os.path.join(img_path, f'{idx:06d}_rgb.png')
            print("target_path=",target_path)
            target_pil = Image.open(target_path).convert("RGB").resize((256, 256))

            # concatenate images
            cond_image_array = np.array(cond_image_pil)
            target_image_array = np.array(target_pil)
            novel_image_array = np.array(novel_view_pil)

            # min_height = min(cond_image_array.shape[0], target_image_array.shape[0], novel_image_array.shape[0])
            # cond_image_array = cond_image_array[:min_height]
            # target_image_array = target_image_array[:min_height]
            # novel_image_array = novel_image_array[:min_height]

            concatenated = np.hstack((cond_image_array, target_image_array, novel_image_array))
            concatenated_image = Image.fromarray(concatenated)
            if not os.path.exists(os.path.join(output_path, 'concat')):
                os.makedirs(os.path.join(output_path, 'concat'))
            concatenated_save_path = os.path.join(output_path, f'concat/{i}_t{idx:03d}_c{nearest_cond_idx:03d}_concatenated.png')
            concatenated_image.save(concatenated_save_path)


            # cond_save_path = os.path.join(output_path, f'{i}_{nearest_cond_idx:06d}_cond.png')
            # novel_save_path = os.path.join(output_path, f'{i}_{idx:06d}_novel.png')
            # cond_image_pil.save(cond_save_path)
            # novel_view_pil.save(novel_save_path)

        


if __name__ == '__main__':
    launch()