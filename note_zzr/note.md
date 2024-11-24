
# Workflow --all

- run
  - launch_inference.sh ->
  - launch.py ->
  - trainer.fit ->
  - system = zero123system: base3dliftsystem: basesystem: lightningmodule ->
    - carry out getting guidance from diffusion and train nerf
  - guidance = zero123_guidance in threestudio ->
  - diffusion in zeronvs_diffusion/zero123/ldm


- Input and Dataloader:
  - Single 2D Image -> datamodule: image/single-image-module
  - Camera Parameters: Depth maps, extrinsic matrix(elevation, azimuth, radius), are generated to set up different viewing angles.
    - use the novel camera parameterization 6DoF+1.viewer ensure accurate representation of real-world scenes
  - field-of-view data.
  - Scene Data: use pretrained models from datasets like CO3D, RealEstate10k, and ACID to handle in-the-wild scenes


- Diffusion Model: A 2D conditional diffusion model synthesizes novel views as anchoring views
  - use the zero123 pipeline to construct a camera-extrinsic-aware diffusion model
  - train: objaverse ... (irrelevant to my task)
  - sampling: 


- NeRF: these synthesized novel views could be converted into 3D-consistent NeRF representation, when doing 3D Reconstruction task.
  - train:
    - network input: x, y, z, theta, phi
    - network output: c, sigma
    - use volume-rendering equation, from c and sigma generate novel view image
    - compute loss between gt and syn, update the weights
  - for every object, use the several diffusion-base anchored images to train a nerf and generate novel view.
  - Here in zero123, it is a volume-radiance-field nerf???

  - SDS Distillation: calculate a loss between NeRF's synthesized images and diffusion model's synthesized images. then gradient discent and update NeRF's parameters 


- it uses SDS or SJC? which one?
- zero123 use SJC but zeronvs use SDS?
- in zeroNVS, use SDS, and deprecate the 3d reconstruction part in zero123?

- but why use nerf?
- diffusion can generate novel view independently. voxel radiance field is used for 3d reconstruction. 
- use the diffusion generated images to train a nerf, and use nerf to conveniently and efficiently generate more novel views??? for time-saving?

- SDS Anchoring: Anchoring is used to generate multiple plausible novel views and increase diversity.
  - ??? what position at the workflow?
  - my task is trying to find some good anchoring views???


- Output:
  - multiple new wiews images(rendered images for novel perspective)
  - video
  - 3D-Consistent NeRFs: the model distills novel views into NeRFs for 3D-consistent outputs
    - NeRF is sparse view or dense view???
    - NeRF output is what representation? or sdf?







# code structure

## threestudio
loaded before diffusion...
- models: nerf, sdf, ......
- systems
- scripts

## zeronvs_diffusion - zero123: diffusion part
- ldm: latent diffusion model
  - data: dataloader
  - models: diffusion: ddpm...
  - modules: 
    - diffusionmodules
    - distribution
    - encoder
    - evaluate
    - ...
  - thirdp - psp ?????  







# HaveDone
- for each novel camera pose find the nearest input camera pose, then use it generates novel view [ ]
  - find out diffusion inside ZeroNVS
  - use some 50 pics in dataset and try to generate the 51 pic
  - why 50 -> 1? not 1 -> 50?
  - do not train SDS / nerf
  - follow the github link to ban SDS
  - get familiar with the code and project
  
  - but why use NeRF for novel view instead of straightly use diffusion?
    - for higher image quality
  - why then ban SDS and then then straightly use diffusion?


# HaveDone
- for a single image in our dataset, render use SDS and use without SDS, and concat the same view into one image for compare.
  - try to ban the remove validation image folder command to get the source image. [-]
  - try to find the built-in camera pose in nerf sampling [-]
    - zero123 system?
    - zero123 guidance? x
    - uncond?
    - multiview?
    - zero123_scene.yaml?
    - zeronvs_config.yaml?
  - so if we just have one image as condition, the find-nearest-camera-pose function is useless.
  - we can simply pass the found camera pose in sds to the diffusion_guidance (can we?)(maybe add some processing process according to sds)
- note, we just need a single cond image can initialize zero123_guidance
- so we should choose a novel view camera pose in the sds first, and use the find-nearest func find a cond image and generate a diffusion-only image, and pass the cond image to generate a sds image. 
- another problem: different origin point choice. of sds and dif.


# ToDo
- take *rgb as conds, cameras.npz as cp, test-split-20 is camera pose need to be eval.
- generate eval_camera_pose as png




# error
## 1
- when choose --train, error: cuda out of memory
- change the batchsize in config.yaml

## 2
- when choose --export
- system.context_type = cuda




world = world -> camera -> image = intrinsic * cw(E)




# what
the simple and effective way is change the direction of y axis. so if i simply change the figure's y axis... the gl9 is correct! but i don't know what is the y axis of world coord in the zero123_diffusion and sds...test it!

the second question is, why change the R's y column have no effect???  
ah! that is beacuse, the forward_vector is just the third column's negative!  
e.g.  
[[-0.5000, -0.4460,  0.7423],  
[ 0.8660, -0.2575,  0.4286],  
[ 0.0000,  0.8572,  0.5150]]  
forward_vector = P @ [0, 0, -1]  
up_vector = P @ [0, 1, 0]  
right_vector = P @ [1, 0, 0]  


so the answer is obvious: the world coordinate system's y is inversed. not the camera coordinate system's y is inversed.  
if change camera's y, only flip over the camera vector itself in place.  
change the world's y, the forward vector will flip over to a different angle.  


so, the data/view_camera/15_gl11_y-x.png prove that, the translation and the rotation recorded are both inaccurate.  
the recorded camera pose's horizontal angle is righter than real camera pose's euler angle. 



50~/80~/use.


# Two Ways for comparison
- a given cond image, and two pose: one default, one somelike 30 degree. without coordinate system converting. [-]
- a required target pose in dataset's coordinate system. 
  - use find_nearest_camera to get a cond image.
  - use the cond image to generate noSDS image.
  - use $P_n(default) = M_nd * P_d(cond)$ get $M_nd$, then $P_n(target) = M_nd * P_d(target)$ to get $P_n(target)$. where n is nerf coordinate, d is dataset coordinate. [-]
  - change the validate process in zero123, let it generates the target coordinate.




# the coordinate system convert
- translation scale! however we will never know the 'r' in sds when training has what measurment unit(maybe meter?)
- and we also don't know what measurment unit is in our dataset.(hopefully still meter?)
- to guess is the best way... 1:10???


gl3 is the right one!!!
so i have tried manymanymany converter, and the first one indeed is the right one......







# no hang up
nohup python -u "/home/stu7/projects/ZeroNVS/launch_noSDS_random_all.py" > data/nohup_command_outputs/out_10_22_12_28.log 2>&1 &

nohup bash launch_inference.sh > data/nohup_command_outputs/out_11_8_23_09.log 2>&1 &

ps -aux | grep nohup

kill -9 [pid]m