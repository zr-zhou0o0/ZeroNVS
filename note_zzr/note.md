
# Workflow --all
**collaborated with GPT-4O**

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







# ToDo
- for each novel camera pose find the nearest input camera pose, then use it generates novel view [ ]
  - find out diffusion inside ZeroNVS
  - use some 50 pics in dataset and try to generate the 51 pic
  - why 50 -> 1? not 1 -> 50?
  - do not train SDS / nerf
  - follow the github link to ban SDS
  - get familiar with the code and project
  
  - **but why use NeRF for novel view instead of straightly use diffusion?**
    - for higher image quality
  - **why then ban SDS and then then straightly use diffusion?**


1. For each novel camera pose: give a novel camera pose  
    we should find the nearest input camera pose: give many input camera pose paired with images
2. Generate the novel view use the nearest input image: use the nearest camera pose and the paired image(still a single image as condition) to generate images
3. (Use all images (include input images and generated images) to train NeRF filed through the SDS method.)

Use `python read_camera.py` to examine the data and become familiar with camera coordinates.

? what is scale mat / world mat?  
giving scale mat @ world mat = projection, scale mat is intrinsic and world mat is extrinsic?  
then why don't we get intrinsic straightly from scale mat?





# error
## 1
- when choose --train, error: cuda out of memory
- change the batchsize in config.yaml

## 2
- when choose --export
- system.context_type = cuda




world = world -> camera -> image = intrinsic * cw(E)