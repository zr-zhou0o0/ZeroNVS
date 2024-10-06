
# Workflow --all
**collaborated with GPT-4O**

- run
  - launch_inference.sh ->
  - launch.py ->
  - trainer.fit ->
  - system = zero123system: base3dliftsystem: basesystem: lightningmodule

- Input and Dataloader:
  - Single 2D Image -> datamodule: image/single-image-module
  - Camera Parameters: Depth maps, extrinsic matrix(elevation, azimuth, radius), are generated to set up different viewing angles.
    - use the novel camera parameterization 6DoF+1.viewer ensure accurate representation of real-world scenes
  - field-of-view data.
  - Scene Data: use pretrained models from datasets like CO3D, RealEstate10k, and ACID to handle in-the-wild scenes

- Diffusion Model: A 2D conditional diffusion model synthesizes novel views as anchoring views

- NeRF: these synthesized novel views are converted into 3D-consistent NeRF representation
  - SDS Distillation: calculate a loss between NeRF's synthesized images and diffusion model's synthesized images. then gradient discent and update NeRF's parameters 

- SDS Anchoring: Anchoring is used to generate multiple plausible novel views and increase diversity.
  - ??? what position at the workflow?
  - my task is trying to find some good anchoring views???

- Output:
  - multiple new wiews images(rendered images for novel perspective)
  - video
  - 3D-Consistent NeRFs: the model distills novel views into NeRFs for 3D-consistent outputs
    - NeRF is sparse view or dense view???
    - NeRF output is what representation? or sdf?







# questions and note
- Pytorch Lightning: a lib for separating research code from engineering code.
  - encapsulates operations such as model training, validation, testing, and prediction
  - core components: LightningModule. Each model needs to inherit the LightningModule class and implement some key methods. responsible for defining the structure of the model, forward propagation, optimizer, etc.
- why two config? zeronvs config and zero123 config?
- $ in .sh is reference




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
  - do not train SDS / nerf
  - follow the github link to ban SDS
  - get familiar with the code and project





# error
## 1
- when choose --train, error: cuda out of memory
- change the batchsize in config.yaml

## 2
- when choose --export
- system.context_type = cuda
