# coordinate system used for cond_camera and target_camera: opengl


# similar to: threestudio/models/guidance/zero123_guidance.py
from ldm.models.diffusion import options

options.LDM_DISTILLATION_ONLY = True

from threestudio.models.guidance import zero123_guidance
from omegaconf import OmegaConf 
# omegaconf is a library for parsing config such as yaml or dictionary.
# which supports merge, load, override...


# reference launch_inference.sh
# zeronvs_config.yaml is the pretrained config, zero123_scene.yaml is the main config
# launch_inference.sh will override some values in zero123_scene
image_path = "data/image_test/000000_rgb.png" # 384x384, Expected size 32 but got size 48
# image_path = "rs_dtu_4/DTU/scan6/image/000000.png" # 400x300, Expected size 32 but got size 37
guidance_cfg = dict(
    pretrained_model_name_or_path= "zeronvs.ckpt", # XXX 
    pretrained_config= "zeronvs_config.yaml", # XXX 
    guidance_scale= 7.5,
    cond_image_path =image_path,
    min_step_percent=[0,.75,.02,1000],
    max_step_percent=[1000, 0.98, 0.025, 2500],
    vram_O=False # whether optimize ram use
)

guidance = zero123_guidance.Zero123Guidance(OmegaConf.create(guidance_cfg))
from PIL import Image
import numpy as np
import torch

cond_image_pil = Image.open(image_path).convert("RGB")
cond_image_pil = cond_image_pil.resize((256, 256)) # XXX
cond_image = torch.from_numpy(np.array(cond_image_pil)).cuda() / 255.

c_crossattn, c_concat = guidance.get_img_embeds(
    cond_image.permute((2, 0, 1))[None]) # change (H, W, C) to (C, H, W)

# XXX HERE conditional camera and target camera
cond_camera = np.eye(4)  # identity camera pose
target_camera = cond_camera.copy()
target_camera[:3, -1] = np.array([.125, .125, .125])  # perturb the cond pose

target_camera = torch.from_numpy(target_camera[None]).cuda().to(torch.float32)
cond_camera = torch.from_numpy(cond_camera[None]).cuda().to(torch.float32)
camera_batch = {
    "target_cam2world": target_camera,
    "cond_cam2world": cond_camera,
    "fov_deg": torch.from_numpy(np.array([45.0])).cuda().to(torch.float32) # what is this?
}

guidance.cfg.precomputed_scale=.7
cond = guidance.get_cond_from_known_camera(
    camera_batch,
    c_crossattn=c_crossattn,
    c_concat=c_concat,
    # precomputed_scale=.7,
)

print("------camerabatch--------")
print(camera_batch["cond_cam2world"].shape) # (1,4,4)
print(camera_batch["target_cam2world"].shape) # (1,4,4)
print("------crossattn----------")
print(c_crossattn.shape) # (1,1,768)
print("--------concat------------")
print(c_concat.shape) # (1,4,48,48) Expected size 32 but got size 48 for tensor number 1 in the list.

novel_view = guidance.gen_from_cond(cond) # BUG 
novel_view_pil = Image.fromarray(np.clip(novel_view[0]*255, 0, 255).astype(np.uint8))
# display(cond_image_pil)
# display(novel_view_pil)

cond_image_pil.save("data/image_output/cond.png")
novel_view_pil.save("data/image_output/novel.png")