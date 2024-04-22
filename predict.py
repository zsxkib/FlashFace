from cog import BasePredictor, Input, Path
from typing import List
from PIL import Image
import copy
import random
import sys
import os
import numpy as np
import torch
import torch.cuda.amp as amp
import torchvision.transforms as T


sys.path.append("flashface/all_finetune")
from config import cfg
from models import sd_v1_ref_unet
from ops.context_diffusion import ContextGaussianDiffusion
from utils import Compose, PadToSquare, get_padding, seed_everything

from ldm import data, models, ops
from ldm.models.retinaface import crop_face, retinaface
from ldm.models.vae import sd_v1_vae

from PIL import Image, ImageDraw

# model path
SKIP_LOAD = False
DEBUG_VIEW = False
SKEP_LOAD = False
LOAD_FLAG = True
DEFAULT_INPUT_IMAGES = 4
MAX_INPUT_IMAGES = 4
SIZE = 768
with_lora = False
enable_encoder = False

weight_path = "./cache/flashface.ckpt"

gpu = "cuda"

padding_to_square = PadToSquare(224)

retinaface_transforms = T.Compose([PadToSquare(size=640), T.ToTensor()])

retinaface = retinaface(pretrained=True, device="cuda").eval().requires_grad_(False)


def detect_face(imgs=None):

    # read images
    pil_imgs = imgs
    b = len(pil_imgs)
    vis_pil_imgs = copy.deepcopy(pil_imgs)

    # detection
    imgs = torch.stack([retinaface_transforms(u) for u in pil_imgs]).to(gpu)
    boxes, kpts = retinaface.detect(imgs, min_thr=0.6)

    # undo padding and scaling
    face_imgs = []

    for i in range(b):
        # params
        scale = 640 / max(pil_imgs[i].size)
        left, top, _, _ = get_padding(
            round(scale * pil_imgs[i].width), round(scale * pil_imgs[i].height), 640
        )

        # undo padding
        boxes[i][:, [0, 2]] -= left
        boxes[i][:, [1, 3]] -= top
        kpts[i][:, :, 0] -= left
        kpts[i][:, :, 1] -= top

        # undo scaling
        boxes[i][:, :4] /= scale
        kpts[i][:, :, :2] /= scale

        # crop faces
        crops = crop_face(pil_imgs[i], boxes[i], kpts[i])
        if len(crops) != 1:
            raise ValueError(
                f"Found {len(crops)} faces in image {i+1}, please ensure there is only one face in each image"
            )

        face_imgs += crops

        # draw boxes on the pil image
        draw = ImageDraw.Draw(vis_pil_imgs[i])
        for box in boxes[i]:
            box = box[:4].tolist()
            box = [int(x) for x in box]
            draw.rectangle(box, outline="red", width=4)

    face_imgs = face_imgs

    return face_imgs


if not DEBUG_VIEW and not SKEP_LOAD:
    clip_tokenizer = data.CLIPTokenizer(padding="eos")
    clip = (
        getattr(models, cfg.clip_model)(pretrained=True)
        .eval()
        .requires_grad_(False)
        .textual.to(gpu)
    )
    autoencoder = sd_v1_vae(pretrained=True).eval().requires_grad_(False).to(gpu)

    unet = sd_v1_ref_unet(
        pretrained=True, version="sd-v1-5_nonema", enable_encoder=enable_encoder
    ).to(gpu)

    unet.replace_input_conv()
    unet = unet.eval().requires_grad_(False).to(gpu)
    unet.share_cache["num_pairs"] = cfg.num_pairs

    if LOAD_FLAG:
        model_weight = torch.load(weight_path, map_location="cpu")
        msg = unet.load_state_dict(model_weight, strict=True)
        print(msg)

    # diffusion
    sigmas = ops.noise_schedule(
        schedule=cfg.schedule,
        n=cfg.num_timesteps,
        beta_min=cfg.scale_min,
        beta_max=cfg.scale_max,
    )
    diffusion = ContextGaussianDiffusion(
        sigmas=sigmas, prediction_type=cfg.prediction_type
    )
    diffusion.num_pairs = cfg.num_pairs

face_transforms = Compose(
    [T.ToTensor(), T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]
)


def encode_text(m, x):
    # embeddings
    x = m.token_embedding(x) + m.pos_embedding

    # transformer
    for block in m.transformer:
        x = block(x)

    # output
    x = m.norm(x)

    return x


def generate(
    pos_prompt,
    neg_prompt,
    steps=35,
    face_bbox=[0.0, 0.0, 0.0, 0.0],
    lamda_feat=0.9,
    face_guidence=2.2,
    num_sample=1,
    text_control_scale=7.5,
    seed=-1,
    step_to_launch_face_guidence=600,
    reference_face_1=None,
    reference_face_2=None,
    reference_face_3=None,
    reference_face_4=None,
    default_pos_prompt="best quality, masterpiece,ultra-detailed, UHD 4K, photographic",
    default_neg_prompt="blurry, ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, extra limbs, disfigured, deformed, body out of frame, bad anatomy, watermark, signature, cut off, low contrast, underexposed, overexposed, bad art, beginner, amateur, distorted face",
    need_detect=True,
    lamda_feat_before_ref_guidence=0.85,
):
    reference_faces = [
        reference_face_1,
        reference_face_2,
        reference_face_3,
        reference_face_4,
    ]
    # filter none
    reference_faces = [ref for ref in reference_faces if ref is not None]
    solver = "ddim"
    if default_pos_prompt is not None:
        pos_prompt = pos_prompt + ", " + default_pos_prompt
    if neg_prompt is not None and len(neg_prompt) > 0:
        neg_prompt = neg_prompt + ", " + default_neg_prompt
    else:
        neg_prompt = default_neg_prompt
    if seed == -1:
        seed = random.randint(0, 2147483647)
    print(seed)
    seed_everything(seed)
    print("final pos_prompt: ", pos_prompt)
    print("final neg_prompt: ", neg_prompt)

    if need_detect:
        reference_faces = detect_face(reference_faces)

        # for i, ref_img in enumerate(reference_faces):
        #     ref_img.save(f'./{i + 1}.png')
        print(f"detected {len(reference_faces)} faces")
        if len(reference_faces) == 0:
            raise ValueError(
                "No face detected in the reference images, please upload images with clear face"
            )

        if len(reference_faces) < 4:
            expand_reference_faces = copy.deepcopy(reference_faces)
            while len(expand_reference_faces) < 4:
                # random select from ref_imgs
                expand_reference_faces.append(random.choice(reference_faces))
            reference_faces = expand_reference_faces

    # process the ref_imgs
    H = W = 768
    if isinstance(face_bbox, str):
        face_bbox = eval(face_bbox)
    normalized_bbox = face_bbox
    print(normalized_bbox)
    face_bbox = [
        int(normalized_bbox[0] * W),
        int(normalized_bbox[1] * H),
        int(normalized_bbox[2] * W),
        int(normalized_bbox[3] * H),
    ]
    max_size = max(face_bbox[2] - face_bbox[1], face_bbox[3] - face_bbox[1])
    empty_mask = torch.zeros((H, W))

    empty_mask[
        face_bbox[1] : face_bbox[1] + max_size, face_bbox[0] : face_bbox[0] + max_size
    ] = 1

    empty_mask = empty_mask[::8, ::8].cuda()
    empty_mask = empty_mask[None].repeat(num_sample, 1, 1)

    pasted_ref_faces = []
    show_refs = []
    for ref_img in reference_faces:
        ref_img = ref_img.convert("RGB")
        ref_img = padding_to_square(ref_img)
        to_paste = ref_img

        to_paste = face_transforms(to_paste)
        pasted_ref_faces.append(to_paste)

    faces = torch.stack(pasted_ref_faces, dim=0).to(gpu)

    c = encode_text(clip, clip_tokenizer([pos_prompt]).to(gpu))
    c = c[None].repeat(num_sample, 1, 1, 1).flatten(0, 1)
    c = {"context": c}

    single_null_context = encode_text(clip, clip_tokenizer([neg_prompt]).cuda()).to(gpu)
    null_context = single_null_context
    nc = {"context": null_context[None].repeat(num_sample, 1, 1, 1).flatten(0, 1)}

    ref_z0 = cfg.ae_scale * torch.cat(
        [
            autoencoder.sample(u, deterministic=True)
            for u in faces.split(cfg.ae_batch_size)
        ]
    )
    #  ref_z0 = ref_z0[None].repeat(num_sample, 1,1,1,1).flatten(0,1)
    unet.share_cache["num_pairs"] = 4
    unet.share_cache["ref"] = ref_z0
    unet.share_cache["similarity"] = torch.tensor(lamda_feat).cuda()
    unet.share_cache["ori_similarity"] = torch.tensor(lamda_feat).cuda()
    unet.share_cache["lamda_feat_before_ref_guidence"] = torch.tensor(
        lamda_feat_before_ref_guidence
    ).cuda()
    unet.share_cache["ref_context"] = single_null_context.repeat(len(ref_z0), 1, 1)
    unet.share_cache["masks"] = empty_mask
    unet.share_cache["classifier"] = face_guidence
    unet.share_cache["step_to_launch_face_guidence"] = step_to_launch_face_guidence

    diffusion.classifier = face_guidence

    # sample
    with amp.autocast(dtype=cfg.flash_dtype), torch.no_grad():
        z0 = diffusion.sample(
            solver=solver,
            noise=torch.empty(num_sample, 4, 768 // 8, 768 // 8, device=gpu).normal_(),
            model=unet,
            model_kwargs=[c, nc],
            steps=steps,
            guide_scale=text_control_scale,
            guide_rescale=0.5,
            show_progress=True,
            discretization=cfg.discretization,
        )

    imgs = autoencoder.decode(z0 / cfg.ae_scale)
    del unet.share_cache["ori_similarity"]
    # output
    imgs = (
        (imgs.permute(0, 2, 3, 1) * 127.5 + 127.5)
        .cpu()
        .numpy()
        .clip(0, 255)
        .astype(np.uint8)
    )

    # convert to PIL image
    imgs = [Image.fromarray(img) for img in imgs]
    imgs = imgs + show_refs

    return imgs


class Predictor(BasePredictor):
    def setup(self) -> None:
        # Load the necessary models and weights
        pass

    def predict(
        self,
        pos_prompt: str = Input(description="Positive prompt"),
        neg_prompt: str = Input(description="Negative prompt"),
        steps: int = Input(description="Number of steps", default=35),
        face_bbox: str = Input(description="Face position", default="[0., 0., 0., 0.]"),
        lamda_feat: float = Input(
            description="Reference feature strength", default=0.9
        ),
        face_guidence: float = Input(
            description="Reference guidance strength", default=2.2
        ),
        num_sample: int = Input(description="Number of generated images", default=1),
        text_control_scale: float = Input(
            description="Text guidance strength", default=7.5
        ),
        seed: int = Input(description="Seed", default=-1),
        step_to_launch_face_guidence: int = Input(
            description="Step index to launch reference guidance", default=600
        ),
        reference_face_1: Path = Input(description="Reference face image 1"),
        reference_face_2: Path = Input(
            description="Reference face image 2", default=None
        ),
        reference_face_3: Path = Input(
            description="Reference face image 3", default=None
        ),
        reference_face_4: Path = Input(
            description="Reference face image 4", default=None
        ),
        default_pos_prompt: str = Input(
            description="Default positive prompt postfix",
            default="best quality, masterpiece,ultra-detailed, UHD 4K, photographic",
        ),
        default_neg_prompt: str = Input(
            description="Default negative prompt postfix",
            default="blurry, ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, extra limbs, disfigured, deformed, body out of frame, bad anatomy, watermark, signature, cut off, low contrast, underexposed, overexposed, bad art, beginner, amateur, distorted face",
        ),
    ) -> List[Path]:

        reference_face_1 = Image.open(str(reference_face_1)) if reference_face_1 is not None else None
        reference_face_2 = Image.open(str(reference_face_2)) if reference_face_2 is not None else None
        reference_face_3 = Image.open(str(reference_face_3)) if reference_face_3 is not None else None
        reference_face_4 = Image.open(str(reference_face_4)) if reference_face_4 is not None else None

        print(f"[!] ({type(pos_prompt)}) pos_prompt={pos_prompt}")
        print(f"[!] ({type(neg_prompt)}) neg_prompt={neg_prompt}")
        print(f"[!] ({type(steps)}) steps={steps}")
        print(f"[!] ({type(face_bbox)}) face_bbox={face_bbox}")
        print(f"[!] ({type(lamda_feat)}) lamda_feat={lamda_feat}")
        print(f"[!] ({type(face_guidence)}) face_guidence={face_guidence}")
        print(f"[!] ({type(num_sample)}) num_sample={num_sample}")
        print(f"[!] ({type(text_control_scale)}) text_control_scale={text_control_scale}")
        print(f"[!] ({type(seed)}) seed={seed}")
        print(f"[!] ({type(step_to_launch_face_guidence)}) step_to_launch_face_guidence={step_to_launch_face_guidence}")
        print(f"[!] ({type(reference_face_1)}) reference_face_1={reference_face_1}")
        print(f"[!] ({type(reference_face_2)}) reference_face_2={reference_face_2}")
        print(f"[!] ({type(reference_face_3)}) reference_face_3={reference_face_3}")
        print(f"[!] ({type(reference_face_4)}) reference_face_4={reference_face_4}")
        print(f"[!] ({type(default_pos_prompt)}) default_pos_prompt={default_pos_prompt}")
        print(f"[!] ({type(default_neg_prompt)}) default_neg_prompt={default_neg_prompt}")

        return generate(
            pos_prompt=pos_prompt,
            neg_prompt=neg_prompt,
            steps=steps,
            face_bbox=face_bbox,
            lamda_feat=lamda_feat,
            face_guidence=face_guidence,
            num_sample=num_sample,
            text_control_scale=text_control_scale,
            seed=seed,
            step_to_launch_face_guidence=step_to_launch_face_guidence,
            reference_face_1=reference_face_1,
            reference_face_2=reference_face_2,
            reference_face_3=reference_face_3,
            reference_face_4=reference_face_4,
            default_pos_prompt=default_pos_prompt,
            default_neg_prompt=default_neg_prompt,
        )
