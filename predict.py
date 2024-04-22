from cog import BasePredictor, Input, Path
import sys
import os
import copy
import torch
import time
import random
import mimetypes
import subprocess
import numpy as np
from PIL import Image
from typing import List
import torch.cuda.amp as amp
import torchvision.transforms as T

mimetypes.add_type("image/webp", ".webp")


sys.path.append("flashface/all_finetune")
from config import cfg
from models import sd_v1_ref_unet
from ops.context_diffusion import ContextGaussianDiffusion
from utils import Compose, PadToSquare, get_padding, seed_everything

from ldm import data, models, ops
from ldm.models.retinaface import crop_face, retinaface
from ldm.models.vae import sd_v1_vae

from PIL import Image, ImageDraw

MODEL_CACHE = "cache"


def download_weights(url, dest):
    start = time.time()
    print("[!] Initiating download from URL: ", url)
    print("[~] Destination path: ", dest)
    command = ["pget", "-vf", url, dest]
    if ".tar" in url:
        command.append("-x")
    try:
        subprocess.check_call(command, close_fds=False)
    except subprocess.CalledProcessError as e:
        print(
            f"[ERROR] Failed to download weights. Command '{' '.join(e.cmd)}' returned non-zero exit status {e.returncode}."
        )
        raise
    print("[+] Download completed in: ", time.time() - start, "seconds")


class Predictor(BasePredictor):
    def setup(self) -> None:
        # Load the necessary models and weights
        urls = [
            "https://weights.replicate.delivery/default/FlashFace/cache/README.md",
            "https://weights.replicate.delivery/default/FlashFace/cache/bpe_simple_vocab_16e6.txt.gz",
            "https://weights.replicate.delivery/default/FlashFace/cache/flashface.ckpt",
            "https://weights.replicate.delivery/default/FlashFace/cache/openai-clip-vit-large-14.pth",
            "https://weights.replicate.delivery/default/FlashFace/cache/retinaface_resnet50.pth",
            "https://weights.replicate.delivery/default/FlashFace/cache/sd-v1-vae.pth",
        ]

        if not os.path.exists(MODEL_CACHE):
            os.makedirs(MODEL_CACHE)

        for url in urls:
            filename = url.split("/")[-1]
            dest_path = os.path.join(MODEL_CACHE, filename)
            if not os.path.exists(dest_path):
                download_weights(url, dest_path)

        # Initialize variables and models
        self.gpu = "cuda"
        self.padding_to_square = PadToSquare(224)
        self.retinaface_transforms = T.Compose([PadToSquare(size=640), T.ToTensor()])
        self.retinaface = (
            retinaface(pretrained=True, device=self.gpu).eval().requires_grad_(False)
        )
        self.face_transforms = Compose(
            [T.ToTensor(), T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]
        )

        self.clip_tokenizer = data.CLIPTokenizer(padding="eos")
        self.clip = (
            getattr(models, cfg.clip_model)(pretrained=True)
            .eval()
            .requires_grad_(False)
            .textual.to(self.gpu)
        )
        self.autoencoder = (
            sd_v1_vae(pretrained=True).eval().requires_grad_(False).to(self.gpu)
        )

        self.unet = sd_v1_ref_unet(
            pretrained=True, version="sd-v1-5_nonema", enable_encoder=False
        ).to(self.gpu)

        self.unet.replace_input_conv()
        self.unet = self.unet.eval().requires_grad_(False).to(self.gpu)
        self.unet.share_cache["num_pairs"] = cfg.num_pairs

        # Load model weights
        weight_path = f"./{MODEL_CACHE}/flashface.ckpt"
        model_weight = torch.load(weight_path, map_location="cpu")
        msg = self.unet.load_state_dict(model_weight, strict=True)
        print(msg)

        # Initialize diffusion
        sigmas = ops.noise_schedule(
            schedule=cfg.schedule,
            n=cfg.num_timesteps,
            beta_min=cfg.scale_min,
            beta_max=cfg.scale_max,
        )
        self.diffusion = ContextGaussianDiffusion(
            sigmas=sigmas, prediction_type=cfg.prediction_type
        )
        self.diffusion.num_pairs = cfg.num_pairs

    def detect_face(self, imgs=None):
        # read images
        pil_imgs = imgs
        b = len(pil_imgs)
        vis_pil_imgs = copy.deepcopy(pil_imgs)

        # convert RGBA images to RGB
        for i in range(b):
            if pil_imgs[i].mode == "RGBA":
                pil_imgs[i] = pil_imgs[i].convert("RGB")

        # detection
        imgs = torch.stack([self.retinaface_transforms(u) for u in pil_imgs]).to(
            self.gpu
        )
        boxes, kpts = self.retinaface.detect(imgs, min_thr=0.6)

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

    def encode_text(self, m, x):
        # embeddings
        x = m.token_embedding(x) + m.pos_embedding

        # transformer
        for block in m.transformer:
            x = block(x)

        # output
        x = m.norm(x)

        return x

    def generate(
        self,
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
            reference_faces = self.detect_face(reference_faces)

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
            face_bbox[1] : face_bbox[1] + max_size,
            face_bbox[0] : face_bbox[0] + max_size,
        ] = 1

        empty_mask = empty_mask[::8, ::8].cuda()
        empty_mask = empty_mask[None].repeat(num_sample, 1, 1)

        pasted_ref_faces = []
        show_refs = []
        for ref_img in reference_faces:
            ref_img = ref_img.convert("RGB")
            ref_img = self.padding_to_square(ref_img)
            to_paste = ref_img

            to_paste = self.face_transforms(to_paste)
            pasted_ref_faces.append(to_paste)

        faces = torch.stack(pasted_ref_faces, dim=0).to(self.gpu)

        c = self.encode_text(self.clip, self.clip_tokenizer([pos_prompt]).to(self.gpu))
        c = c[None].repeat(num_sample, 1, 1, 1).flatten(0, 1)
        c = {"context": c}

        single_null_context = self.encode_text(
            self.clip, self.clip_tokenizer([neg_prompt]).cuda()
        ).to(self.gpu)
        null_context = single_null_context
        nc = {"context": null_context[None].repeat(num_sample, 1, 1, 1).flatten(0, 1)}

        ref_z0 = cfg.ae_scale * torch.cat(
            [
                self.autoencoder.sample(u, deterministic=True)
                for u in faces.split(cfg.ae_batch_size)
            ]
        )
        self.unet.share_cache["num_pairs"] = 4
        self.unet.share_cache["ref"] = ref_z0
        self.unet.share_cache["similarity"] = torch.tensor(lamda_feat).cuda()
        self.unet.share_cache["ori_similarity"] = torch.tensor(lamda_feat).cuda()
        self.unet.share_cache["lamda_feat_before_ref_guidence"] = torch.tensor(
            lamda_feat_before_ref_guidence
        ).cuda()
        self.unet.share_cache["ref_context"] = single_null_context.repeat(
            len(ref_z0), 1, 1
        )
        self.unet.share_cache["masks"] = empty_mask
        self.unet.share_cache["classifier"] = face_guidence
        self.unet.share_cache["step_to_launch_face_guidence"] = (
            step_to_launch_face_guidence
        )

        self.diffusion.classifier = face_guidence

        # sample
        with amp.autocast(dtype=cfg.flash_dtype), torch.no_grad():
            z0 = self.diffusion.sample(
                solver=solver,
                noise=torch.empty(
                    num_sample, 4, 768 // 8, 768 // 8, device=self.gpu
                ).normal_(),
                model=self.unet,
                model_kwargs=[c, nc],
                steps=steps,
                guide_scale=text_control_scale,
                guide_rescale=0.5,
                show_progress=True,
                discretization=cfg.discretization,
            )

        imgs = self.autoencoder.decode(z0 / cfg.ae_scale)
        del self.unet.share_cache["ori_similarity"]
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

    def predict(
        self,
        positive_prompt: str = Input(description="Positive prompt"),
        negative_prompt: str = Input(description="Negative prompt", default="nsfw"),
        steps: int = Input(description="Number of steps", default=35),
        face_bounding_box: str = Input(
            description="Face position", default="[0., 0., 0., 0.]"
        ),
        lamda_feature: float = Input(
            description="Reference feature strength", default=0.9
        ),
        face_guidance: float = Input(
            description="Reference guidance strength", default=2.2
        ),
        num_sample: int = Input(description="Number of generated images", default=1),
        text_control_scale: float = Input(
            description="Text guidance strength", default=7.5
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
        step_to_launch_face_guidance: int = Input(
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
        default_position_prompt: str = Input(
            description="Default positive prompt postfix",
            default="best quality, masterpiece,ultra-detailed, UHD 4K, photographic",
        ),
        default_negative_prompt: str = Input(
            description="Default negative prompt postfix",
            default="blurry, ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, extra limbs, disfigured, deformed, body out of frame, bad anatomy, watermark, signature, cut off, low contrast, underexposed, overexposed, bad art, beginner, amateur, distorted face",
        ),
        output_format: str = Input(
            description="Format of the output images",
            choices=["webp", "jpg", "png"],
            default="webp",
        ),
        output_quality: int = Input(
            description="Quality of the output images, from 0 to 100. 100 is best quality, 1 is lowest quality.",
            default=80,
            ge=1,
            le=100,
        ),
    ) -> List[Path]:

        pos_prompt = positive_prompt
        neg_prompt = negative_prompt
        face_bbox = face_bounding_box
        face_guidence = face_guidance
        step_to_launch_face_guidence = step_to_launch_face_guidance
        lamda_feat = lamda_feature
        default_pos_prompt = default_position_prompt
        default_neg_prompt = default_negative_prompt

        if seed is None:
            seed = -1

        reference_face_1 = Image.open(str(reference_face_1)) if reference_face_1 else None
        reference_face_2 = Image.open(str(reference_face_2)) if reference_face_2 else None
        reference_face_3 = Image.open(str(reference_face_3)) if reference_face_3 else None
        reference_face_4 = Image.open(str(reference_face_4)) if reference_face_4 else None

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

        faces = self.generate(
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

        saved_image_paths = []
        for index, face in enumerate(faces):
            extension = output_format.lower()
            extension = "jpeg" if extension == "jpg" else extension
            image_path = f"/tmp/image_{index}.{extension}"  # Adjusted path format to include variable extension

            print(f"[~] Saving to {image_path}...")
            print(f"[~] Output format: {extension.upper()}")
            if output_format != "png":
                print(f"[~] Output quality: {output_quality}")

            save_params = {"format": extension.upper()}
            if output_format != "png":
                save_params["quality"] = output_quality
                save_params["optimize"] = True

            face.save(image_path, **save_params)
            print(f"Saved image {index} at {image_path}")
            saved_image_paths.append(image_path)

        return [Path(f) for f in saved_image_paths]
