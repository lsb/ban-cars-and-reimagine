from io import BytesIO
from pathlib import Path
import torch
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from skimage.feature import canny

from diffusers import AutoPipelineForInpainting
from PIL import Image, ImageFilter
import numpy as np
import time
from huggingface_hub import snapshot_download

segformer_model = "nvidia/segformer-b0-finetuned-cityscapes-768-768"
sdinpaint_model = "stable-diffusion-v1-5/stable-diffusion-inpainting"

ctlnetmodelname = "IDKiro/sdxs-512-dreamshaper-sketch"
sdmodelname = "IDKiro/sdxs-512-dreamshaper"


import modal
from data_uri_parser import DataURI

container_image = modal.Image.debian_slim().pip_install("diffusers", "transformers", "torch", "accelerate", "data_uri_parser", "scikit-image") #, "fastapi", "weave")

app = modal.App("segformer-sdinpainting", image=container_image)

@app.cls(gpu=modal.gpu.H100(), container_idle_timeout=60, image=container_image)
class Model:
    @modal.build()
    def build(self):
        snapshot_download(segformer_model)
        snapshot_download(sdinpaint_model)
        snapshot_download(ctlnetmodelname)
        snapshot_download(sdmodelname)
    
    @modal.enter()
    def enter(self):
        # os.environ['WANDB_API_KEY'] = 'a61467d79ff707c7496c5c9f0a74e510817bb7d2'
        self.segprocessor = SegformerImageProcessor.from_pretrained(
            segformer_model,
            local_files_only=True,
        )
        self.segmodel = SegformerForSemanticSegmentation.from_pretrained(
            segformer_model,
            local_files_only=True,
        )
        self.segmodel.to('cuda')
        # self.segmodel.to(torch.float16)
        # self.inpainter =  AutoPipelineForInpainting.from_pretrained(
        #     sdinpaint_model,
        #     variant='fp16',
        #     torch_dtype=torch.float16,
        #     local_files_only=True,
        #     safety_checker=None,
        # )
        # self.inpainter.to('cuda')

        self.xscontrolnet = ControlNetModel.from_pretrained(
            ctlnetmodelname, torch_dtype=torch.float16,
        )
        self.xscontrolnet.to('cuda')
        self.xssdmodel = StableDiffusionControlNetPipeline.from_pretrained(
            sdmodelname, controlnet=self.xscontrolnet, torch_dtype=torch.float16,
            safety_checker=None)
        self.xssdmodel.to('cuda')

        # absolutely no self.inpainter.unet = torch.compile(self.inpainter.unet)
        # absolutely no self.inpainter.vae = torch.compile(self.inpainter.vae)
    
    @modal.web_endpoint(docs=True)
    def inference(self, image_data_uri, redaction_categories, prompt, size, steps, controlnet_conditioning=0, manual_seed=42069):
        with torch.no_grad():
            size = int(size)
            steps = int(steps)
            manual_seed = int(manual_seed)
            controlnet_conditioning = float(controlnet_conditioning)
            image = Image.open(BytesIO(DataURI(image_data_uri).data)).convert("RGB").resize((768,768))
            start_time = time.time()
            # purple_image = Image.fromarray(
            #     np.array(image) * np.array([[[1, 0, 1]]], dtype=np.uint8)
            # )
            seginputs = self.segprocessor(image, return_tensors="pt").to("cuda")
            outputs = self.segmodel(**seginputs)
            logits = outputs.logits[0]

            ban_cars_mask = [0] * len(self.segmodel.config.label2id)
            for category in redaction_categories.split(","):
                ban_cars_mask[self.segmodel.config.label2id[category]] = 1
            
            car_mask = Image.fromarray(
                torch.tensor(ban_cars_mask, dtype=torch.uint8)[ torch.argmax(logits, dim=0).cpu() ].numpy() * 255
            )
            # blur the car mask
            car_mask = car_mask.filter(ImageFilter.GaussianBlur(5))
            seg_time = time.time() - start_time
            if False and controlnet_conditioning == 0:
                inpainted_image = self.inpainter(
                    image=image,
                    prompt=prompt,
                    mask_image=car_mask,
                    generator=torch.manual_seed(manual_seed),
                    num_inference_steps=steps,
                    width=size,
                    height=size,
                    guidance_scale=12.0,
                    strength=1.0,
                ).images[0]
            else:
                image_canny = Image.fromarray(canny(np.array(image.convert("L").resize((size,size))), sigma=1.0).astype(np.uint8) * 255)
                image_canny_with_mask = Image.fromarray(255 - np.where(
                    np.array(car_mask.resize((size,size))) == [0],
                    image_canny,
                    np.array([255], dtype=np.uint8)))
                inpainted_image = self.xssdmodel(
                    prompt=prompt,
                    image=image_canny_with_mask,
                    num_inference_steps=1,
                    guidance_scale=0.0,
                    controlnet_conditioning_scale=controlnet_conditioning,
                    generator=torch.manual_seed(manual_seed),
                    height=size,
                    width=size,
                ).images[0]

            paint_time = time.time() - start_time - seg_time

            bs = BytesIO()
            inpainted_image.save(bs, format="jpeg", quality=50, subsampling=2)
            result_image_data_uri = DataURI.from_stream(bs, "x.jpeg")
            return {
                "image": result_image_data_uri,
                "redaction_percent": int(np.average(np.array(car_mask) / 255) * 100),
                "seg_time": seg_time,
                "paint_time": paint_time,
                "compiled": "50",
            }
