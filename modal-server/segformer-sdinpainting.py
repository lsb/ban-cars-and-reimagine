from io import BytesIO
from pathlib import Path
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from diffusers import AutoPipelineForInpainting
from PIL import Image
import torch
import numpy as np
from huggingface_hub import snapshot_download

segformer_model = "nvidia/segformer-b0-finetuned-cityscapes-768-768"
sdinpaint_model = "stable-diffusion-v1-5/stable-diffusion-inpainting"
banned_categories = ['car', 'road', 'sidewalk', 'traffic light', 'traffic sign']


import modal
from data_uri_parser import DataURI

container_image = modal.Image.debian_slim().pip_install("diffusers", "transformers", "torch==2.4.1", "accelerate", "data_uri_parser")

app = modal.App("segformer-sdinpainting", image=container_image)

@app.cls(gpu=modal.gpu.H100(), container_idle_timeout=60, image=container_image)
class Model:
    @modal.build()
    def build(self):
        snapshot_download(segformer_model)
        snapshot_download(sdinpaint_model)
    
    @modal.enter()
    def enter(self):
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
        self.inpainter =  AutoPipelineForInpainting.from_pretrained(
            sdinpaint_model,
            variant='fp16',
            torch_dtype=torch.float16,
            local_files_only=True,
        )
        self.inpainter.to('cuda')
    
    @modal.web_endpoint(docs=True)
    def inference(self, image_data_uri, redaction_categories, prompt, size, steps, manual_seed=42069):
        with torch.no_grad():
            size = int(size)
            steps = int(steps)
            manual_seed = int(manual_seed)
            image = Image.open(BytesIO(DataURI(image_data_uri).data)).convert("RGB")
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

            bs = BytesIO()
            inpainted_image.save(bs, format="jpeg", subsampling=2)
            result_image_data_uri = DataURI.from_stream(bs, "x.jpeg")
            return {
                "image": result_image_data_uri,
                "redaction_percent": int(np.average(np.array(car_mask) / 255) * 100),
            }
