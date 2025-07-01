from cog import BasePredictor, Input, Path
from diffusers import StableDiffusionPipeline, DDIMScheduler, PNDMScheduler
import torch

class Predictor(BasePredictor):
    def setup(self):
        # Load model once
        self.pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16
        ).to("cuda")

    def predict(
        self,
        prompt: str   = Input(description="What to generate"),
        negative_prompt: str = Input(description="What to avoid", default=""),
        sampler: str = Input(description="Which sampler", choices=["DDIM","PNDM"], default="DDIM"),
        steps: int   = Input(description="How many steps", ge=1, le=100, default=50),
        width: int   = Input(description="Image width", ge=128, le=2048, default=512),
        height: int  = Input(description="Image height", ge=128, le=2048, default=512),
        lora_repo: str = Input(description="LoRA repo (username/model)", default=""),
    ) -> Path:
        # Swap in the chosen scheduler
        if sampler == "PNDM":
            self.pipe.scheduler = PNDMScheduler.from_config(self.pipe.scheduler.config)
        else:
            self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)

        # Load LoRA if given
        if lora_repo:
            from huggingface_hub import snapshot_download
            import glob, os
            token = os.getenv("HF_TOKEN")
            repo_dir = snapshot_download(lora_repo, use_auth_token=token)
            weights = glob.glob(f"{repo_dir}/*.safetensors")[0]
            self.pipe.unet.load_attn_procs(weights)

        # Generate
        image = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width, height=height,
            num_inference_steps=steps
        ).images[0]

        out = Path("output.png")
        image.save(out)
        return out
