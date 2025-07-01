from cog import BasePredictor, Input, Path
from diffusers import (
    StableDiffusionPipeline,
    DDIMScheduler,
    PNDMScheduler,
    LMSDiscreteScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
    KDPM2DiscreteScheduler,
    KDPM2AncestralDiscreteScheduler,
)
import torch
import glob, os
from huggingface_hub import snapshot_download

class Predictor(BasePredictor):
    def setup(self):
        self.pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16,
            use_auth_token=os.getenv("HF_TOKEN", None)
        ).to("cuda")

    def predict(
        self,
        prompt: str   = Input(description="What to generate"),
        negative_prompt: str = Input(description="What to avoid", default=""),
        scheduler: str = Input(
            description="Which scheduler to use",
            choices=[
                "DDIM",
                "PNDM",
                "LMS",
                "Euler",
                "Euler Ancestral",
                "DPMSolver Multistep",
                "K_DP_M2",
                "K_DP_M2 Ancestral"
            ],
            default="DDIM"
        ),
        steps: int   = Input(description="Inference steps", ge=1, le=200, default=50),
        width: int   = Input(description="Width", ge=128, le=2048, default=512),
        height: int  = Input(description="Height", ge=128, le=2048, default=512),
        lora_repo: str = Input(description="LoRA repo (user/model)", default=""),
    ) -> Path:
        # Map dropdown to scheduler class
        sched_map = {
            "DDIM": DDIMScheduler,
            "PNDM": PNDMScheduler,
            "LMS": LMSDiscreteScheduler,
            "Euler": EulerDiscreteScheduler,
            "Euler Ancestral": EulerAncestralDiscreteScheduler,
            "DPMSolver Multistep": DPMSolverMultistepScheduler,
            "K_DP_M2": KDPM2DiscreteScheduler,
            "K_DP_M2 Ancestral": KDPM2AncestralDiscreteScheduler,
        }
        # replace scheduler
        self.pipe.scheduler = sched_map[scheduler].from_config(self.pipe.scheduler.config)

        # load LoRA if provided
        if lora_repo:
            repo_dir = snapshot_download(lora_repo, use_auth_token=os.getenv("HF_TOKEN", None))
            weights = glob.glob(f"{repo_dir}/*.safetensors")[0]
            self.pipe.unet.load_attn_procs(weights)

        img = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=steps
        ).images[0]

        out = Path("output.png")
        img.save(out)
        return out
