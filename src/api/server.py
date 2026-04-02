"""
FastAPI Server
Run: uvicorn src.api.server:app --host 0.0.0.0 --port 8000
"""

import torch
from fastapi import FastAPI
from typing import List

from ..models.toonmotion import ToonMotionDiffusion, ToonMotionConfig
from ..inference.generate import generate_motion
from ..inference.rig_export import motion_to_rig_controllers
from .schemas import GenerateRequest, GenerateResponse, CharacterInfo, HealthResponse

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def create_app(checkpoint_path=None):
    app = FastAPI(title="ToonMotion-Diffusion API", version="1.0.0")

    if checkpoint_path:
        from ..training.trainer import Trainer
        model, config = Trainer.load_checkpoint(checkpoint_path, DEVICE)
    else:
        config = ToonMotionConfig()
        model = ToonMotionDiffusion(config, DEVICE).to(DEVICE)
    model.eval()

    @app.post("/generate", response_model=GenerateResponse)
    async def generate(req: GenerateRequest):
        result = generate_motion(model, req.prompt, req.character, req.num_steps, req.guidance_scale, DEVICE)
        rig = motion_to_rig_controllers(result["motion"], req.character)
        return GenerateResponse(motion=result["motion"].tolist(), metadata=result["metadata"], rig_controllers=rig)

    @app.get("/characters", response_model=List[CharacterInfo])
    async def characters():
        return [
            CharacterInfo(id=0, name="Pocoyo", num_joints=18, description="Big head, snappy movement"),
            CharacterInfo(id=1, name="Elly", num_joints=18, description="Elephant proportions"),
            CharacterInfo(id=2, name="Pato", num_joints=18, description="Duck body, comic movement"),
            CharacterInfo(id=3, name="Maya", num_joints=18, description="Insect, wing-like arms"),
        ]

    @app.get("/health", response_model=HealthResponse)
    async def health():
        return HealthResponse(status="ok", model="ToonMotion-Diffusion v1.0", device=DEVICE)

    return app


app = create_app()