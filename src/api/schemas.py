from pydantic import BaseModel, Field
from typing import Dict, List


class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Text description of desired motion")
    character: str = Field("Pocoyo")
    num_steps: int = Field(50, ge=5, le=200)
    guidance_scale: float = Field(7.5, ge=1.0, le=20.0)


class GenerateResponse(BaseModel):
    motion: List[List[List[float]]]
    metadata: Dict
    rig_controllers: Dict[str, List[float]]


class CharacterInfo(BaseModel):
    id: int
    name: str
    num_joints: int
    description: str


class HealthResponse(BaseModel):
    status: str
    model: str
    device: str