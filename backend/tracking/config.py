"""Inference settings. Defaults are starting points to evaluate on our own footage."""
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

DECODE_VERSION = "pyav-v1"
DETECT_VERSION = "players-v1"
TRACK_VERSION = "ocsort-v1"
SCHEMA_VERSION = "1.0"
DEFAULT_WEIGHTS = Path(__file__).resolve().parent / "weights" / "players-v1.pt"


class DetectorConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
    imgsz: int = Field(default=1280, ge=32, multiple_of=32)
    confidence: float = Field(default=0.25, gt=0, lt=1)
    device: str = "cpu"
    max_detections: int = Field(default=100, ge=1)


class TrackerConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
    max_age: int = Field(default=30, ge=1)
    min_hits: int = Field(default=3, ge=1)
    iou_threshold: float = Field(default=0.3, gt=0, le=1)


class DecodeConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
    stride: Literal[1, 2] = 1
    max_seconds: float | None = Field(default=None, gt=0)
    max_frames: int | None = Field(default=None, ge=1)
