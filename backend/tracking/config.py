"""Inference settings. Defaults are starting points to evaluate on our own footage.

No threshold here has been calibrated against labeled holdout footage. Every stage
has its own version string; bump it when that stage's algorithm or output changes
so only that stage and the stages after it are recomputed.
"""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

DECODE_VERSION = "pyav-v1"
# v3: detect players and referees, one row per box with per-class scores.
DETECT_VERSION = "people-v3"
# v3: class-agnostic tracking of all people, tentative observations recorded,
# selectable BoxMOT tracker, tracker reset at shot cuts, tracks_raw.json output.
TRACK_VERSION = "tracker-v3"
# Role vote, backfill, stitching, interpolation, short-track removal, preview.
POSTPROCESS_VERSION = "postprocess-v2"
SCHEMA_VERSION = "1.1"
WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"
DEFAULT_WEIGHTS = WEIGHTS_DIR / "players-v1.pt"
# ReID checkpoints are looked up here by their BoxMOT catalog filename; the
# filename selects the architecture, and the file's SHA-256 enters the cache key.
REID_WEIGHTS_DIR = WEIGHTS_DIR / "reid"
DEFAULT_REID_WEIGHTS = "osnet_x0_25_msmt17.pt"

TrackerKind = Literal["ocsort", "deepocsort", "botsort", "strongsort", "hybridsort", "boosttrack"]
Role = Literal["player", "referee", "unknown"]


class DetectorConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
    imgsz: int = Field(default=1280, ge=32, multiple_of=32)
    # Keep weak observations available for association; they cannot start tracks.
    confidence: float = Field(default=0.1, gt=0, lt=1)
    device: str = "cpu"
    max_detections: int = Field(default=100, ge=1)
    # Frames per YOLO forward pass. Boxes match single-frame mode to ~1e-4 px but
    # are not bit-identical, so this is part of the detection cache key.
    batch_size: int = Field(default=8, ge=1, le=256)


class TrackerConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
    # Any BoxMOT 25.0.0 box tracker below. ocsort is motion-only; the others can
    # use ReID appearance embeddings (strongsort always does).
    kind: TrackerKind = "ocsort"
    # Before association, boxes overlapping a stronger box by at least this IoU are
    # treated as the same person and their class scores merged (see
    # detect.merge_duplicates). 0.7 matches Ultralytics' default NMS IoU; values
    # above 1 disable merging.
    duplicate_iou: float = Field(default=0.7, gt=0, le=1.01)
    # Counted in processed frames, so its duration depends on FPS and stride.
    max_age: int = Field(default=30, ge=1)
    min_hits: int = Field(default=3, ge=1)
    iou_threshold: float = Field(default=0.3, gt=0, le=1)
    track_threshold: float = Field(default=0.5, gt=0, lt=1)
    use_byte: bool = True
    # Use ReID embeddings where the tracker supports toggling them.
    appearance: bool = True
    reid_weights: str = Field(default=DEFAULT_REID_WEIGHTS, pattern=r"^[A-Za-z0-9_.-]+\.pt$")


class ShotConfig(BaseModel):
    """Hard-cut detection from HSV-histogram changes between consecutive frames."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
    enabled: bool = True
    # Bhattacharyya distance in [0, 1]. Starting point: fixed-camera frames of the
    # demo clip stay far below it; a hard cut to another view usually exceeds it.
    threshold: float = Field(default=0.35, gt=0, le=1)
    # Cuts closer than this to the previous cut (or the start) are ignored.
    min_shot_seconds: float = Field(default=1.0, ge=0)
    # Width of the downscaled frame used for the histogram.
    width: int = Field(default=160, ge=16, le=1920)


class PostprocessConfig(BaseModel):
    """Offline track cleanup. All values are uncalibrated starting points."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
    # (b) Restore observations hidden while a track waited for min_hits.
    backfill: bool = True
    # Role vote: referee share of the summed per-class scores. A track is a referee
    # above boundary + margin, a player below boundary - margin, otherwise unknown.
    # 0.5 means "more referee than player evidence". The NBA-trained referee class
    # is under-confident on HS officials, so calibrate the boundary on labeled
    # tracks before lowering it; do not tune it to one clip.
    role_boundary: float = Field(default=0.5, gt=0, lt=1)
    role_margin: float = Field(default=0.1, ge=0, lt=0.5)
    # (c) Link a track's end to another's start across a gap no longer than this.
    # Off by default (0) until calibrated on labeled tracks: reviewed links on the
    # demo clip were mostly wrong at 1-2 s gaps, and 2 of 3 at <= 1 s still ran into
    # multi-person boxes. A false merge corrupts identity; a fragment only splits it.
    # Try 1.0 when measuring against ground truth.
    stitch_max_gap_s: float = Field(default=0.0, ge=0)
    # Distance gate in box heights: base + per_second * gap. Prediction error of a
    # constant-velocity model grows with the gap; a player covers ~3 heights/s.
    stitch_gate_base: float = Field(default=0.5, ge=0)
    stitch_gate_per_s: float = Field(default=1.0, ge=0)
    # Seconds of the ending track used to estimate its final velocity.
    stitch_velocity_window_s: float = Field(default=0.5, gt=0)
    # Box heights at the two ends may differ by at most this factor.
    stitch_max_height_ratio: float = Field(default=1.5, ge=1)
    # Boxes wider than tall at either end usually enclose several people in a
    # crowd; their foot point and colors are unreliable, so they are not linked.
    stitch_max_aspect: float = Field(default=1.0, gt=0)
    # Torso-color (jersey) gate: Bhattacharyya distance between the mean HSV
    # histogram of the last boxes of one track and the first boxes of the other.
    # 0.45 is near the 90th percentile of same-track distances 1 s apart on the
    # demo clip; it mainly rejects links across teams. 1 disables the gate.
    stitch_max_appearance: float = Field(default=0.45, gt=0, le=1)
    # (a) Fill gaps up to this long inside a track with observed=false boxes.
    interpolate_max_gap_s: float = Field(default=1.0, ge=0)
    # (d) Remove tracks spanning less than this after stitching.
    min_track_s: float = Field(default=0.5, ge=0)


class DecodeConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
    stride: Literal[1, 2] = 1
    max_seconds: float | None = Field(default=None, gt=0)
    max_frames: int | None = Field(default=None, ge=1)
