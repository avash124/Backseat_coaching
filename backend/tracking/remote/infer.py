"""Upload local inputs and run the person pipeline on a Modal GPU.

Flags mirror ``python -m tracking.pipeline``. Appearance trackers (every
``--tracker`` except ocsort, unless ``--no-appearance``) also upload the local
ReID checkpoint from tracking/weights/reid to the weights volume.
"""

import json
import re
from pathlib import Path, PurePosixPath
from uuid import uuid4

from tracking.artifacts import sha256_file
from tracking.config import (
    DEFAULT_WEIGHTS,
    DecodeConfig,
    DetectorConfig,
    PostprocessConfig,
    ShotConfig,
    TrackerConfig,
)
from tracking.court import CourtConfig
from tracking.remote.app import app, image, outputs_volume, videos_volume, weights_volume


def validate_run_name(run_name: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", run_name):
        raise ValueError("Run name must be 1-80 letters, digits, hyphens, or underscores")
    return run_name


def validate_reid_name(reid_name: str) -> str:
    """Accept only reid/<sha256>/<checkpoint>.pt inside the weights volume."""
    parts = PurePosixPath(reid_name).parts
    if (
        len(parts) != 3
        or parts[0] != "reid"
        or not re.fullmatch(r"[0-9a-f]{64}", parts[1])
        or not re.fullmatch(r"[A-Za-z0-9_.-]+\.pt", parts[2])
    ):
        raise ValueError("ReID weights must be stored as reid/<sha256>/<name>.pt")
    return reid_name


@app.function(
    image=image,
    gpu="L4",
    timeout=4 * 60 * 60,
    volumes={"/weights": weights_volume, "/videos": videos_volume, "/outputs": outputs_volume},
)
def infer(
    video_name: str,
    weights_name: str,
    run_name: str,
    detector: dict,
    tracker: dict,
    decode: dict,
    court: dict | None = None,
    shots: dict | None = None,
    postprocess: dict | None = None,
    reid_name: str | None = None,
    preview: bool = True,
) -> dict:
    from tracking.pipeline import run_pipeline

    validate_run_name(run_name)
    for name in (video_name, weights_name):
        if not name or Path(name).name != name:
            raise ValueError("Input names must refer to files directly in their volumes")
    weights_volume.reload()
    videos_volume.reload()
    outputs_volume.reload()
    try:
        return run_pipeline(
            Path("/videos") / video_name,
            Path("/weights") / weights_name,
            Path("/outputs") / run_name,
            DetectorConfig.model_validate(detector),
            TrackerConfig.model_validate(tracker),
            DecodeConfig.model_validate(decode),
            CourtConfig.model_validate(court) if court is not None else None,
            shot_config=ShotConfig.model_validate(shots or {}),
            postprocess_config=PostprocessConfig.model_validate(postprocess or {}),
            reid_weights=Path("/weights") / validate_reid_name(reid_name) if reid_name else None,
            preview=preview,
        )
    finally:
        outputs_volume.commit()


@app.local_entrypoint()
def main(
    video: str,
    weights: str = str(DEFAULT_WEIGHTS),
    run_name: str = "",
    max_seconds: float = 30.0,
    imgsz: int = DetectorConfig().imgsz,
    confidence: float = DetectorConfig().confidence,
    batch_size: int = DetectorConfig().batch_size,
    stride: int = 1,
    tracker: str = TrackerConfig().kind,
    duplicate_iou: float = TrackerConfig().duplicate_iou,
    max_age: int = TrackerConfig().max_age,
    min_hits: int = TrackerConfig().min_hits,
    iou_threshold: float = TrackerConfig().iou_threshold,
    track_threshold: float = TrackerConfig().track_threshold,
    use_byte: bool = True,
    appearance: bool = TrackerConfig().appearance,
    reid_weights: str = TrackerConfig().reid_weights,
    shots: bool = ShotConfig().enabled,
    shot_threshold: float = ShotConfig().threshold,
    min_shot_seconds: float = ShotConfig().min_shot_seconds,
    backfill: bool = PostprocessConfig().backfill,
    role_boundary: float = PostprocessConfig().role_boundary,
    role_margin: float = PostprocessConfig().role_margin,
    stitch_max_gap: float = PostprocessConfig().stitch_max_gap_s,
    interpolate_max_gap: float = PostprocessConfig().interpolate_max_gap_s,
    min_track_seconds: float = PostprocessConfig().min_track_s,
    preview: bool = True,
    court: str = "",
):
    """Use --max-seconds 0 for the whole video. Use a unique run name per concurrent job."""
    from tracking.track import ensure_reid_weights, reid_weights_path, uses_reid

    video_path, weights_path = Path(video).resolve(), Path(weights).resolve()
    for path in (video_path, weights_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    run_name = validate_run_name(run_name or f"players-{uuid4().hex[:12]}")
    detector = DetectorConfig(
        device="0", imgsz=imgsz, confidence=confidence, batch_size=batch_size
    )
    tracker_config = TrackerConfig(
        kind=tracker,
        duplicate_iou=duplicate_iou,
        max_age=max_age,
        min_hits=min_hits,
        iou_threshold=iou_threshold,
        track_threshold=track_threshold,
        use_byte=use_byte,
        appearance=appearance,
        reid_weights=reid_weights,
    )
    shot_config = ShotConfig(
        enabled=shots, threshold=shot_threshold, min_shot_seconds=min_shot_seconds
    )
    postprocess = PostprocessConfig(
        backfill=backfill,
        role_boundary=role_boundary,
        role_margin=role_margin,
        stitch_max_gap_s=stitch_max_gap,
        interpolate_max_gap_s=interpolate_max_gap,
        min_track_s=min_track_seconds,
    )
    if detector.confidence > tracker_config.track_threshold:
        raise ValueError("Detection confidence must be <= track_threshold")
    court_config = CourtConfig.from_file(Path(court)) if court else None
    decode = DecodeConfig(stride=stride, max_seconds=None if max_seconds == 0 else max_seconds)
    video_name = sha256_file(video_path) + video_path.suffix.lower()
    weights_name = sha256_file(weights_path) + ".pt"
    reid_path = reid_name = None
    if uses_reid(tracker_config):
        reid_path = ensure_reid_weights(reid_weights_path(tracker_config))
        # The file name selects the ReID architecture, so keep it under a hash folder.
        reid_name = f"reid/{sha256_file(reid_path)}/{reid_path.name}"
    print("Uploading video and weights...", flush=True)
    with videos_volume.batch_upload(force=True) as upload:
        upload.put_file(video_path, "/" + video_name)
    with weights_volume.batch_upload(force=True) as upload:
        upload.put_file(weights_path, "/" + weights_name)
        if reid_path is not None:
            upload.put_file(reid_path, "/" + reid_name)
    print(f"Run: {run_name}. Outputs will be in nextup-inference:/{run_name}", flush=True)
    result = infer.remote(
        video_name,
        weights_name,
        run_name,
        detector.model_dump(),
        tracker_config.model_dump(),
        decode.model_dump(),
        court_config.model_dump() if court_config else None,
        shot_config.model_dump(),
        postprocess.model_dump(),
        reid_name,
        preview,
    )
    print(json.dumps(result, indent=2))
    print(
        f"Download: modal volume get nextup-inference /{run_name} tracking/runs/{run_name} --env main"
    )
