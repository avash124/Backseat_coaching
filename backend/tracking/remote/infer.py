"""Upload local inputs and run the player pipeline on a Modal GPU."""

import json
import re
from pathlib import Path
from uuid import uuid4

from tracking.artifacts import sha256_file
from tracking.config import DEFAULT_WEIGHTS, DecodeConfig, DetectorConfig, TrackerConfig
from tracking.remote.app import app, image, outputs_volume, videos_volume, weights_volume


def validate_run_name(run_name: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", run_name):
        raise ValueError("Run name must be 1-80 letters, digits, hyphens, or underscores")
    return run_name


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
        )
    finally:
        outputs_volume.commit()


@app.local_entrypoint()
def main(
    video: str,
    weights: str = str(DEFAULT_WEIGHTS),
    run_name: str = "",
    max_seconds: float = 30.0,
    imgsz: int = 1280,
    confidence: float = 0.25,
    stride: int = 1,
    max_age: int = 30,
    min_hits: int = 3,
    iou_threshold: float = 0.3,
):
    """Use --max-seconds 0 for the whole video. Use a unique run name per concurrent job."""
    video_path, weights_path = Path(video).resolve(), Path(weights).resolve()
    for path in (video_path, weights_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    run_name = validate_run_name(run_name or f"players-{uuid4().hex[:12]}")
    detector = DetectorConfig(device="0", imgsz=imgsz, confidence=confidence)
    tracker = TrackerConfig(max_age=max_age, min_hits=min_hits, iou_threshold=iou_threshold)
    decode = DecodeConfig(stride=stride, max_seconds=None if max_seconds == 0 else max_seconds)
    video_name = sha256_file(video_path) + video_path.suffix.lower()
    weights_name = sha256_file(weights_path) + ".pt"
    print("Uploading video and weights...", flush=True)
    with videos_volume.batch_upload(force=True) as upload:
        upload.put_file(video_path, "/" + video_name)
    with weights_volume.batch_upload(force=True) as upload:
        upload.put_file(weights_path, "/" + weights_name)
    print(f"Run: {run_name}. Outputs will be in nextup-inference:/{run_name}", flush=True)
    result = infer.remote(
        video_name,
        weights_name,
        run_name,
        detector.model_dump(),
        tracker.model_dump(),
        decode.model_dump(),
    )
    print(json.dumps(result, indent=2))
    print(
        f"Download: modal volume get nextup-inference /{run_name} tracking/runs/{run_name} --env main"
    )
