"""Video -> cached player detections -> OC-SORT tracks and a silent preview.

Run from backend: python -m tracking.pipeline --video clip.mp4 --output tracking/runs/demo
Each output directory belongs to one run; do not write to it from concurrent jobs.
"""

import argparse
import json
from importlib.metadata import version
from itertools import zip_longest
from pathlib import Path
from time import perf_counter

import numpy as np

from tracking.artifacts import (
    atomic_path,
    complete_stage,
    fingerprint,
    sha256_file,
    stage_is_complete,
    tracks_writer,
    write_json,
)
from tracking.config import (
    DECODE_VERSION,
    DEFAULT_WEIGHTS,
    DETECT_VERSION,
    SCHEMA_VERSION,
    TRACK_VERSION,
    DecodeConfig,
    DetectorConfig,
    TrackerConfig,
)
from tracking.decode import VideoReader
from tracking.detect import PlayerDetector
from tracking.schemas import FrameTracks, VideoMetadata
from tracking.track import PlayerTracker
from tracking.viz import PreviewWriter

DETECT_FILES = ("detections.jsonl", "video.json")
TRACK_FILES = ("tracks.json", "annotated.mp4")


def _detect(video: Path, weights: Path, output: Path, detector_config, decode_config):
    detector = PlayerDetector(weights, detector_config)
    frame_count = box_count = 0
    with (
        VideoReader(video, decode_config) as reader,
        atomic_path(output / "detections.jsonl") as temporary,
        temporary.open("w", encoding="utf-8") as handle,
    ):
        for frame in reader:
            detections = detector.detect(frame.image)
            handle.write(
                json.dumps(
                    {
                        "frame_index": frame.index,
                        "timestamp_s": frame.timestamp_s,
                        "detections": detections.tolist(),
                    },
                    allow_nan=False,
                )
                + "\n"
            )
            frame_count += 1
            box_count += len(detections)
            if frame_count % 100 == 0:
                print(
                    f"Detected {frame_count} frames ({frame.timestamp_s:.1f}s of video)", flush=True
                )
        if not frame_count:
            raise ValueError("No frames decoded from this video")
        metadata = reader.metadata.model_dump()
    write_json(
        output / "video.json",
        {
            "video": metadata,
            "frame_count": frame_count,
            "detection_count": box_count,
            "player_class_id": detector.player_id,
        },
    )


def _track(video: Path, output: Path, config, decode_config, confidence: float, header: dict):
    tracker = PlayerTracker(config, confidence)
    metadata = VideoMetadata.model_validate(header["video"])
    with (
        VideoReader(video, decode_config) as reader,
        (output / "detections.jsonl").open(encoding="utf-8") as detections_file,
        tracks_writer(output / "tracks.json", header) as save_frame,
        atomic_path(output / "annotated.mp4") as preview_path,
        PreviewWriter(preview_path, metadata, decode_config.stride) as preview,
    ):
        for frame, line in zip_longest(reader, detections_file):
            if frame is None or line is None:
                raise ValueError("Cached detections do not match the decoded frame count")
            record = json.loads(line)
            if (
                record["frame_index"] != frame.index
                or abs(record["timestamp_s"] - frame.timestamp_s) > 1e-6
            ):
                raise ValueError("Cached detections do not match the video timestamps")
            detections = np.asarray(record["detections"], dtype=np.float32).reshape(-1, 6)
            tracked_frame = FrameTracks(
                frame_index=frame.index,
                timestamp_s=frame.timestamp_s,
                tracks=tracker.update(detections, frame.image),
            )
            save_frame(tracked_frame)
            preview.write(frame.image, tracked_frame)


def run_pipeline(
    video: Path,
    weights: Path,
    output: Path,
    detector_config: DetectorConfig | None = None,
    tracker_config: TrackerConfig | None = None,
    decode_config: DecodeConfig | None = None,
) -> dict:
    detector_config = detector_config or DetectorConfig()
    tracker_config = tracker_config or TrackerConfig()
    decode_config = decode_config or DecodeConfig()
    video, weights, output = Path(video).resolve(), Path(weights).resolve(), Path(output).resolve()
    for path in (video, weights):
        if not path.is_file():
            raise FileNotFoundError(path)
    artifacts = (
        *DETECT_FILES,
        *TRACK_FILES,
        "run.json",
        "detect.complete.json",
        "track.complete.json",
    )
    if any(path == output / name for path in (video, weights) for name in artifacts):
        raise ValueError("Use an output directory that cannot overwrite an input file")
    output.mkdir(parents=True, exist_ok=True)
    # Cache keys include source/weights contents, versions, parameters, and runtime packages.
    runtime = {
        name: version(name)
        for name in (
            "ultralytics",
            "torch",
            "torchvision",
            "numpy",
            "opencv-python",
            "av",
            "boxmot",
        )
    }
    provenance = {
        "video_sha256": sha256_file(video),
        "weights_sha256": sha256_file(weights),
        "decode_version": DECODE_VERSION,
        "detect_version": DETECT_VERSION,
        "decode": decode_config.model_dump(),
        "detector": detector_config.model_dump(),
        "detector_runtime": {name: value for name, value in runtime.items() if name != "boxmot"},
    }
    detect_key = fingerprint(provenance)
    detect_cached = stage_is_complete(output, "detect", detect_key, DETECT_FILES)
    started = perf_counter()
    if not detect_cached:
        print("Running player detection...", flush=True)
        _detect(video, weights, output, detector_config, decode_config)
        complete_stage(output, "detect", detect_key, DETECT_FILES)
    else:
        print("Reusing saved detections.", flush=True)
    detection_seconds = perf_counter() - started
    metadata = json.loads((output / "video.json").read_text(encoding="utf-8"))
    provenance.update(
        {
            "detections_sha256": sha256_file(output / "detections.jsonl"),
            "track_version": TRACK_VERSION,
            "tracker": tracker_config.model_dump(),
            "boxmot_version": runtime["boxmot"],
            "player_class_id": metadata["player_class_id"],
            "schema_version": SCHEMA_VERSION,
        }
    )
    track_key = fingerprint({"detect_key": detect_key, **provenance})
    track_cached = stage_is_complete(output, "track", track_key, TRACK_FILES)
    started = perf_counter()
    if not track_cached:
        print("Running OC-SORT and writing preview...", flush=True)
        _track(
            video,
            output,
            tracker_config,
            decode_config,
            detector_config.confidence,
            {
                "schema_version": SCHEMA_VERSION,
                "video": metadata["video"],
                "provenance": provenance,
            },
        )
        complete_stage(output, "track", track_key, TRACK_FILES)
    else:
        print("Reusing saved tracks and preview.", flush=True)
    result = {
        "output_directory": str(output),
        "tracks": str(output / "tracks.json"),
        "preview": str(output / "annotated.mp4"),
        "frames": metadata["frame_count"],
        "detections": metadata["detection_count"],
        "reused_detection": detect_cached,
        "reused_tracking": track_cached,
        "detection_seconds": detection_seconds,
        "tracking_and_preview_seconds": perf_counter() - started,
        "provenance": provenance,
    }
    write_json(output / "run.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--stride", type=int, choices=(1, 2), default=1)
    parser.add_argument("--max-seconds", type=float)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--max-age", type=int, default=30)
    parser.add_argument("--min-hits", type=int, default=3)
    parser.add_argument("--iou-threshold", type=float, default=0.3)
    args = parser.parse_args()
    result = run_pipeline(
        args.video,
        args.weights,
        args.output,
        DetectorConfig(imgsz=args.imgsz, confidence=args.confidence, device=args.device),
        TrackerConfig(
            max_age=args.max_age, min_hits=args.min_hits, iou_threshold=args.iou_threshold
        ),
        DecodeConfig(stride=args.stride, max_seconds=args.max_seconds, max_frames=args.max_frames),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
