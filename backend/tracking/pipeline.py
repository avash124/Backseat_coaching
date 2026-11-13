"""Video -> cached person detections -> raw tracks -> cleaned tracks and a silent preview.

Run from backend: python -m tracking.pipeline --video clip.mp4 --output tracking/runs/demo

Stages, each cached separately by a key over its inputs, settings, and version:
  detect       YOLO players + referees              detections.jsonl, video.json
  track        duplicate merge, court filter,       tracks_raw.json, track_stats.json,
               BoxMOT tracker reset at shot cuts    track_appearance.json
  postprocess  roles, backfill, stitching,          tracks.json, postprocess.json,
               interpolation, short-track removal   tracking_summary.json, annotated.mp4
Changing postprocess settings reuses detections and raw tracks; changing tracker,
court, or shot settings reuses detections. tracks.json keeps referee and
unknown-role tracks; filter boxes on role == "player" for players.
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
    POSTPROCESS_VERSION,
    SCHEMA_VERSION,
    TRACK_VERSION,
    DecodeConfig,
    DetectorConfig,
    PostprocessConfig,
    ShotConfig,
    TrackerConfig,
)
from tracking.court import CourtConfig, CourtFilter
from tracking.decode import VideoReader
from tracking.detect import PersonDetector, merge_duplicates
from tracking.postprocess import postprocess_tracks, proxy_metrics
from tracking.schemas import (
    RawFrameTracks,
    RawObservation,
    ShotCut,
    TracksArtifact,
    VideoMetadata,
)
from tracking.shots import ShotDetector
from tracking.track import (
    TRACKER_KINDS,
    AppearanceSummary,
    PlayerTracker,
    ensure_reid_weights,
    load_reid_model,
    reid_weights_path,
    uses_reid,
)
from tracking.viz import PreviewWriter

DETECT_FILES = ("detections.jsonl", "video.json")
TRACK_FILES = ("tracks_raw.json", "track_stats.json", "track_appearance.json")
POSTPROCESS_FILES = ("tracks.json", "postprocess.json", "tracking_summary.json")
PREVIEW_FILE = "annotated.mp4"


def _split_scores(rows: np.ndarray, class_ids: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Return (N, 6) tracker rows and (N, K) class scores; one-hot if scores are absent."""
    rows = np.asarray(rows, np.float32)
    if rows.ndim == 2 and rows.shape[1] == 6 + len(class_ids):
        return rows[:, :6], rows[:, 6:]
    if rows.ndim != 2 or rows.shape[1] != 6:
        raise ValueError("Detector rows must have 6 columns plus one score per class")
    scores = np.zeros((len(rows), len(class_ids)), np.float32)
    for column, class_id in enumerate(class_ids):
        scores[rows[:, 5] == class_id, column] = rows[rows[:, 5] == class_id, 4]
    return rows, scores


def _detect(video: Path, weights: Path, output: Path, detector_config, decode_config):
    started = perf_counter()
    detector = PersonDetector(weights, detector_config)
    class_ids = list(detector.class_ids)
    frame_count = box_count = 0
    with (
        VideoReader(video, decode_config) as reader,
        atomic_path(output / "detections.jsonl") as temporary,
        temporary.open("w", encoding="utf-8") as handle,
    ):
        batch = []

        def flush():
            nonlocal frame_count, box_count
            for frame, rows in zip(batch, detector.detect_batch([f.image for f in batch])):
                boxes, scores = _split_scores(rows, class_ids)
                handle.write(
                    json.dumps(
                        {
                            "frame_index": frame.index,
                            "timestamp_s": frame.timestamp_s,
                            "detections": boxes.tolist(),
                            "class_scores": scores.tolist(),
                        },
                        allow_nan=False,
                    )
                    + "\n"
                )
                frame_count += 1
                box_count += len(boxes)
                if frame_count % 100 == 0:
                    print(f"Detected {frame_count} frames ({frame.timestamp_s:.1f}s)", flush=True)
            batch.clear()

        for frame in reader:
            batch.append(frame)
            if len(batch) >= detector_config.batch_size:
                flush()
        flush()
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
            # Order of class_scores columns in detections.jsonl and tracks_raw.json.
            "class_names": {str(i): detector.class_names[i] for i in class_ids},
            "detection_seconds": round(perf_counter() - started, 3),
        },
    )


def _track(
    video: Path,
    output: Path,
    config: TrackerConfig,
    shot_config: ShotConfig,
    decode_config: DecodeConfig,
    detector_config: DetectorConfig,
    header: dict,
    court_config: CourtConfig | None,
    reid_weights: Path | None,
):
    started = perf_counter()
    reid_model = load_reid_model(reid_weights, detector_config.device) if reid_weights else None
    tracker = PlayerTracker(config, detector_config.confidence, reid_model)
    shots = ShotDetector(shot_config)
    appearance = AppearanceSummary()
    metadata = VideoMetadata.model_validate(header["video"])
    court = CourtFilter(court_config, metadata.width, metadata.height) if court_config else None
    cuts: list[ShotCut] = []
    class_ids = [int(key) for key in header["class_names"]]
    counts = dict.fromkeys(
        ("frames", "candidates", "duplicates", "outside", "confirmed", "tentative"), 0
    )
    with (
        VideoReader(video, decode_config) as reader,
        (output / "detections.jsonl").open(encoding="utf-8") as detections_file,
        tracks_writer(output / "tracks_raw.json", header) as save_frame,
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
            boxes = np.asarray(record["detections"], dtype=np.float32).reshape(-1, 6)
            scores = np.asarray(record["class_scores"], dtype=np.float32).reshape(
                len(boxes), len(class_ids)
            )
            counts["candidates"] += len(boxes)
            merged_boxes, scores = merge_duplicates(boxes, scores, config.duplicate_iou, class_ids)
            counts["duplicates"] += len(boxes) - len(merged_boxes)
            boxes = merged_boxes
            if court is not None:
                inside = court.contains(boxes)
                counts["outside"] += int((~inside).sum())
                boxes, scores = boxes[inside], scores[inside]
            cut = shots.update(frame.index, frame.timestamp_s, frame.image)
            if cut is not None:
                cuts.append(cut)
                tracker.reset()
            confirmed = tracker.update(boxes, frame.image)
            observations = []
            for item in sorted(confirmed + tracker.tentative, key=lambda x: x.track_id):
                if not 0 <= item.detection_index < len(scores):
                    raise ValueError(f"{config.kind} returned a track without its detection")
                appearance.add(frame.image, item.track_id, item.bbox_xyxy)
                observations.append(
                    RawObservation(
                        track_id=item.track_id,
                        bbox_xyxy=item.bbox_xyxy,
                        confidence=item.confidence,
                        class_id=item.class_id,
                        class_scores=tuple(float(v) for v in scores[item.detection_index]),
                        confirmed=item.confirmed,
                    )
                )
            save_frame(
                RawFrameTracks(
                    frame_index=frame.index, timestamp_s=frame.timestamp_s, tracks=observations
                )
            )
            counts["frames"] += 1
            counts["confirmed"] += len(confirmed)
            counts["tentative"] += len(tracker.tentative)
    write_json(output / "track_appearance.json", appearance.to_json())
    write_json(
        output / "track_stats.json",
        {
            "frames": counts["frames"],
            "candidate_detections": counts["candidates"],
            # Rows folded into an overlapping stronger row (same person).
            "merged_duplicate_detections": counts["duplicates"],
            "outside_court_detections": counts["outside"],
            "court_filter_enabled": court is not None,
            "confirmed_observations": counts["confirmed"],
            "tentative_observations": counts["tentative"],
            "tracker": config.kind,
            "reid": reid_model is not None,
            "shot_cuts": [cut.model_dump() for cut in cuts],
            "tracking_seconds": round(perf_counter() - started, 3),
        },
    )


def _postprocess(
    video: Path,
    output: Path,
    config: PostprocessConfig,
    decode_config: DecodeConfig,
    header: dict,
    court_config: CourtConfig | None,
    preview: bool,
):
    started = perf_counter()
    raw = json.loads((output / "tracks_raw.json").read_text(encoding="utf-8"))
    stats = json.loads((output / "track_stats.json").read_text(encoding="utf-8"))
    video_info = json.loads((output / "video.json").read_text(encoding="utf-8"))
    class_names = {int(key): name for key, name in raw["class_names"].items()}
    cuts = [ShotCut.model_validate(cut) for cut in stats["shot_cuts"]]
    appearance = json.loads((output / "track_appearance.json").read_text(encoding="utf-8"))
    result = postprocess_tracks(
        raw["frames"],
        list(class_names.values()),
        [cut.frame_index for cut in cuts],
        config,
        appearance["tracks"],
    )
    artifact = TracksArtifact(
        schema_version=SCHEMA_VERSION,
        video=VideoMetadata.model_validate(raw["video"]),
        provenance=header["provenance"],
        class_names=class_names,
        shot_cuts=cuts,
        track_info=result.track_info,
        frames=result.frames,
    )
    top_level = artifact.model_dump(mode="json", exclude={"frames"})
    with tracks_writer(output / "tracks.json", top_level) as save_frame:
        for frame in artifact.frames:
            save_frame(frame)
    write_json(
        output / "postprocess.json",
        {
            "config": config.model_dump(),
            "stats": result.stats,
            "stitches": [record.model_dump() for record in result.stitches],
            "raw_track_roles": {str(key): role for key, role in result.raw_roles.items()},
        },
    )
    postprocess_seconds = perf_counter() - started
    started = perf_counter()
    if not preview:
        # Do not leave a preview from earlier settings next to the new tracks.
        (output / PREVIEW_FILE).unlink(missing_ok=True)
    else:
        court = (
            CourtFilter(court_config, artifact.video.width, artifact.video.height)
            if court_config
            else None
        )
        cut_at = {cut.frame_index: cut for cut in cuts}
        with (
            VideoReader(video, decode_config) as reader,
            atomic_path(output / PREVIEW_FILE) as preview_path,
            PreviewWriter(preview_path, artifact.video, decode_config.stride, court) as writer,
        ):
            for frame, tracked in zip(reader, artifact.frames, strict=True):
                writer.write(frame.image, tracked, cut_at.get(frame.index))
    frames = [frame.model_dump() for frame in artifact.frames]
    raw_frames = [
        {
            "timestamp_s": frame["timestamp_s"],
            "tracks": [
                {**box, "role": result.raw_roles.get(box["track_id"], "unknown")}
                for box in frame["tracks"]
                if box["confirmed"]
            ],
        }
        for frame in raw["frames"]
    ]
    final = proxy_metrics(frames)
    write_json(
        output / "tracking_summary.json",
        {
            "frames": len(frames),
            "candidate_detections": stats["candidate_detections"],
            "merged_duplicate_detections": stats["merged_duplicate_detections"],
            "outside_court_detections": stats["outside_court_detections"],
            "court_filter_enabled": stats["court_filter_enabled"],
            "tracker": stats["tracker"],
            "shot_cuts": len(cuts),
            "unique_track_ids": len(artifact.track_info),
            "unique_player_ids": final["unique_player_ids"],
            "stitches": len(result.stitches),
            "proxy_metrics": final,
            # Same metrics on the confirmed tracker output, before postprocessing.
            "proxy_metrics_raw": proxy_metrics(raw_frames),
            "postprocess": result.stats,
            "stage_seconds": {
                "detect": video_info.get("detection_seconds"),
                "track": stats["tracking_seconds"],
                "postprocess": round(postprocess_seconds, 3),
                "preview": round(perf_counter() - started, 3) if preview else None,
            },
            "note": "Label-free proxies, not ID-switch counts or accuracy metrics.",
        },
    )


def run_pipeline(
    video: Path,
    weights: Path,
    output: Path,
    detector_config: DetectorConfig | None = None,
    tracker_config: TrackerConfig | None = None,
    decode_config: DecodeConfig | None = None,
    court_config: CourtConfig | None = None,
    *,
    shot_config: ShotConfig | None = None,
    postprocess_config: PostprocessConfig | None = None,
    reid_weights: Path | None = None,
    preview: bool = True,
) -> dict:
    detector_config = detector_config or DetectorConfig()
    tracker_config = tracker_config or TrackerConfig()
    decode_config = decode_config or DecodeConfig()
    shot_config = shot_config or ShotConfig()
    postprocess_config = postprocess_config or PostprocessConfig()
    if detector_config.confidence > tracker_config.track_threshold:
        raise ValueError("Detection confidence must be <= track_threshold")
    if court_config is None:
        print("No court boundary supplied; off-court people can still be tracked.", flush=True)
    video, weights, output = Path(video).resolve(), Path(weights).resolve(), Path(output).resolve()
    for path in (video, weights):
        if not path.is_file():
            raise FileNotFoundError(path)
    postprocess_files = POSTPROCESS_FILES + ((PREVIEW_FILE,) if preview else ())
    artifacts = (
        *DETECT_FILES,
        *TRACK_FILES,
        *POSTPROCESS_FILES,
        PREVIEW_FILE,
        "run.json",
        "detect.complete.json",
        "track.complete.json",
        "postprocess.complete.json",
    )
    if any(path == output / name for path in (video, weights) for name in artifacts):
        raise ValueError("Use an output directory that cannot overwrite an input file")
    reid_sha256 = None
    if uses_reid(tracker_config):
        reid_weights = ensure_reid_weights(
            Path(reid_weights) if reid_weights else reid_weights_path(tracker_config)
        ).resolve()
        if reid_weights.name != tracker_config.reid_weights:
            raise ValueError("The ReID file name must match TrackerConfig.reid_weights")
        reid_sha256 = sha256_file(reid_weights)
    else:
        reid_weights = None
    output.mkdir(parents=True, exist_ok=True)
    # Cache keys include source/weights contents, versions, parameters, and runtime packages.
    runtime = {
        name: version(name)
        for name in ("ultralytics", "torch", "torchvision", "numpy", "opencv-python", "av")
    }
    provenance = {
        "video_sha256": sha256_file(video),
        "weights_sha256": sha256_file(weights),
        "decode_version": DECODE_VERSION,
        "detect_version": DETECT_VERSION,
        "decode": decode_config.model_dump(),
        "detector": detector_config.model_dump(),
        "detector_runtime": runtime,
    }
    detect_key = fingerprint(provenance)
    detect_cached = stage_is_complete(output, "detect", detect_key, DETECT_FILES)
    started = perf_counter()
    if not detect_cached:
        print("Running person detection...", flush=True)
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
            "shots": shot_config.model_dump(),
            "court": court_config.model_dump() if court_config else None,
            "boxmot_version": version("boxmot"),
            "reid_weights_sha256": reid_sha256,
            "reid_device": detector_config.device if reid_sha256 else None,
            "player_class_id": metadata["player_class_id"],
            "class_names": metadata["class_names"],
            "schema_version": SCHEMA_VERSION,
        }
    )
    track_key = fingerprint({"detect_key": detect_key, **provenance})
    track_cached = stage_is_complete(output, "track", track_key, TRACK_FILES)
    started = perf_counter()
    raw_header = {
        "schema_version": SCHEMA_VERSION,
        "video": metadata["video"],
        "provenance": dict(provenance),
        "class_names": metadata["class_names"],
    }
    if not track_cached:
        print(f"Running {tracker_config.kind} tracking...", flush=True)
        _track(
            video,
            output,
            tracker_config,
            shot_config,
            decode_config,
            detector_config,
            raw_header,
            court_config,
            reid_weights,
        )
        complete_stage(output, "track", track_key, TRACK_FILES)
    else:
        print("Reusing saved raw tracks.", flush=True)
    tracking_seconds = perf_counter() - started
    provenance.update(
        {
            "tracks_raw_sha256": sha256_file(output / "tracks_raw.json"),
            "postprocess_version": POSTPROCESS_VERSION,
            "postprocess": postprocess_config.model_dump(),
            "preview": preview,
        }
    )
    postprocess_key = fingerprint({"track_key": track_key, **provenance})
    postprocess_cached = stage_is_complete(
        output, "postprocess", postprocess_key, postprocess_files
    )
    started = perf_counter()
    if not postprocess_cached:
        print("Post-processing tracks" + (" and writing preview..." if preview else "..."))
        _postprocess(
            video,
            output,
            postprocess_config,
            decode_config,
            {"provenance": provenance},
            court_config,
            preview,
        )
        complete_stage(output, "postprocess", postprocess_key, postprocess_files)
    else:
        print("Reusing saved post-processed tracks.", flush=True)
    result = {
        "output_directory": str(output),
        "tracks": str(output / "tracks.json"),
        "tracks_raw": str(output / "tracks_raw.json"),
        "preview": str(output / PREVIEW_FILE) if preview else None,
        "tracking_summary": str(output / "tracking_summary.json"),
        "frames": metadata["frame_count"],
        "detections": metadata["detection_count"],
        "reused_detection": detect_cached,
        "reused_tracking": track_cached,
        "reused_postprocess": postprocess_cached,
        "detection_seconds": detection_seconds,
        "tracking_seconds": tracking_seconds,
        "postprocess_and_preview_seconds": perf_counter() - started,
        "provenance": provenance,
    }
    write_json(output / "run.json", result)
    return result


def add_tracking_arguments(parser: argparse.ArgumentParser):
    """Tracker, shot, and post-processing flags shared with remote/infer.py's defaults."""
    tracker, shots, post = TrackerConfig(), ShotConfig(), PostprocessConfig()
    parser.add_argument("--tracker", choices=TRACKER_KINDS, default=tracker.kind)
    parser.add_argument(
        "--duplicate-iou",
        type=float,
        default=tracker.duplicate_iou,
        help="Merge boxes overlapping this much into one person; >1 disables",
    )
    parser.add_argument("--max-age", type=int, default=tracker.max_age, help="Processed frames")
    parser.add_argument("--min-hits", type=int, default=tracker.min_hits)
    parser.add_argument("--iou-threshold", type=float, default=tracker.iou_threshold)
    parser.add_argument("--track-threshold", type=float, default=tracker.track_threshold)
    parser.add_argument("--use-byte", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--appearance",
        action=argparse.BooleanOptionalAction,
        default=tracker.appearance,
        help="Use ReID embeddings in trackers that support them (not ocsort)",
    )
    parser.add_argument(
        "--reid-weights",
        default=tracker.reid_weights,
        help="BoxMOT ReID checkpoint name, looked up in tracking/weights/reid",
    )
    parser.add_argument("--shots", action=argparse.BooleanOptionalAction, default=shots.enabled)
    parser.add_argument("--shot-threshold", type=float, default=shots.threshold)
    parser.add_argument("--min-shot-seconds", type=float, default=shots.min_shot_seconds)
    parser.add_argument("--backfill", action=argparse.BooleanOptionalAction, default=post.backfill)
    parser.add_argument(
        "--role-boundary",
        type=float,
        default=post.role_boundary,
        help="Referee share of the class vote separating player from referee",
    )
    parser.add_argument("--role-margin", type=float, default=post.role_margin)
    parser.add_argument(
        "--stitch-max-gap", type=float, default=post.stitch_max_gap_s, help="Seconds; 0 disables"
    )
    parser.add_argument(
        "--interpolate-max-gap",
        type=float,
        default=post.interpolate_max_gap_s,
        help="Seconds; 0 disables",
    )
    parser.add_argument("--min-track-seconds", type=float, default=post.min_track_s)


def tracking_configs(args) -> tuple[TrackerConfig, ShotConfig, PostprocessConfig]:
    return (
        TrackerConfig(
            kind=args.tracker,
            duplicate_iou=args.duplicate_iou,
            max_age=args.max_age,
            min_hits=args.min_hits,
            iou_threshold=args.iou_threshold,
            track_threshold=args.track_threshold,
            use_byte=args.use_byte,
            appearance=args.appearance,
            reid_weights=args.reid_weights,
        ),
        ShotConfig(
            enabled=args.shots,
            threshold=args.shot_threshold,
            min_shot_seconds=args.min_shot_seconds,
        ),
        PostprocessConfig(
            backfill=args.backfill,
            role_boundary=args.role_boundary,
            role_margin=args.role_margin,
            stitch_max_gap_s=args.stitch_max_gap,
            interpolate_max_gap_s=args.interpolate_max_gap,
            min_track_s=args.min_track_seconds,
        ),
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--imgsz", type=int, default=DetectorConfig().imgsz)
    parser.add_argument("--confidence", type=float, default=DetectorConfig().confidence)
    parser.add_argument(
        "--batch-size", type=int, default=DetectorConfig().batch_size, help="Frames per YOLO pass"
    )
    parser.add_argument("--court", type=Path, help="Court boundary JSON for a fixed-camera shot")
    parser.add_argument("--stride", type=int, choices=(1, 2), default=1)
    parser.add_argument("--max-seconds", type=float)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument(
        "--reid-path", type=Path, help="Explicit ReID checkpoint file (name must match)"
    )
    parser.add_argument("--preview", action=argparse.BooleanOptionalAction, default=True)
    add_tracking_arguments(parser)
    args = parser.parse_args()
    tracker_config, shot_config, postprocess_config = tracking_configs(args)
    result = run_pipeline(
        args.video,
        args.weights,
        args.output,
        DetectorConfig(
            imgsz=args.imgsz,
            confidence=args.confidence,
            device=args.device,
            batch_size=args.batch_size,
        ),
        tracker_config,
        DecodeConfig(stride=args.stride, max_seconds=args.max_seconds, max_frames=args.max_frames),
        CourtConfig.from_file(args.court) if args.court else None,
        shot_config=shot_config,
        postprocess_config=postprocess_config,
        reid_weights=args.reid_path,
        preview=args.preview,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
