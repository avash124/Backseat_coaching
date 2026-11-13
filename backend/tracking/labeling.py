"""Labeling tools: sample frames to annotate, seed tracking ground truth, import CVAT exports.

Run from ``backend/``. The step-by-step human workflow (sample -> CVAT -> correct -> export ->
import -> evaluate) is in the docstring of ``tracking/evaluate.py``.

  python -m tracking.labeling sample --video V --group g001-b --start-s 205 --out DIR
  python -m tracking.labeling export-tracks --tracks RUN/tracks.json --group g001-a --out DIR
  python -m tracking.labeling import-cvat --format yolo --zip EXPORT.zip --out DIR --images DIR
  python -m tracking.labeling import-cvat --format mot --zip EXPORT.zip --out DIR/gt/gt.txt

Shared conventions (the dataset builder depends on these; change them only together)
-------------------------------------------------------------------------------------
- The split unit is a *group*: a game, or a game segment while we only have one game. Group
  ids match ``[a-z0-9-]+``. No underscores, so ``stem.rpartition("_")`` always recovers it.
- Frame images are ``{group_id}_{frame_index:07d}.jpg``. ``frame_index`` is the 0-based index
  of the decoded frame in the source video, enumerated exactly as ``tracking/decode.py`` does.
- Labels are YOLO txt files with the same stem: ``class cx cy w h``, normalized to [0, 1].
  Canonical class ids are 0 player and 1 referee. 2 basketball and 3 hoop are reserved. The ids
  are frozen: reordering them silently corrupts every label file already written.
- Quarantine: ``tracking/evalsets/quarantine.yaml`` lists groups that never enter training.
- Tracking ground truth is MOTChallenge ``gt.txt``: ``frame,id,x,y,w,h,conf,class,visibility``
  with frame = frame_index + 1, pixel xywh, conf 1 (0 = ignore this box), class 1 = player.

Sampling heuristic (``sample``)
-------------------------------
Picks frames at least ``--min-gap-s`` apart (default 4 s). Frames closer than that are
near-duplicates: they inflate every metric and add little information. Two signals are used:

1. **Spacing and coverage.** The maximum count that fits a range is computed exactly by the
   greedy earliest-first rule. Unless ``--max-frames`` asks for fewer, that many frames are
   taken. Each next pick must land in a window from ``gap`` to ``remaining_span / remaining``
   seconds after the previous pick, and must leave room for the rest. This spreads the picks
   over the whole range.
2. **Diversity.** Every decoded frame gets a 32x18 grayscale thumbnail. Within the window,
   the frame chosen is the one whose thumbnail differs most (mean absolute difference) from
   every frame already chosen (farthest-point). Ties go to the frame nearest the window
   centre. At maximum density the window is about one frame wide, so the result is a
   regular grid. When there is slack, near-repeats such as dead-ball stretches are skipped.

``--quota START:END:COUNT`` (repeatable) samples each range separately. It enforces a minimum
number of frames from each part of the game, e.g. more frames from transition-heavy minutes.
The spacing rule also holds across range boundaries. Detection-count signals were left out
deliberately: they would require running the detector on every frame.

Pre-labels are written by the current weights at a low threshold (``--conf``, default 0.25).
They are a starting point for correction, never ground truth.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import zipfile
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import yaml

TRACKING_DIR = Path(__file__).resolve().parent
BACKEND_DIR = TRACKING_DIR.parent
DEFAULT_WEIGHTS = TRACKING_DIR / "weights" / "players-v1.pt"
QUARANTINE_FILE = TRACKING_DIR / "evalsets" / "quarantine.yaml"
GROUPS_FILE = TRACKING_DIR / "evalsets" / "groups.yaml"

GROUP_ID_RE = re.compile(r"[a-z0-9-]+")
FRAME_STEM_RE = re.compile(r"(?P<group>[a-z0-9-]+)_(?P<frame>\d{7})")
# Public datasets use other zero-padding; accepted only where explicitly requested.
LOOSE_FRAME_STEM_RE = re.compile(r"(?P<group>[a-z0-9-]+)_(?P<frame>\d+)")
CLASS_NAMES = ("player", "referee", "basketball", "hoop")  # index = canonical id; frozen
LABEL_CLASSES = ("player", "referee")  # classes labeled today
MOT_PLAYER_CLASS = 1
MOT_LABELS = ("player",)  # MOT class k = MOT_LABELS[k - 1]
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")
SPACING_TOLERANCE_S = 1e-6
THUMBNAIL_SIZE = (32, 18)


# --------------------------------------------------------------------------- conventions


def check_group_id(group_id: str) -> str:
    if not isinstance(group_id, str) or not GROUP_ID_RE.fullmatch(group_id):
        raise ValueError(f"Group id {group_id!r} must match [a-z0-9-]+ (no underscores)")
    return group_id


def frame_stem(group_id: str, frame_index: int) -> str:
    check_group_id(group_id)
    if not 0 <= int(frame_index) <= 9_999_999:
        raise ValueError(f"Frame index {frame_index} does not fit seven digits")
    return f"{group_id}_{int(frame_index):07d}"


def parse_frame_stem(stem: str, strict: bool = True) -> tuple[str, int]:
    match = (FRAME_STEM_RE if strict else LOOSE_FRAME_STEM_RE).fullmatch(stem)
    if not match:
        expected = "{group_id}_{frame_index:07d}" if strict else "{group_id}_{digits}"
        raise ValueError(f"File stem {stem!r} does not match {expected}")
    return match["group"], int(match["frame"])


def load_quarantine(path: Path = QUARANTINE_FILE) -> frozenset[str]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("groups"), list):
        raise TypeError(f"{path} must contain a 'groups:' list")
    return frozenset(check_group_id(group) for group in data["groups"])


def load_group_registry(path: Path = GROUPS_FILE) -> dict[str, dict]:
    """Registered groups (evalsets/groups.yaml): source video hash and time span per group."""
    if not Path(path).is_file():
        return {}
    groups = (yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}).get("groups") or {}
    if not isinstance(groups, dict):
        raise TypeError(f"{path}: 'groups' must be a mapping of group id -> span")
    for group_id in groups:
        check_group_id(group_id)
    return groups


def check_registered_span(
    group_id: str,
    video_sha256: str,
    spans: Sequence[tuple[float, float | None]],
    registry: dict[str, dict],
):
    """Refuse sampling a registered group from another video or outside its time span."""
    entry = registry.get(group_id)
    if entry is None:
        return
    expected = entry.get("video_sha256")
    if expected and expected != video_sha256:
        raise ValueError(f"Group {group_id} is registered to a different video ({expected[:12]})")
    low, high = float(entry.get("start_s") or 0.0), entry.get("end_s")
    for start, end in spans:
        if start < low - 1e-9 or (high is not None and (end is None or end > high + 1e-9)):
            raise ValueError(
                f"Range [{start}, {end}) leaves group {group_id}'s span [{low}, {high})"
            )


def sha256_file(path: Path) -> str:
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- YOLO labels


def boxes_to_yolo_lines(boxes: np.ndarray, width: int, height: int) -> list[str]:
    """Convert (N, 5) ``[class_id, x1, y1, x2, y2]`` pixel boxes to YOLO label lines."""
    lines = []
    for class_id, x1, y1, x2, y2 in np.asarray(boxes, dtype=float).reshape(-1, 5):
        x1, x2 = np.clip([x1, x2], 0, width)
        y1, y2 = np.clip([y1, y2], 0, height)
        if x2 <= x1 or y2 <= y1:
            continue
        lines.append(
            f"{int(class_id)} {(x1 + x2) / 2 / width:.6f} {(y1 + y2) / 2 / height:.6f} "
            f"{(x2 - x1) / width:.6f} {(y2 - y1) / height:.6f}"
        )
    return lines


def write_yolo_labels(path: Path, boxes: np.ndarray, width: int, height: int):
    lines = boxes_to_yolo_lines(boxes, width, height)
    Path(path).write_text("".join(line + "\n" for line in lines), encoding="utf-8")


def parse_yolo_text(text: str, source: str = "<labels>") -> np.ndarray:
    """Parse YOLO lines into (N, 5) ``[class_id, cx, cy, w, h]`` (normalized), validated."""
    rows = []
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"{source}:{number}: expected 5 fields, got {len(fields)}")
        try:
            class_id = int(fields[0])
            cx, cy, w, h = (float(value) for value in fields[1:])
        except ValueError as error:
            raise ValueError(f"{source}:{number}: {error}") from None
        values = (cx, cy, w, h)
        if class_id < 0 or not all(math.isfinite(v) and -1e-6 <= v <= 1 + 1e-6 for v in values):
            raise ValueError(f"{source}:{number}: invalid class or normalized coordinates")
        if w <= 0 or h <= 0:
            raise ValueError(f"{source}:{number}: zero-area box")
        rows.append((class_id, cx, cy, w, h))
    return np.asarray(rows, dtype=float).reshape(-1, 5)


def read_yolo_labels(path: Path, width: int, height: int) -> np.ndarray:
    """Read a YOLO label file as (N, 5) ``[class_id, x1, y1, x2, y2]`` in pixels."""
    rows = parse_yolo_text(Path(path).read_text(encoding="utf-8"), str(path))
    class_id, cx, cy, w, h = rows.T
    return np.column_stack(
        (
            class_id,
            (cx - w / 2) * width,
            (cy - h / 2) * height,
            (cx + w / 2) * width,
            (cy + h / 2) * height,
        )
    ).reshape(-1, 5)


# --------------------------------------------------------------------------- MOT files


def format_mot_rows(rows: Iterable[Sequence[float]]) -> str:
    """Rows are ``(frame, id, x, y, w, h, conf, class, visibility)`` with frame already 1-based."""
    ordered = sorted((tuple(row) for row in rows), key=lambda row: (int(row[0]), int(row[1])))
    return "".join(
        f"{int(f)},{int(i)},{x:.2f},{y:.2f},{w:.2f},{h:.2f},{c:g},{int(k)},{v:g}\n"
        for f, i, x, y, w, h, c, k, v in ordered
    )


def write_mot(path: Path, rows: Iterable[Sequence[float]]) -> int:
    text = format_mot_rows(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return text.count("\n")


def parse_mot_text(text: str, source: str = "<gt.txt>") -> np.ndarray:
    """Parse MOT lines into an (N, 9) float array; extra columns are ignored."""
    rows = []
    seen = set()
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) < 9:
            raise ValueError(f"{source}:{number}: MOT ground truth needs 9 columns")
        try:
            row = [float(value) for value in fields[:9]]
        except ValueError as error:
            raise ValueError(f"{source}:{number}: {error}") from None
        if not all(math.isfinite(value) for value in row):
            raise ValueError(f"{source}:{number}: non-finite value")
        if row[0] < 1 or row[0] != int(row[0]) or row[1] != int(row[1]):
            raise ValueError(f"{source}:{number}: frame must be an integer >= 1, id an integer")
        if row[4] <= 0 or row[5] <= 0:
            raise ValueError(f"{source}:{number}: box width and height must be positive")
        key = (int(row[0]), int(row[1]))
        if key in seen:
            raise ValueError(f"{source}:{number}: duplicate id {key[1]} in frame {key[0]}")
        seen.add(key)
        rows.append(row)
    return np.asarray(rows, dtype=float).reshape(-1, 9)


def read_mot(path: Path) -> np.ndarray:
    return parse_mot_text(Path(path).read_text(encoding="utf-8"), str(path))


# --------------------------------------------------------------------------- tracks.json


def load_tracks(path: Path) -> dict:
    """Read tracks.json generically: extra fields and newer schema versions are tolerated."""
    artifact = json.loads(Path(path).read_text(encoding="utf-8"))
    frames = artifact.get("frames")
    if not isinstance(frames, list):
        raise TypeError(f"{path}: expected a 'frames' list")
    previous = -1
    for frame in frames:
        index = frame.get("frame_index")
        if not isinstance(index, int) or index <= previous:
            raise ValueError(f"{path}: frame_index must be integers increasing strictly")
        previous = index
        for track in frame.get("tracks", []):
            box = track.get("bbox_xyxy")
            if not isinstance(track.get("track_id"), int) or not box or len(box) != 4:
                raise ValueError(f"{path}: frame {index} has a track without id or bbox_xyxy")
    return artifact


def keep_track(track: dict, include_all_roles: bool = False, observed_only: bool = False) -> bool:
    """Default policy: tracks with a ``role`` count only when the role is "player"."""
    if observed_only and track.get("observed") is False:
        return False
    role = track.get("role")
    return include_all_roles or role is None or role == "player"


def tracks_to_mot_rows(
    artifact: dict,
    width: int,
    height: int,
    start_frame: int = 0,
    end_frame: int | None = None,
    include_all_roles: bool = False,
    observed_only: bool = False,
) -> list[tuple]:
    """MOT rows for tracks in ``[start_frame, end_frame]``; MOT id = track_id + 1."""
    rows = []
    for frame in artifact["frames"]:
        index = frame["frame_index"]
        if index < start_frame or (end_frame is not None and index > end_frame):
            continue
        for track in frame.get("tracks", []):
            if not keep_track(track, include_all_roles, observed_only):
                continue
            x1, y1, x2, y2 = (float(value) for value in track["bbox_xyxy"])
            x1, x2 = max(0.0, x1), min(float(width), x2)
            y1, y2 = max(0.0, y1), min(float(height), y2)
            if x2 <= x1 or y2 <= y1:
                continue
            rows.append(
                (index + 1, track["track_id"] + 1, x1, y1, x2 - x1, y2 - y1, 1, MOT_PLAYER_CLASS, 1)
            )
    return rows


# --------------------------------------------------------------------------- frame selection


def _capacity(times: np.ndarray, min_gap_s: float) -> tuple[np.ndarray, np.ndarray]:
    """next[j]: first index >= gap after j. capacity[j]: max picks starting with j (exact)."""
    following = np.searchsorted(times, times + min_gap_s - SPACING_TOLERANCE_S, side="left")
    capacity = np.ones(len(times), dtype=np.int64)
    for j in range(len(times) - 1, -1, -1):
        if following[j] < len(times):
            capacity[j] = 1 + capacity[following[j]]
    return following, capacity


def max_sample_count(times: Sequence[float], min_gap_s: float) -> int:
    times = np.asarray(times, dtype=float)
    return int(_capacity(times, min_gap_s)[1][0]) if len(times) else 0


def _select_range(times, signatures, chosen, min_gap_s, count):
    """Pick indices into ``times`` (sorted) per the module docstring's heuristic."""
    following, capacity = _capacity(times, min_gap_s)
    total = int(capacity[0]) if count is None else min(int(count), int(capacity[0]))
    picks = []
    last = times[0] - min_gap_s  # virtual previous pick: the first may sit at the range start
    low = 0
    for remaining in range(total, 0, -1):
        high = int(np.flatnonzero(capacity >= remaining)[-1])
        stride = (times[-1] - last) / remaining
        window_start, window_end = last + min_gap_s, last + max(stride, min_gap_s)
        candidates = np.arange(low, high + 1)
        in_window = candidates[
            (times[candidates] >= window_start - SPACING_TOLERANCE_S)
            & (times[candidates] <= window_end + SPACING_TOLERANCE_S)
        ]
        if len(in_window) == 0:
            in_window = candidates[:1]  # earliest feasible frame always keeps the plan valid
        centre = (window_start + window_end) / 2
        if signatures is not None and chosen:
            stacked = np.stack(chosen)
            novelty = np.array(
                [np.abs(stacked - signatures[j]).mean(axis=(1, 2)).min() for j in in_window]
            )
        else:
            novelty = np.zeros(len(in_window))
        order = np.lexsort((np.abs(times[in_window] - centre), -novelty))
        pick = int(in_window[order[0]])
        picks.append(pick)
        if signatures is not None:
            chosen.append(signatures[pick])
        last = times[pick]
        low = int(following[pick])
    return picks


def plan_samples(
    times: Sequence[float],
    min_gap_s: float = 4.0,
    signatures: np.ndarray | None = None,
    ranges: Sequence[tuple[float, float | None, int | None]] | None = None,
) -> list[int]:
    """Choose candidate indices at least ``min_gap_s`` apart.

    ``times`` must increase. ``ranges`` holds ``(start_s, end_s, count)`` entries; start is
    inclusive and end exclusive. ``end_s=None`` means the end of the video, and
    ``count=None`` means the maximum density. Without ranges, all times form one range.
    """
    times = np.asarray(times, dtype=float)
    if min_gap_s <= 0:
        raise ValueError("min_gap_s must be positive")
    if np.any(np.diff(times) <= 0):
        raise ValueError("Candidate times must increase strictly")
    ranges = list(ranges) if ranges else [(-math.inf, None, None)]
    chosen_signatures: list[np.ndarray] = []
    picks: list[int] = []
    for start, end, count in sorted(ranges, key=lambda item: item[0]):
        earliest = start if not picks else max(start, times[picks[-1]] + min_gap_s)
        mask = times >= earliest - SPACING_TOLERANCE_S
        if end is not None:
            mask &= times < end
        indices = np.flatnonzero(mask)
        if len(indices) == 0 or count == 0:
            continue
        local = _select_range(
            times[indices],
            None if signatures is None else signatures[indices],
            chosen_signatures,
            min_gap_s,
            count,
        )
        picks.extend(int(indices[j]) for j in local)
    return picks


# --------------------------------------------------------------------------- video access


@dataclass(frozen=True)
class VideoScan:
    width: int
    height: int
    fps: float
    decoded_frames: int
    frame_indices: np.ndarray
    timestamps: np.ndarray
    thumbnails: np.ndarray


def _decoded_frames(video: Path):
    """Yield ``(frame_index, timestamp_s, frame, (width, height, fps))``.

    Indices and timestamps follow tracking/decode.py exactly: enumerate every decoded frame
    from 0, timestamps relative to the first decoded frame.
    """
    import av

    with av.open(str(video)) as container:
        if not container.streams.video:
            raise ValueError(f"No video stream in {video}")
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        rate = stream.average_rate or stream.guessed_rate
        if rate is None:
            raise ValueError("Video has no usable frame rate; transcode it first")
        fps = float(rate)
        origin = None
        for index, frame in enumerate(container.decode(stream)):
            timestamp = frame.time
            if timestamp is None:
                timestamp = (origin or 0.0) + index / fps
            if origin is None:
                origin = timestamp
            yield index, timestamp - origin, frame, (stream.width, stream.height, fps)


def scan_video(
    video: Path, start_s: float = 0.0, end_s: float | None = None, step: int = 1
) -> VideoScan:
    indices, timestamps, thumbnails = [], [], []
    width = height = 0
    fps = 0.0
    decoded = 0
    width_px, height_px = THUMBNAIL_SIZE
    for index, timestamp, frame, (width, height, fps) in _decoded_frames(video):
        if end_s is not None and timestamp >= end_s:
            break
        decoded = index + 1
        if timestamp < start_s or index % step:
            continue
        thumbnail = frame.to_ndarray(
            width=width_px, height=height_px, format="gray", interpolation="AREA"
        )
        indices.append(index)
        timestamps.append(timestamp)
        thumbnails.append(thumbnail.astype(np.float32))
    if not indices:
        raise ValueError(f"No frames of {video} fall in [{start_s}, {end_s})")
    return VideoScan(
        width,
        height,
        fps,
        decoded,
        np.asarray(indices),
        np.asarray(timestamps),
        np.stack(thumbnails),
    )


def write_frames(
    video: Path, frame_indices: Iterable[int], directory: Path, group_id: str, quality: int = 95
) -> list[dict]:
    """Decode the video once more and save the selected frames as JPEG images."""
    import cv2

    wanted = sorted({int(index) for index in frame_indices})
    if not wanted:
        return []
    directory.mkdir(parents=True, exist_ok=True)
    written = []
    position = 0
    for index, timestamp, frame, _ in _decoded_frames(video):
        if index != wanted[position]:
            continue
        name = frame_stem(group_id, index) + ".jpg"
        image = frame.to_ndarray(format="bgr24")
        if not cv2.imwrite(str(directory / name), image, [cv2.IMWRITE_JPEG_QUALITY, quality]):
            raise OSError(f"Could not write {directory / name}")
        written.append({"frame_index": index, "timestamp_s": round(timestamp, 6), "image": name})
        position += 1
        if position == len(wanted):
            break
    if len(written) != len(wanted):
        raise ValueError("The video decoded fewer frames on the second pass than the first")
    return written


# --------------------------------------------------------------------------- detector


class Detector:
    """Load YOLO directly; return canonical-class boxes ``(N, 6) [x1, y1, x2, y2, conf, cls]``.

    Model class names are mapped to canonical ids by name. ``class_map`` adds aliases such
    as ``{"person": "player"}`` for stock COCO weights. Unmapped model classes are dropped.
    ``nms=False`` selects YOLO26's NMS-free one-to-one head, as tracking/detect.py does.
    """

    def __init__(
        self,
        weights: Path,
        imgsz: int = 1280,
        conf: float = 0.25,
        device: str = "cpu",
        max_det: int = 100,
        class_map: dict[str, str] | None = None,
        nms: bool = False,
    ):
        from ultralytics import YOLO

        weights = Path(weights).resolve()
        if not weights.is_file():
            raise FileNotFoundError(f"Weights not found: {weights}")
        self.weights = weights
        self.model = YOLO(str(weights))
        self.names = {int(key): value for key, value in self.model.names.items()}
        aliases = {name: name for name in CLASS_NAMES} | dict(class_map or {})
        self.class_lookup = {
            model_id: CLASS_NAMES.index(aliases[name])
            for model_id, name in self.names.items()
            if aliases.get(name) in CLASS_NAMES
        }
        if not self.class_lookup:
            raise ValueError(f"No canonical classes among the weights' names {self.names}")
        self.settings = {"imgsz": imgsz, "conf": conf, "device": device, "max_det": max_det}
        self.nms = nms

    def __call__(self, images: Sequence) -> list[np.ndarray]:
        outputs = []
        for start in range(0, len(images), 8):
            results = self.model.predict(
                source=[str(item) if isinstance(item, Path) else item for item in
                        images[start:start + 8]],
                classes=sorted(self.class_lookup),
                nms=self.nms,
                verbose=False,
                **self.settings,
            )
            for result in results:
                boxes = result.boxes
                classes = boxes.cls.cpu().numpy().astype(int)
                canonical = np.array([self.class_lookup[c] for c in classes], dtype=np.float32)
                outputs.append(
                    np.column_stack(
                        (boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy(), canonical)
                    ).astype(np.float32).reshape(-1, 6)
                )
        return outputs


# --------------------------------------------------------------------------- sample command


def _zip_files(target: Path, members: Iterable[tuple[str, Path | str]]):
    with zipfile.ZipFile(target, "w") as archive:
        for name, source in members:
            if isinstance(source, Path):
                compression = (
                    zipfile.ZIP_STORED if source.suffix.lower() in IMAGE_SUFFIXES
                    else zipfile.ZIP_DEFLATED
                )
                archive.write(source, name, compress_type=compression)
            else:
                archive.writestr(name, source, compress_type=zipfile.ZIP_DEFLATED)


def write_annotation_packages(directory: Path, names: Sequence[str] = LABEL_CLASSES) -> dict:
    """Write data.yaml and CVAT/Roboflow zips for ``directory/{images,labels}``."""
    images = sorted(p for p in (directory / "images").iterdir() if p.suffix in IMAGE_SUFFIXES)
    labels = [directory / "labels" / f"{image.stem}.txt" for image in images]
    if list(names) != list(CLASS_NAMES[: len(names)]):
        raise ValueError("Package names must be a prefix of the canonical CLASS_NAMES")
    (directory / "data.yaml").write_text(
        "# Next-Up labeling batch. Labels are model PRE-LABELS until a human corrects them.\n"
        "# Class ids are canonical (tracking/labeling.py CLASS_NAMES); do not reorder.\n"
        f"train: images\nval: images\nnc: {len(names)}\n"
        f"names: [{', '.join(repr(name) for name in names)}]\n",
        encoding="utf-8",
    )
    cvat_images = directory / "cvat_images.zip"
    _zip_files(cvat_images, ((image.name, image) for image in images))
    cvat_annotations = directory / "cvat_yolo11_annotations.zip"
    _zip_files(
        cvat_annotations,
        [
            ("obj.names", "".join(f"{name}\n" for name in names)),
            (
                "obj.data",
                (
                    f"classes = {len(names)}\ntrain = data/train.txt\n"
                    "names = data/obj.names\nbackup = backup/\n"
                ),
            ),
            ("train.txt", "".join(f"data/obj_train_data/{image.name}\n" for image in images)),
            *((f"obj_train_data/{label.name}", label) for label in labels),
        ],
    )
    yolo_dataset = directory / "yolo_dataset.zip"
    _zip_files(
        yolo_dataset,
        [
            ("data.yaml", directory / "data.yaml"),
            *((f"images/{image.name}", image) for image in images),
            *((f"labels/{label.name}", label) for label in labels),
        ],
    )
    return {
        "data_yaml": "data.yaml",
        "cvat_images_zip": cvat_images.name,
        "cvat_yolo11_annotations_zip": cvat_annotations.name,
        "yolo_dataset_zip": yolo_dataset.name,
    }


def sample(
    video: Path,
    group_id: str,
    out: Path,
    start_s: float | None = None,
    end_s: float | None = None,
    min_gap_s: float = 4.0,
    max_frames: int | None = None,
    quotas: Sequence[tuple[float, float | None, int | None]] | None = None,
    candidate_step: int = 1,
    detector: Detector | None = None,
    jpeg_quality: int = 95,
    overwrite: bool = False,
    density_gaps: Sequence[float] = (4.0, 2.0, 1.0),
    group_registry: dict[str, dict] | None = None,
) -> dict:
    """Select frames, write images and pre-labels, and package them for annotation.

    ``start_s``/``end_s`` default to the group's registered span (evalsets/groups.yaml), or
    to the whole video for an unregistered group.
    """
    check_group_id(group_id)
    video, out = Path(video).resolve(), Path(out).resolve()
    registry = load_group_registry() if group_registry is None else group_registry
    entry = registry.get(group_id) or {}
    if start_s is None:
        start_s = float(entry.get("start_s") or 0.0)
    if end_s is None and entry.get("end_s") is not None:
        end_s = float(entry["end_s"])
    video_sha256 = sha256_file(video)
    check_registered_span(
        group_id,
        video_sha256,
        [(q[0], q[1]) for q in quotas] if quotas else [(start_s, end_s)],
        registry,
    )
    images_dir, labels_dir = out / "images", out / "labels"
    if not overwrite and any(path.exists() and any(path.iterdir())
                             for path in (images_dir, labels_dir)):
        raise FileExistsError(f"{out} already has images or labels; pass overwrite to replace")
    for directory in (images_dir, labels_dir):
        directory.mkdir(parents=True, exist_ok=True)
        if overwrite:
            for stale in directory.iterdir():
                if stale.is_file():
                    stale.unlink()
    scan_start = min([start_s, *(q[0] for q in quotas or [])])
    scan_ends = [end_s, *(q[1] for q in quotas or [])] if quotas else [end_s]
    scan_end = None if any(end is None for end in scan_ends) else max(scan_ends)
    scan = scan_video(video, scan_start, scan_end, candidate_step)
    ranges = list(quotas) if quotas else [(start_s, end_s, max_frames)]
    picks = plan_samples(scan.timestamps, min_gap_s, scan.thumbnails, ranges)
    selected = scan.frame_indices[picks]
    frames = write_frames(video, selected, images_dir, group_id, jpeg_quality)

    prelabel = None
    class_counts: Counter = Counter()
    if detector is not None:
        paths = [images_dir / frame["image"] for frame in frames]
        predictions = detector(paths)
        with (out / "prelabels.jsonl").open("w", encoding="utf-8") as handle:
            for frame, boxes in zip(frames, predictions):
                boxes = boxes[np.isin(boxes[:, 5], [CLASS_NAMES.index(n) for n in LABEL_CLASSES])]
                labels = np.column_stack((boxes[:, 5], boxes[:, :4])) if len(boxes) else boxes
                write_yolo_labels(
                    labels_dir / f"{Path(frame['image']).stem}.txt",
                    labels.reshape(-1, 5),
                    scan.width,
                    scan.height,
                )
                counts = Counter(CLASS_NAMES[int(c)] for c in boxes[:, 5])
                class_counts.update(counts)
                frame["prelabel_boxes"] = dict(counts)
                handle.write(
                    json.dumps(
                        {
                            "image": frame["image"],
                            "boxes_xyxy_conf_cls": np.round(boxes, 3).tolist(),
                        }
                    )
                    + "\n"
                )
        weights = getattr(detector, "weights", None)
        prelabel = {
            "status": "prelabel",
            "weights": str(weights) if weights else None,
            "weights_sha256": sha256_file(weights) if weights else None,
            "model_names": detector.names,
            "nms": detector.nms,
            **detector.settings,
            "boxes": dict(class_counts),
        }
    else:
        for frame in frames:
            (labels_dir / f"{Path(frame['image']).stem}.txt").write_text("", encoding="utf-8")

    timestamps = [frame["timestamp_s"] for frame in frames]
    in_range = scan.timestamps >= start_s
    if end_s is not None:
        in_range &= scan.timestamps < end_s
    try:
        quarantined = group_id in load_quarantine()
    except (OSError, ValueError):
        quarantined = None
    batch = {
        "kind": "labeling_batch",
        "batch_version": 1,
        "created_utc": _utc_now(),
        "group_id": group_id,
        "quarantined": quarantined,
        "video": {
            "path": str(video),
            "sha256": video_sha256,
            "width": scan.width,
            "height": scan.height,
            "fps": scan.fps,
            "frames_decoded_through_range": scan.decoded_frames,
        },
        "selection": {
            "start_s": start_s,
            "end_s": end_s,
            "quotas": [list(q) for q in quotas] if quotas else None,
            "min_gap_s": min_gap_s,
            "max_frames": max_frames,
            "candidate_step_frames": candidate_step,
            "heuristic": "max-density spacing windows + farthest-point thumbnail novelty",
            "selected": len(frames),
            "min_spacing_s": float(np.diff(timestamps).min()) if len(timestamps) > 1 else None,
            "max_frames_at_gap_s": {
                f"{gap:g}": max_sample_count(scan.timestamps[in_range], gap)
                for gap in density_gaps
            },
        },
        "prelabel": prelabel,
        "frames": frames,
    }
    batch["packages"] = write_annotation_packages(out)
    _write_json(out / "batch.json", batch)
    return batch


# --------------------------------------------------------------------------- export-tracks


def _video_size(video: Path) -> tuple[int, int, float]:
    import av

    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        rate = stream.average_rate or stream.guessed_rate
        return stream.width, stream.height, float(rate) if rate else 0.0


def export_tracks(
    tracks: Path,
    out: Path,
    group_id: str,
    video: Path | None = None,
    start_frame: int = 0,
    end_frame: int | None = None,
    include_all_roles: bool = False,
    observed_only: bool = False,
) -> dict:
    """tracks.json -> MOT gt.txt seed + CVAT "MOT 1.1" zip, for correction rather than drawing."""
    check_group_id(group_id)
    tracks, out = Path(tracks).resolve(), Path(out).resolve()
    artifact = load_tracks(tracks)
    meta = artifact.get("video") or {}
    width, height, fps = meta.get("width"), meta.get("height"), meta.get("fps")
    if video is not None:
        video_width, video_height, video_fps = _video_size(video)
        if width and (width, height) != (video_width, video_height):
            raise ValueError("tracks.json frame size does not match the video")
        width, height, fps = video_width, video_height, fps or video_fps
    if not width or not height:
        raise ValueError("Frame size unknown: tracks.json has no video block; pass the video")
    frame_indices = [
        frame["frame_index"]
        for frame in artifact["frames"]
        if frame["frame_index"] >= start_frame and (end_frame is None
                                                     or frame["frame_index"] <= end_frame)
    ]
    if not frame_indices:
        raise ValueError("No tracks.json frames fall in the requested range")
    rows = tracks_to_mot_rows(
        artifact, width, height, start_frame, end_frame, include_all_roles, observed_only
    )
    gt_path = out / "gt" / "gt.txt"
    write_mot(gt_path, rows)
    (out / "gt" / "labels.txt").write_text("".join(f"{n}\n" for n in MOT_LABELS), "utf-8")
    first, last = frame_indices[0], frame_indices[-1]
    (out / "seqinfo.ini").write_text(
        "[Sequence]\n"
        f"name={group_id}_{first:07d}-{last:07d}\nimDir=img1\nframeRate={fps or 30:g}\n"
        f"seqLength={last + 1}\nimWidth={width}\nimHeight={height}\nimExt=.jpg\n",
        encoding="utf-8",
    )
    zip_path = out / "cvat_mot11.zip"
    _zip_files(zip_path, [("gt/gt.txt", gt_path), ("gt/labels.txt", out / "gt" / "labels.txt")])
    strides = Counter(np.diff(frame_indices).tolist())
    seed = {
        "kind": "tracking_gt_seed",
        "status": "seed",
        "note": "Uncorrected tracker output. Evaluating against it is a plumbing check only.",
        "created_utc": _utc_now(),
        "group_id": group_id,
        "tracks": str(tracks),
        "tracks_sha256": sha256_file(tracks),
        "tracks_schema_version": artifact.get("schema_version"),
        "video": str(Path(video).resolve()) if video else None,
        "width": width,
        "height": height,
        "fps": fps,
        "frame_range": [first, last],
        "frames": len(frame_indices),
        "frame_stride": strides.most_common(1)[0][0] if strides else 1,
        "rows": len(rows),
        "ids": len({row[1] for row in rows}),
        "role_filter": "all" if include_all_roles else "player-or-unset",
        "observed_only": observed_only,
        "gt_sha256": sha256_file(gt_path),
    }
    _write_json(out / "seed.json", seed)
    return seed


# --------------------------------------------------------------------------- import-cvat


def _zip_text(archive: zipfile.ZipFile, name: str) -> str:
    return archive.read(name).decode("utf-8-sig")


def import_cvat_yolo(
    zip_path: Path,
    out: Path,
    images: Path | None = None,
    group_id: str | None = None,
    drop_labels: Sequence[str] = (),
) -> dict:
    """CVAT "YOLO 1.1" (or Ultralytics YOLO) export -> ``out/labels`` with canonical ids."""
    out = Path(out).resolve()
    with zipfile.ZipFile(zip_path) as archive:
        members = archive.namelist()
        if "obj.names" in members:
            names = [line.strip() for line in _zip_text(archive, "obj.names").splitlines()
                     if line.strip()]
        elif "data.yaml" in members:
            raw = yaml.safe_load(_zip_text(archive, "data.yaml"))["names"]
            names = [raw[k] for k in sorted(raw)] if isinstance(raw, dict) else list(raw)
        else:
            raise ValueError("Export has neither obj.names nor data.yaml")
        unknown = [n for n in names if n not in CLASS_NAMES and n not in drop_labels]
        if unknown:
            raise ValueError(f"Labels {unknown} are not canonical; rename them or --drop-label")
        remap = {i: (CLASS_NAMES.index(n) if n in CLASS_NAMES else None)
                 for i, n in enumerate(names)}
        label_members = [
            m for m in members
            if m.endswith(".txt") and (
                re.match(r"(.*/)?obj_[^/]*_data/[^/]+\.txt$", m) or "labels/" in m
            )
        ]
        out_labels = out / "labels"
        out_labels.mkdir(parents=True, exist_ok=True)
        counts: Counter = Counter()
        stems = set()
        for member in sorted(label_members):
            stem = Path(member).stem
            group, _ = parse_frame_stem(stem)
            if group_id is not None and group != group_id:
                raise ValueError(f"{member} belongs to group {group}, expected {group_id}")
            if stem in stems:
                raise ValueError(f"Duplicate label file for {stem}")
            stems.add(stem)
            rows = parse_yolo_text(_zip_text(archive, member), member)
            lines = []
            for class_id, cx, cy, w, h in rows:
                if int(class_id) not in remap:
                    raise ValueError(f"{member}: class {int(class_id)} not in the export names")
                target = remap[int(class_id)]
                if target is None:
                    counts["dropped"] += 1
                    continue
                counts[CLASS_NAMES[target]] += 1
                lines.append(f"{target} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
            (out_labels / f"{stem}.txt").write_text("".join(lines), encoding="utf-8")
    if images is not None:
        image_stems = {p.stem for p in Path(images).iterdir() if p.suffix in IMAGE_SUFFIXES}
        missing, orphans = sorted(image_stems - stems), sorted(stems - image_stems)
        if missing or orphans:
            raise ValueError(f"Images without labels: {missing[:5]}; labels without images: "
                             f"{orphans[:5]}")
    summary = {
        "kind": "cvat_import",
        "format": "yolo",
        "status": "corrected",
        "created_utc": _utc_now(),
        "source_zip": str(Path(zip_path).resolve()),
        "source_zip_sha256": sha256_file(Path(zip_path)),
        "export_names": names,
        "label_files": len(stems),
        "boxes": dict(counts),
    }
    _write_json(out / "import.json", summary)
    return summary


def import_cvat_mot(zip_path: Path, out: Path, label: str = "player") -> dict:
    """CVAT "MOT 1.1" export -> our gt.txt (class 1 = player; other labels dropped)."""
    out = Path(out).resolve()
    with zipfile.ZipFile(zip_path) as archive:
        members = archive.namelist()
        gt_name = next((m for m in members if m.endswith("gt/gt.txt") or m == "gt.txt"), None)
        if gt_name is None:
            raise ValueError("Export has no gt/gt.txt")
        labels_name = gt_name[: -len("gt.txt")] + "labels.txt"
        labels = (
            [line.strip() for line in _zip_text(archive, labels_name).splitlines() if line.strip()]
            if labels_name in members else list(MOT_LABELS)
        )
        rows = parse_mot_text(_zip_text(archive, gt_name), gt_name)
    if label not in labels:
        raise ValueError(f"Label {label!r} not in the export's labels {labels}")
    wanted = labels.index(label) + 1
    kept = rows[rows[:, 7] == wanted]
    kept[:, 7] = MOT_PLAYER_CLASS
    write_mot(out, kept.tolist())
    summary = {
        "kind": "cvat_import",
        "format": "mot",
        "status": "corrected",
        "created_utc": _utc_now(),
        "source_zip": str(Path(zip_path).resolve()),
        "source_zip_sha256": sha256_file(Path(zip_path)),
        "labels": labels,
        "rows_kept": len(kept),
        "rows_dropped": int(len(rows) - len(kept)),
        "ids": len(np.unique(kept[:, 1])) if len(kept) else 0,
        "gt_sha256": sha256_file(out),
    }
    _write_json(out.with_name("import.json"), summary)
    return summary


# --------------------------------------------------------------------------- CLI


def _parse_quota(text: str) -> tuple[float, float | None, int | None]:
    try:
        start, end, count = text.split(":")
        return float(start), float(end) if end else None, int(count) if count else None
    except ValueError:
        raise argparse.ArgumentTypeError("quota must be START:END:COUNT (seconds)") from None


def main(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)

    sample_cmd = commands.add_parser("sample", help="Pick frames and pre-label them")
    sample_cmd.add_argument("--video", type=Path, required=True)
    sample_cmd.add_argument("--group", required=True, help="Group id, e.g. g001-b")
    sample_cmd.add_argument("--out", type=Path, required=True)
    sample_cmd.add_argument("--start-s", type=float, help="Default: the group's registered start")
    sample_cmd.add_argument("--end-s", type=float, help="Default: the group's registered end")
    sample_cmd.add_argument("--min-gap-s", type=float, default=4.0)
    sample_cmd.add_argument("--max-frames", type=int, help="Default: maximum density")
    sample_cmd.add_argument("--quota", type=_parse_quota, action="append",
                            help="START:END:COUNT; repeatable; overrides start/end/max-frames")
    sample_cmd.add_argument("--candidate-step", type=int, default=1)
    sample_cmd.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    sample_cmd.add_argument("--conf", type=float, default=0.25, help="Pre-label threshold")
    sample_cmd.add_argument("--imgsz", type=int, default=1280)
    sample_cmd.add_argument("--device", default="cpu")
    sample_cmd.add_argument("--no-prelabel", action="store_true")
    sample_cmd.add_argument("--jpeg-quality", type=int, default=95)
    sample_cmd.add_argument("--overwrite", action="store_true")

    export_cmd = commands.add_parser("export-tracks", help="tracks.json -> MOT GT seed")
    export_cmd.add_argument("--tracks", type=Path, required=True)
    export_cmd.add_argument("--out", type=Path, required=True)
    export_cmd.add_argument("--group", required=True)
    export_cmd.add_argument("--video", type=Path)
    export_cmd.add_argument("--start-frame", type=int, default=0)
    export_cmd.add_argument("--end-frame", type=int)
    export_cmd.add_argument("--all-roles", action="store_true")
    export_cmd.add_argument("--observed-only", action="store_true")

    import_cmd = commands.add_parser("import-cvat", help="Corrected CVAT export -> conventions")
    import_cmd.add_argument("--format", choices=("yolo", "mot"), required=True)
    import_cmd.add_argument("--zip", type=Path, required=True)
    import_cmd.add_argument("--out", type=Path, required=True,
                            help="yolo: output directory; mot: path of gt.txt to write")
    import_cmd.add_argument("--images", type=Path, help="yolo: check one label per image")
    import_cmd.add_argument("--group", help="yolo: require this group id")
    import_cmd.add_argument("--drop-label", action="append", default=[])
    import_cmd.add_argument("--label", default="player", help="mot: label to keep as class 1")

    args = parser.parse_args(argv)
    if args.command == "sample":
        detector = None if args.no_prelabel else Detector(
            args.weights, args.imgsz, args.conf, args.device
        )
        result = sample(
            args.video, args.group, args.out, args.start_s, args.end_s, args.min_gap_s,
            args.max_frames, args.quota, args.candidate_step, detector, args.jpeg_quality,
            args.overwrite,
        )
        result = {key: value for key, value in result.items() if key != "frames"}
    elif args.command == "export-tracks":
        result = export_tracks(
            args.tracks, args.out, args.group, args.video, args.start_frame, args.end_frame,
            args.all_roles, args.observed_only,
        )
    elif args.format == "yolo":
        result = import_cvat_yolo(args.zip, args.out, args.images, args.group, args.drop_label)
    else:
        result = import_cvat_mot(args.zip, args.out, args.label)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
