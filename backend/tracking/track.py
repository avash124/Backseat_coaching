"""Stateful BoxMOT adapter. Create one instance per video; call ``reset`` at shot cuts.

All person classes are associated together (class-agnostic), so a box whose
per-frame class flips between player and referee keeps its ID; roles are decided
per track after tracking (see postprocess.py).

Supported BoxMOT 25.0.0 trackers, built through their public constructors:
ocsort (motion only), deepocsort, botsort, strongsort, hybridsort, boosttrack.
Appearance trackers receive one shared ReID backend, loaded once per run.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tracking.config import REID_WEIGHTS_DIR, TrackerConfig

TRACKER_KINDS = ("ocsort", "deepocsort", "botsort", "strongsort", "hybridsort", "boosttrack")
# Trackers whose appearance branch can be switched off keep it on only when
# TrackerConfig.appearance is true; StrongSORT cannot run without embeddings.
APPEARANCE_OPTIONAL = frozenset({"deepocsort", "botsort", "hybridsort", "boosttrack"})


def uses_reid(config: TrackerConfig) -> bool:
    return config.kind == "strongsort" or (config.kind in APPEARANCE_OPTIONAL and config.appearance)


def reid_weights_path(config: TrackerConfig, directory: Path = REID_WEIGHTS_DIR) -> Path:
    return Path(directory) / config.reid_weights


def ensure_reid_weights(path: Path) -> Path:
    """Return an existing ReID checkpoint, downloading BoxMOT catalog weights if missing."""
    path = Path(path)
    if path.is_file():
        return path
    from boxmot.reid.core.catalog import TRAINED_URLS
    from boxmot.resources.download import download_file

    if path.name not in TRAINED_URLS:
        raise FileNotFoundError(f"ReID weights not found and not in the BoxMOT catalog: {path}")
    try:
        download_file(TRAINED_URLS[path.name], path)
    except Exception as error:  # Network/certificate failures: explain the manual fix.
        raise FileNotFoundError(
            f"Could not download {path.name} ({error}). Download {TRAINED_URLS[path.name]} "
            f"manually to {path}."
        ) from error
    return path


def load_reid_model(weights: Path, device: str = "cpu"):
    """Load a BoxMOT ReID backend exposing get_features(boxes_xyxy, bgr_image)."""
    from boxmot.reid.core import ReID

    return ReID(weights=Path(weights), device=device, half=False).model


def build_boxmot_tracker(config: TrackerConfig, confidence: float, reid_model=None):
    """Map the shared settings onto one BoxMOT tracker; other options keep BoxMOT defaults."""
    import boxmot

    if uses_reid(config) and reid_model is None:
        raise ValueError(f"{config.kind} with appearance needs a ReID model")
    appearance = {"reid_model": reid_model} if uses_reid(config) else {}
    common = {
        "max_age": config.max_age,
        "max_obs": max(50, config.max_age + 1),
        # Higher BoxMOT min_hits also suppresses already-confirmed tracks after
        # gaps. Initial confirmation is applied in PlayerTracker instead.
        "min_hits": 1,
        "per_class": False,
    }
    if config.kind == "ocsort":
        return boxmot.OcSort(
            det_thresh=config.track_threshold,
            min_conf=confidence,
            use_byte=config.use_byte,
            iou_threshold=config.iou_threshold,
            **common,
        )
    if config.kind == "deepocsort":
        return boxmot.DeepOcSort(
            det_thresh=config.track_threshold,
            iou_threshold=config.iou_threshold,
            use_embeddings=uses_reid(config),
            **appearance,
            **common,
        )
    if config.kind == "botsort":
        # BoT-SORT keeps lost tracks for track_buffer frames (at frame_rate=30).
        return boxmot.BotSort(
            track_high_thresh=config.track_threshold,
            track_low_thresh=confidence,
            new_track_thresh=config.track_threshold,
            track_buffer=config.max_age,
            frame_rate=30,
            use_embeddings=uses_reid(config),
            det_thresh=config.track_threshold,
            **appearance,
            **common,
        )
    if config.kind == "strongsort":
        # StrongSORT has no low-score pass; weak boxes would otherwise start tracks.
        return boxmot.StrongSort(
            min_conf=config.track_threshold,
            max_iou_dist=1 - config.iou_threshold,
            n_init=1,
            **appearance,
            **common,
        )
    if config.kind == "hybridsort":
        return boxmot.HybridSort(
            det_thresh=config.track_threshold,
            low_thresh=confidence,
            use_byte=config.use_byte,
            iou_threshold=config.iou_threshold,
            use_embeddings=uses_reid(config),
            **appearance,
            **common,
        )
    if config.kind == "boosttrack":
        return boxmot.BoostTrack(
            det_thresh=config.track_threshold,
            iou_threshold=config.iou_threshold,
            use_embeddings=uses_reid(config),
            **appearance,
            **common,
        )
    raise ValueError(f"Unsupported tracker kind: {config.kind}")


APPEARANCE_DESCRIPTOR = "hsv-8x3x3-torso-v1"
APPEARANCE_BOXES = 10


def torso_histogram(image: np.ndarray, box) -> np.ndarray | None:
    """Normalized 8x3x3 HSV histogram of the jersey area (15-85% width, 10-60% height)."""
    import cv2

    x1, y1, x2, y2 = (float(v) for v in box)
    width, height = x2 - x1, y2 - y1
    top, bottom = int(max(y1 + 0.1 * height, 0)), int(y1 + 0.6 * height)
    left, right = int(max(x1 + 0.15 * width, 0)), int(x2 - 0.15 * width)
    crop = image[top:bottom, left:right]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    histogram = cv2.calcHist([hsv], [0, 1, 2], None, [8, 3, 3], [0, 180, 0, 256, 0, 256])
    histogram = histogram.reshape(-1)
    return histogram / max(float(histogram.sum()), 1.0)


class AppearanceSummary:
    """Mean torso histogram of each track's first and last boxes, for offline stitching."""

    def __init__(self, boxes: int = APPEARANCE_BOXES):
        self.boxes = boxes
        self.first: dict[int, list[np.ndarray]] = {}
        self.last: dict[int, list[np.ndarray]] = {}

    def add(self, image: np.ndarray, track_id: int, box):
        histogram = torso_histogram(image, box)
        if histogram is None:
            return
        first = self.first.setdefault(track_id, [])
        if len(first) < self.boxes:
            first.append(histogram)
        last = self.last.setdefault(track_id, [])
        last.append(histogram)
        del last[: -self.boxes]

    def to_json(self) -> dict:
        def mean(histograms):
            return [round(float(v), 5) for v in np.mean(histograms, axis=0)]

        return {
            "descriptor": APPEARANCE_DESCRIPTOR,
            "boxes": self.boxes,
            "tracks": {
                str(track_id): {"start": mean(self.first[track_id]), "end": mean(self.last[track_id])}
                for track_id in sorted(self.first)
            },
        }


@dataclass(frozen=True)
class TrackedDetection:
    """One detector box associated with a track in the current frame."""

    track_id: int
    bbox_xyxy: tuple[float, float, float, float]
    confidence: float
    class_id: int
    # Row of the detections array passed to update().
    detection_index: int
    # False until the track has min_hits consecutive observations.
    confirmed: bool
    observed: bool = True


class PlayerTracker:
    """Use weak boxes for recovery and confirm each track only once in its lifetime.

    ``update`` returns confirmed tracks; observations of still-tentative tracks
    are kept in ``tentative`` for that frame so they can be restored later if
    the track is confirmed. IDs stay unique across ``reset`` calls.
    """

    def __init__(self, config: TrackerConfig, confidence: float, reid_model=None):
        if confidence > config.track_threshold:
            raise ValueError("Detection confidence must be <= track_threshold")
        self.config = config
        self.confidence = confidence
        self.reid_model = reid_model
        self.id_offset = 0
        self.next_id = 0
        self.tentative: list[TrackedDetection] = []
        self._start_sequence()

    def _start_sequence(self):
        self.tracker = build_boxmot_tracker(self.config, self.confidence, self.reid_model)
        self.frame_index = 0
        self.hits: dict[int, int] = {}
        self.last_seen: dict[int, int] = {}

    def reset(self):
        """Start a new shot: forget all tracks; later IDs never reuse earlier ones."""
        self.id_offset = self.next_id
        self._start_sequence()

    def update(self, detections: np.ndarray, frame: np.ndarray) -> list[TrackedDetection]:
        detections = np.asarray(detections, dtype=np.float32)
        if detections.ndim != 2 or detections.shape[1] != 6:
            raise ValueError("Detections must have shape (N, 6), including empty (0, 6)")
        self.frame_index += 1
        self.tentative = []
        # Empty frames must still age unmatched tracks.
        rows = self.tracker.update(detections, frame)
        height, width = frame.shape[:2]
        tracks = []
        for row in rows:
            detection_index = int(row[7])
            # Report the detector's box and score; trackers may output filtered states.
            source = (
                detections[detection_index]
                if 0 <= detection_index < len(detections)
                else np.asarray(row[[0, 1, 2, 3, 5, 6]], np.float32)
            )
            box = source[:4].copy()
            box[[0, 2]] = np.clip(box[[0, 2]], 0, width)
            box[[1, 3]] = np.clip(box[[1, 3]], 0, height)
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            track_id = int(row[4]) + self.id_offset
            self.next_id = max(self.next_id, track_id + 1)
            previous = self.last_seen.get(track_id, self.frame_index - 1)
            hits = self.hits.get(track_id, 0)
            # Consecutive observations are required only while tentative.
            if hits < self.config.min_hits and previous != self.frame_index - 1:
                hits = 0
            self.hits[track_id] = min(hits + 1, self.config.min_hits)
            self.last_seen[track_id] = self.frame_index
            confirmed = self.hits[track_id] >= self.config.min_hits
            item = TrackedDetection(
                track_id=track_id,
                bbox_xyxy=tuple(float(value) for value in box),
                confidence=float(np.clip(source[4], 0, 1)),
                class_id=int(source[5]),
                detection_index=detection_index,
                confirmed=confirmed,
            )
            (tracks if confirmed else self.tentative).append(item)
        for track_id, last_seen in list(self.last_seen.items()):
            if self.frame_index - last_seen > self.config.max_age:
                self.last_seen.pop(track_id)
                self.hits.pop(track_id)
        return tracks
