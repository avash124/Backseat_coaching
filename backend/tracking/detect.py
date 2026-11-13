"""Load YOLO once; detect every person class (player and referee) in batches.

Each frame yields an (N, 6 + K) float32 array:
``[x1, y1, x2, y2, confidence, class_id, score_0, ..., score_{K-1}]`` where the
K scores follow ``PersonDetector.class_ids`` (player first, then referee when the
checkpoint has one). The YOLO26 one-to-one head can report the same box once per
class; those rows are merged so every person appears once, with ``confidence``
the highest class score and ``class_id`` its class. A class score below the
detector floor is recorded as 0.
"""

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from tracking.config import DetectorConfig

PERSON_CLASSES = ("player", "referee")


class PersonDetector:
    def __init__(self, weights: Path, config: DetectorConfig):
        from ultralytics import YOLO

        weights = Path(weights)
        if not weights.is_file():
            raise FileNotFoundError(f"Trained weights not found: {weights}")
        self.model = YOLO(str(weights))
        self.config = config
        by_name = {name: index for index, name in self.model.names.items()}
        if "player" not in by_name or len(by_name) != len(self.model.names):
            raise ValueError(f"Expected one 'player' class; weights contain {self.model.names}")
        self.player_id = by_name["player"]
        self.class_ids = [by_name[name] for name in PERSON_CLASSES if name in by_name]
        self.class_names = {index: self.model.names[index] for index in self.class_ids}

    def detect(self, frame: np.ndarray) -> np.ndarray:
        return self.detect_batch([frame])[0]

    def detect_batch(self, frames: Sequence[np.ndarray]) -> list[np.ndarray]:
        if not frames:
            return []
        results = self.model.predict(
            source=list(frames),
            classes=self.class_ids,
            imgsz=self.config.imgsz,
            conf=self.config.confidence,
            max_det=self.config.max_detections,
            device=self.config.device,
            nms=False,
            verbose=False,
        )
        return [
            merge_class_rows(
                result.boxes.xyxy.cpu().numpy(),
                result.boxes.conf.cpu().numpy(),
                result.boxes.cls.cpu().numpy(),
                self.class_ids,
            )
            for result in results
        ]


def merge_class_rows(
    boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray, class_ids: Sequence[int]
) -> np.ndarray:
    """Collapse rows with identical boxes (one anchor, several classes) into one row."""
    boxes = np.asarray(boxes, np.float32).reshape(-1, 4)
    scores = np.asarray(scores, np.float32).reshape(-1)
    classes = np.asarray(classes).reshape(-1).astype(int)
    columns = {class_id: column for column, class_id in enumerate(class_ids)}
    width = 6 + len(class_ids)
    if not len(boxes):
        return np.empty((0, width), np.float32)
    unique, inverse = np.unique(boxes, axis=0, return_inverse=True)
    class_scores = np.zeros((len(unique), len(class_ids)), np.float32)
    for row, (group, class_id) in enumerate(zip(inverse.reshape(-1), classes)):
        column = columns[int(class_id)]
        class_scores[group, column] = max(class_scores[group, column], scores[row])
    best = class_scores.argmax(axis=1)
    merged = np.column_stack(
        (
            unique,
            class_scores.max(axis=1),
            np.asarray(class_ids, np.float32)[best],
            class_scores,
        )
    ).astype(np.float32)
    # Highest confidence first; ties keep a deterministic box order.
    return merged[np.argsort(-merged[:, 4], kind="stable")]


def box_iou(boxes: np.ndarray) -> np.ndarray:
    """Pairwise IoU of (N, 4) xyxy boxes."""
    boxes = np.asarray(boxes, np.float64).reshape(-1, 4)
    top_left = np.maximum(boxes[:, None, :2], boxes[None, :, :2])
    bottom_right = np.minimum(boxes[:, None, 2:], boxes[None, :, 2:])
    intersection = np.clip(bottom_right - top_left, 0, None).prod(axis=2)
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = areas[:, None] + areas[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(union), where=union > 0)


def merge_duplicates(
    rows: np.ndarray, class_scores: np.ndarray, iou_threshold: float, class_ids: Sequence[int]
) -> tuple[np.ndarray, np.ndarray]:
    """Class-agnostic NMS that keeps the evidence of the suppressed rows.

    The NMS-free head can place a player box and a referee box (or two player
    boxes) on one person from different anchors. Rows overlapping a stronger row
    by at least ``iou_threshold`` are folded into it: its box is kept, and each
    class score becomes the maximum over the group. ``confidence``/``class_id``
    are then the best merged class score and its class.
    """
    rows = np.asarray(rows, np.float32).reshape(-1, 6)
    class_scores = np.asarray(class_scores, np.float32).reshape(len(rows), len(class_ids))
    if len(rows) < 2 or iou_threshold > 1:
        return rows, class_scores
    order = np.argsort(-rows[:, 4], kind="stable")
    rows, class_scores = rows[order], class_scores[order]
    overlaps = box_iou(rows[:, :4]) >= iou_threshold
    absorbed = np.zeros(len(rows), bool)
    keep = []
    for index in range(len(rows)):
        if absorbed[index]:
            continue
        group = overlaps[index] & ~absorbed
        group[:index] = False
        absorbed |= group
        keep.append((index, class_scores[group].max(axis=0)))
    kept_rows = rows[[index for index, _ in keep]].copy()
    kept_scores = np.asarray([scores for _, scores in keep], np.float32)
    kept_rows[:, 4] = kept_scores.max(axis=1)
    kept_rows[:, 5] = np.asarray(class_ids, np.float32)[kept_scores.argmax(axis=1)]
    return kept_rows, kept_scores


# Backwards-compatible name used by earlier callers.
PlayerDetector = PersonDetector
