"""Stateful BoxMOT OC-SORT adapter. Create one instance per continuous sequence."""

import numpy as np

from tracking.config import TrackerConfig
from tracking.schemas import PlayerTrack


class PlayerTracker:
    def __init__(self, config: TrackerConfig, confidence: float):
        from boxmot import OcSort

        self.tracker = OcSort(
            det_thresh=confidence,
            max_age=config.max_age,
            max_obs=max(50, config.max_age + 1),
            min_hits=config.min_hits,
            iou_threshold=config.iou_threshold,
            per_class=False,
        )

    def update(self, detections: np.ndarray, frame: np.ndarray) -> list[PlayerTrack]:
        detections = np.asarray(detections, dtype=np.float32)
        if detections.ndim != 2 or detections.shape[1] != 6:
            raise ValueError("Detections must have shape (N, 6), including empty (0, 6)")
        # Empty updates still advance the tracker and age missing tracks.
        rows = self.tracker.update(detections, frame)
        height, width = frame.shape[:2]
        tracks = []
        for row in rows:
            box = row[:4].copy()
            box[[0, 2]] = np.clip(box[[0, 2]], 0, width)
            box[[1, 3]] = np.clip(box[[1, 3]], 0, height)
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            tracks.append(
                PlayerTrack(
                    track_id=int(row[4]),
                    bbox_xyxy=tuple(box),
                    confidence=float(row[5]),
                    class_id=int(row[6]),
                    observed=int(row[7]) >= 0,
                )
            )
        return tracks
