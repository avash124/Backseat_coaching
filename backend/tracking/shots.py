"""Cheap online hard-cut detection for restarting the tracker between shots.

Each frame is downscaled, converted to HSV, and summarized by a normalized
16x4x4 histogram. A cut starts at a frame whose Bhattacharyya distance to the
previous processed frame reaches the threshold, unless the current shot is still
shorter than ``min_shot_seconds``. Gradual transitions (dissolves, fades) and
pans/zooms within a shot are not detected as cuts; court filtering still assumes
the boundary is valid for the whole video.
"""

import cv2
import numpy as np

from tracking.config import ShotConfig
from tracking.schemas import ShotCut

BINS = (16, 4, 4)
RANGES = (0, 180, 0, 256, 0, 256)


def frame_histogram(image: np.ndarray, width: int) -> np.ndarray:
    height = max(1, round(image.shape[0] * width / image.shape[1]))
    small = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    histogram = cv2.calcHist([hsv], [0, 1, 2], None, list(BINS), list(RANGES))
    return (histogram / max(float(histogram.sum()), 1.0)).astype(np.float32)


def histogram_distance(first: np.ndarray, second: np.ndarray) -> float:
    distance = cv2.compareHist(first, second, cv2.HISTCMP_BHATTACHARYYA)
    return float(min(max(distance, 0.0), 1.0))


class ShotDetector:
    """Feed every processed frame in order; ``update`` returns a cut or None."""

    def __init__(self, config: ShotConfig):
        self.config = config
        self.previous: np.ndarray | None = None
        self.shot_start_s = 0.0

    def update(self, frame_index: int, timestamp_s: float, image: np.ndarray) -> ShotCut | None:
        if not self.config.enabled:
            return None
        histogram = frame_histogram(image, self.config.width)
        previous, self.previous = self.previous, histogram
        if previous is None:
            self.shot_start_s = timestamp_s
            return None
        distance = histogram_distance(previous, histogram)
        if (
            distance < self.config.threshold
            or timestamp_s - self.shot_start_s < self.config.min_shot_seconds
        ):
            return None
        self.shot_start_s = timestamp_s
        return ShotCut(frame_index=frame_index, timestamp_s=timestamp_s, distance=distance)
