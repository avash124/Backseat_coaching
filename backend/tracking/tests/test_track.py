import numpy as np

from tracking.config import TrackerConfig
from tracking.track import PlayerTracker


def test_short_gap_keeps_id_but_expired_track_gets_new_id():
    tracker = PlayerTracker(TrackerConfig(max_age=3, min_hits=1), confidence=0.25)
    image = np.zeros((100, 100, 3), np.uint8)
    box = np.array([[20, 10, 40, 80, 0.9, 0]], np.float32)
    empty = np.empty((0, 6), np.float32)
    initial_id = tracker.update(box, image)[0].track_id
    tracker.update(box, image)
    assert tracker.update(empty, image) == []
    assert tracker.update(box, image)[0].track_id == initial_id
    for _ in range(5):
        tracker.update(empty, image)
    tracker.update(box, image)
    assert tracker.update(box, image)[0].track_id != initial_id


def test_each_video_starts_a_new_tracker():
    image = np.zeros((100, 100, 3), np.uint8)
    box = np.array([[20, 10, 40, 80, 0.9, 0]], np.float32)
    first = PlayerTracker(TrackerConfig(), confidence=0.25)
    assert first.update(box, image)[0].track_id == 0
    second = PlayerTracker(TrackerConfig(), confidence=0.25)
    assert second.update(box, image)[0].track_id == 0
