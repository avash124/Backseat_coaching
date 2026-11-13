import numpy as np
import pytest

from tracking.config import TrackerConfig
from tracking.track import TRACKER_KINDS, PlayerTracker, uses_reid


class FakeReID:
    """Stands in for an OSNet backend: embeddings depend only on box position."""

    def __init__(self):
        self.calls = 0

    def get_features(self, boxes, image):
        self.calls += 1
        boxes = np.asarray(boxes, np.float32)
        return np.column_stack((np.ones(len(boxes)), boxes[:, 0] / 100)).astype(np.float32)


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
    first = PlayerTracker(TrackerConfig(min_hits=1), confidence=0.25)
    assert first.update(box, image)[0].track_id == 0
    second = PlayerTracker(TrackerConfig(min_hits=1), confidence=0.25)
    assert second.update(box, image)[0].track_id == 0


def test_weak_detection_recovers_existing_track_but_cannot_start_one():
    tracker = PlayerTracker(TrackerConfig(min_hits=1, track_threshold=0.5), confidence=0.1)
    image = np.zeros((100, 100, 3), np.uint8)
    strong = np.array([[20, 10, 40, 80, 0.9, 0]], np.float32)
    weak = np.array([[20, 10, 40, 80, 0.15, 0]], np.float32)
    assert tracker.update(weak, image) == []
    tracker.update(strong, image)
    initial_id = tracker.update(strong, image)[0].track_id
    recovered = tracker.update(weak, image)
    assert recovered[0].track_id == initial_id
    assert recovered[0].confidence == pytest.approx(0.15)
    assert recovered[0].observed
    # An unrelated weak box must not create another player.
    unrelated = np.array([[70, 10, 90, 80, 0.2, 0]], np.float32)
    assert tracker.update(unrelated, image) == []


def test_confirmed_track_returns_immediately_after_missed_frame():
    tracker = PlayerTracker(TrackerConfig(min_hits=3), confidence=0.1)
    image = np.zeros((100, 100, 3), np.uint8)
    box = np.array([[20, 10, 40, 80, 0.9, 0]], np.float32)
    empty = np.empty((0, 6), np.float32)
    assert tracker.update(box, image) == []
    assert tracker.update(box, image) == []
    initial_id = tracker.update(box, image)[0].track_id
    assert tracker.update(empty, image) == []
    assert tracker.update(box, image)[0].track_id == initial_id


def test_intermittent_tentative_track_is_not_confirmed():
    tracker = PlayerTracker(TrackerConfig(min_hits=3), confidence=0.1)
    image = np.zeros((100, 100, 3), np.uint8)
    box = np.array([[20, 10, 40, 80, 0.9, 0]], np.float32)
    empty = np.empty((0, 6), np.float32)
    for _ in range(5):
        assert tracker.update(box, image) == []
        assert tracker.update(empty, image) == []


def test_detection_floor_cannot_exceed_track_threshold():
    with pytest.raises(ValueError, match="track_threshold"):
        PlayerTracker(TrackerConfig(track_threshold=0.3), confidence=0.5)


def test_tentative_observations_are_recorded_until_confirmation():
    tracker = PlayerTracker(TrackerConfig(min_hits=3), confidence=0.1)
    image = np.zeros((100, 100, 3), np.uint8)
    box = np.array([[20, 10, 40, 80, 0.9, 0]], np.float32)
    assert tracker.update(box, image) == []
    assert [(t.track_id, t.confirmed, t.detection_index) for t in tracker.tentative] == [
        (0, False, 0)
    ]
    assert tracker.update(box, image) == []
    assert len(tracker.tentative) == 1
    confirmed = tracker.update(box, image)
    assert confirmed[0].confirmed and tracker.tentative == []


def test_class_agnostic_association_keeps_id_when_class_flips():
    tracker = PlayerTracker(TrackerConfig(min_hits=1), confidence=0.1)
    image = np.zeros((100, 100, 3), np.uint8)
    ids = set()
    for frame in range(6):
        class_id = frame % 2  # player, referee, player, ...
        detections = np.array([[20 + frame, 10, 40 + frame, 80, 0.8, class_id]], np.float32)
        tracked = tracker.update(detections, image)
        assert tracked[0].class_id == class_id
        ids.add(tracked[0].track_id)
    assert ids == {0}


def test_reset_starts_new_ids_that_never_collide():
    tracker = PlayerTracker(TrackerConfig(min_hits=1), confidence=0.1)
    image = np.zeros((100, 100, 3), np.uint8)
    boxes = np.array([[20, 10, 40, 80, 0.9, 0], [60, 10, 80, 80, 0.9, 0]], np.float32)
    before = {t.track_id for t in tracker.update(boxes, image)}
    tracker.reset()
    after = {t.track_id for t in tracker.update(boxes, image)}
    assert before == {0, 1} and after == {2, 3}


def test_reported_box_is_the_detection_box():
    tracker = PlayerTracker(TrackerConfig(min_hits=1), confidence=0.1)
    image = np.zeros((100, 100, 3), np.uint8)
    for frame in range(4):
        detections = np.array(
            [[70, 0, 90, 60, 0.3, 1], [20 + 2 * frame, 10, 40 + 2 * frame, 80, 0.9, 0]],
            np.float32,
        )
        for item in tracker.update(detections, image):
            row = detections[item.detection_index]
            assert item.bbox_xyxy == pytest.approx(tuple(row[:4]))
            assert item.confidence == pytest.approx(row[4])


@pytest.mark.parametrize("kind", TRACKER_KINDS)
def test_every_supported_boxmot_tracker_runs_with_shared_reid_model(kind):
    reid = FakeReID()
    config = TrackerConfig(kind=kind, min_hits=2)
    tracker = PlayerTracker(config, confidence=0.1, reid_model=reid)
    image = np.zeros((200, 300, 3), np.uint8)
    seen = []
    for frame in range(8):
        detections = np.array(
            [[20 + frame, 10, 40 + frame, 80, 0.9, 0], [150 + frame, 30, 175 + frame, 110, 0.8, 1]],
            np.float32,
        )
        seen.append({t.track_id for t in tracker.update(detections, image)})
    assert seen[-1] == seen[-2] and len(seen[-1]) == 2
    assert (reid.calls > 0) == uses_reid(config)


def test_appearance_tracker_requires_reid_model():
    with pytest.raises(ValueError, match="ReID"):
        PlayerTracker(TrackerConfig(kind="deepocsort"), confidence=0.1)
    # Motion-only variants do not need one.
    PlayerTracker(TrackerConfig(kind="botsort", appearance=False), confidence=0.1)
