import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from tracking import pipeline
from tracking.config import DecodeConfig, PostprocessConfig, ShotConfig, TrackerConfig
from tracking.schemas import TracksArtifact
from tracking.shots import ShotDetector
from tracking.tests.test_pipeline import FakeDetector, write_video

CLIP = Path(__file__).resolve().parents[1] / "runs" / "inputs" / "basketball-first30s.mp4"


def scene(index: int, shot: int, size=(96, 128)) -> np.ndarray:
    """A textured 'court' with a moving player; shot 1 is mirrored and recolored."""
    height, width = size
    y, x = np.mgrid[:height, :width]
    image = np.stack(
        ((60 + x % 40).astype(np.uint8), (90 + y % 30).astype(np.uint8), np.full_like(x, 150)),
        axis=-1,
    ).astype(np.uint8)
    cv2.rectangle(image, (10 + index, 30), (25 + index, 70), (255, 255, 255), -1)
    if shot:
        image = cv2.flip(image, 1)[:, :, ::-1].copy()
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv[..., 0] = (hsv[..., 0].astype(int) + 60) % 180
        image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return image


def run_detector(config: ShotConfig, frames, fps=30):
    detector = ShotDetector(config)
    cuts = [detector.update(i, i / fps, image) for i, image in enumerate(frames)]
    return [cut.frame_index for cut in cuts if cut is not None]


def test_cut_detected_once_at_the_shot_boundary():
    frames = [scene(i, 0) for i in range(40)] + [scene(i, 1) for i in range(40)]
    assert run_detector(ShotConfig(), frames) == [40]


def test_min_shot_length_and_disable_suppress_cuts():
    frames = [scene(i, 0) for i in range(10)] + [scene(i, 1) for i in range(40)]
    assert run_detector(ShotConfig(min_shot_seconds=1.0), frames) == []
    assert run_detector(ShotConfig(min_shot_seconds=0.2), frames) == [10]
    assert run_detector(ShotConfig(enabled=False), frames + frames) == []


@pytest.mark.skipif(not CLIP.is_file(), reason="Demo clip not available")
def test_real_clip_concatenated_with_flipped_recolored_copy():
    from tracking.decode import VideoReader

    with VideoReader(CLIP, DecodeConfig(max_frames=45)) as reader:
        images = [frame.image for frame in reader]
    recolored = [cv2.flip(image, 1)[:, :, ::-1].copy() for image in images]
    assert run_detector(ShotConfig(), images + recolored) == [45]


def test_pipeline_resets_ids_at_a_cut_and_records_it(tmp_path, monkeypatch):
    frames = [scene(i, 0) for i in range(36)] + [scene(i, 1) for i in range(36)]
    timestamps = [i / 30 for i in range(len(frames))]
    video = tmp_path / "two-shots.mp4"
    write_video(video, frames, timestamps)
    weights = tmp_path / "weights.pt"
    weights.write_bytes(b"stub")

    class StaticDetector(FakeDetector):
        def detect(self, image):
            # The same box in both shots: without a reset it would keep its ID.
            return np.array([[40, 20, 60, 80, 0.9, 0, 0.9, 0.0]], np.float32)

    monkeypatch.setattr(pipeline, "PersonDetector", StaticDetector)
    output = tmp_path / "out"
    pipeline.run_pipeline(
        video,
        weights,
        output,
        tracker_config=TrackerConfig(min_hits=1),
        postprocess_config=PostprocessConfig(min_track_s=0),
    )
    artifact = TracksArtifact.model_validate_json((output / "tracks.json").read_text())
    assert [cut.frame_index for cut in artifact.shot_cuts] == [36]
    before = {t.track_id for frame in artifact.frames[:36] for t in frame.tracks}
    after = {t.track_id for frame in artifact.frames[36:] for t in frame.tracks}
    assert len(before) == len(after) == 1 and before.isdisjoint(after)
    # Stitching may not link the two shots even though the box did not move.
    assert json.loads((output / "postprocess.json").read_text())["stitches"] == []
    stats = json.loads((output / "track_stats.json").read_text())
    assert [cut["frame_index"] for cut in stats["shot_cuts"]] == [36]
    disabled = pipeline.run_pipeline(
        video,
        weights,
        output,
        tracker_config=TrackerConfig(min_hits=1),
        shot_config=ShotConfig(enabled=False),
        postprocess_config=PostprocessConfig(min_track_s=0),
    )
    assert disabled["reused_detection"] and not disabled["reused_tracking"]
    artifact = TracksArtifact.model_validate_json((output / "tracks.json").read_text())
    assert artifact.shot_cuts == []
    assert len({t.track_id for frame in artifact.frames for t in frame.tracks}) == 1
