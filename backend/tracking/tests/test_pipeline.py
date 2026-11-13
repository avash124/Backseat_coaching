import json
from fractions import Fraction
from types import MappingProxyType

import av
import numpy as np
import pytest

from tracking import pipeline
from tracking.config import (
    DEFAULT_WEIGHTS,
    DecodeConfig,
    DetectorConfig,
    PostprocessConfig,
    ShotConfig,
    TrackerConfig,
)
from tracking.court import CourtConfig
from tracking.schemas import RawTracksArtifact, TracksArtifact

NO_CLEANUP = PostprocessConfig(min_track_s=0, stitch_max_gap_s=0, interpolate_max_gap_s=0)


def write_video(path, images, timestamps):
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=30)
        stream.height, stream.width = images[0].shape[:2]
        stream.pix_fmt = "yuv420p"
        stream.codec_context.time_base = Fraction(1, 90000)
        for image, time in zip(images, timestamps):
            frame = av.VideoFrame.from_ndarray(image, format="bgr24")
            frame.pts = round(time * 90000)
            frame.time_base = Fraction(1, 90000)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


class FakeDetector:
    """Replaces YOLO: one moving player box per frame, none in frame 3."""

    starts = 0
    player_id = 0
    class_ids = (0, 1)
    class_names = MappingProxyType({0: "player", 1: "referee"})

    def __init__(self, *_):
        FakeDetector.starts += 1
        self.index = -1

    def detect_batch(self, images):
        return [self.detect(image) for image in images]

    def detect(self, image):
        self.index += 1
        if self.index == 3:
            return np.empty((0, 8), np.float32)
        x = 10 + self.index
        return np.array([[x, 10, x + 20, 50, 0.9, 0, 0.9, 0.2]], np.float32)


@pytest.fixture
def sample_run(tmp_path, monkeypatch):
    video = tmp_path / "input.mp4"
    timestamps = [0, 0.033, 0.067, 0.11, 0.15, 0.18, 0.22, 0.26]
    write_video(video, [np.zeros((64, 64, 3), np.uint8)] * len(timestamps), timestamps)
    weights = tmp_path / "weights.pt"
    weights.write_bytes(b"stub checkpoint; detector is replaced in this test")
    FakeDetector.starts = 0
    monkeypatch.setattr(pipeline, "PersonDetector", FakeDetector)
    return video, weights, tmp_path / "output", timestamps, FakeDetector


def test_pipeline_preserves_timestamps_and_writes_playable_preview(sample_run):
    video, weights, output, timestamps, _ = sample_run
    result = pipeline.run_pipeline(
        video,
        weights,
        output,
        tracker_config=TrackerConfig(min_hits=1),
        postprocess_config=NO_CLEANUP,
    )
    artifact = TracksArtifact.model_validate_json((output / "tracks.json").read_text())
    assert artifact.schema_version == "1.1"
    assert result["frames"] == 8
    assert result["detections"] == 7
    assert [frame.timestamp_s for frame in artifact.frames] == pytest.approx(timestamps, abs=1e-5)
    assert {track.track_id for frame in artifact.frames for track in frame.tracks} == {0}
    assert artifact.frames[3].tracks == []
    assert artifact.class_names == {0: "player", 1: "referee"}
    assert [info.role for info in artifact.track_info] == ["player"]
    with av.open(str(output / "annotated.mp4")) as preview:
        assert preview.streams.video[0].codec_context.name == "h264"
        assert [frame.time for frame in preview.decode(video=0)] == pytest.approx(
            timestamps, abs=1e-5
        )


def test_class_scores_reach_raw_tracks_and_gap_is_interpolated(sample_run):
    video, weights, output, _, _ = sample_run
    pipeline.run_pipeline(
        video,
        weights,
        output,
        tracker_config=TrackerConfig(min_hits=1),
        postprocess_config=PostprocessConfig(min_track_s=0),
    )
    detections = [json.loads(line) for line in (output / "detections.jsonl").open()]
    assert detections[0]["class_scores"] == [pytest.approx([0.9, 0.2])]
    raw = RawTracksArtifact.model_validate_json((output / "tracks_raw.json").read_text())
    assert raw.frames[0].tracks[0].class_scores == pytest.approx((0.9, 0.2))
    assert raw.frames[3].tracks == []
    artifact = TracksArtifact.model_validate_json((output / "tracks.json").read_text())
    filled = artifact.frames[3].tracks[0]
    assert not filled.observed and filled.confidence == 0
    expected_x = 12 + 2 * (0.11 - 0.067) / (0.15 - 0.067)
    assert filled.bbox_xyxy[0] == pytest.approx(expected_x, abs=1e-3)
    summary = json.loads((output / "tracking_summary.json").read_text())
    assert summary["proxy_metrics_raw"]["intra_track_gaps"] == 1
    assert summary["proxy_metrics"]["intra_track_gaps"] == 0
    assert summary["proxy_metrics"]["observed_intra_track_gaps"] == 1


def test_cache_reuse_and_tracker_only_rerun(sample_run):
    video, weights, output, _, detector = sample_run
    pipeline.run_pipeline(video, weights, output)
    second = pipeline.run_pipeline(video, weights, output)
    assert second["reused_detection"] and second["reused_tracking"]
    assert second["reused_postprocess"]
    third = pipeline.run_pipeline(video, weights, output, tracker_config=TrackerConfig(max_age=15))
    assert third["reused_detection"] and not third["reused_tracking"]
    assert not third["reused_postprocess"]
    shots = pipeline.run_pipeline(
        video,
        weights,
        output,
        tracker_config=TrackerConfig(max_age=15),
        shot_config=ShotConfig(threshold=0.9),
    )
    assert shots["reused_detection"] and not shots["reused_tracking"]
    assert detector.starts == 1


def test_postprocess_change_reuses_detection_and_tracking(sample_run):
    video, weights, output, _, detector = sample_run
    pipeline.run_pipeline(video, weights, output, postprocess_config=NO_CLEANUP)
    artifact = TracksArtifact.model_validate_json((output / "tracks.json").read_text())
    assert any(frame.tracks for frame in artifact.frames)
    raw = (output / "tracks_raw.json").read_bytes()
    result = pipeline.run_pipeline(
        video, weights, output, postprocess_config=PostprocessConfig(min_track_s=5)
    )
    assert result["reused_detection"] and result["reused_tracking"]
    assert not result["reused_postprocess"]
    assert (output / "tracks_raw.json").read_bytes() == raw
    artifact = TracksArtifact.model_validate_json((output / "tracks.json").read_text())
    assert all(not frame.tracks for frame in artifact.frames)
    assert detector.starts == 1
    no_preview = pipeline.run_pipeline(video, weights, output, preview=False)
    assert no_preview["reused_tracking"] and not no_preview["reused_postprocess"]
    assert not (output / "annotated.mp4").exists()


def test_batch_size_changes_detection_key(sample_run):
    video, weights, output, _, detector = sample_run
    pipeline.run_pipeline(video, weights, output, DetectorConfig(batch_size=3))
    repeat = pipeline.run_pipeline(video, weights, output, DetectorConfig(batch_size=3))
    assert repeat["reused_detection"]
    pipeline.run_pipeline(video, weights, output, DetectorConfig(batch_size=1))
    assert detector.starts == 2
    assert len((output / "detections.jsonl").read_text().splitlines()) == 8


def test_corrupt_artifacts_are_rebuilt(sample_run):
    video, weights, output, _, detector = sample_run
    pipeline.run_pipeline(video, weights, output)
    (output / "tracks.json").write_text("incomplete")
    result = pipeline.run_pipeline(video, weights, output)
    assert result["reused_detection"] and result["reused_tracking"]
    assert not result["reused_postprocess"]
    TracksArtifact.model_validate_json((output / "tracks.json").read_text())
    (output / "tracks_raw.json").write_text("incomplete")
    result = pipeline.run_pipeline(video, weights, output)
    assert result["reused_detection"] and not result["reused_tracking"]
    (output / "detections.jsonl").write_text("incomplete")
    result = pipeline.run_pipeline(video, weights, output)
    assert not result["reused_detection"]
    assert detector.starts == 2


def test_sampling_and_duration_limit_use_video_time(sample_run):
    video, weights, output, _, _ = sample_run
    pipeline.run_pipeline(
        video, weights, output, decode_config=DecodeConfig(stride=2, max_seconds=0.2)
    )
    payload = json.loads((output / "tracks.json").read_text())
    assert [frame["frame_index"] for frame in payload["frames"]] == [0, 2, 4]
    assert [frame["timestamp_s"] for frame in payload["frames"]] == pytest.approx([0, 0.067, 0.15])


def test_input_cannot_be_overwritten_by_output(sample_run):
    video, weights, _, _, _ = sample_run
    for name in ("annotated.mp4", "tracks_raw.json"):
        video = video.rename(video.with_name(name))
        with pytest.raises(ValueError, match="overwrite"):
            pipeline.run_pipeline(video, weights, video.parent)


def test_court_change_reuses_detections_and_removes_off_court_tracks(sample_run):
    video, weights, output, _, detector = sample_run
    pipeline.run_pipeline(video, weights, output)
    # Fake detector's feet are y=50 in a 64px image. Exclude that area.
    court = CourtConfig(polygon=((0, 0), (1, 0), (1, 0.5), (0, 0.5)), margin=0)
    result = pipeline.run_pipeline(video, weights, output, court_config=court)
    assert result["reused_detection"] and not result["reused_tracking"]
    assert detector.starts == 1
    artifact = TracksArtifact.model_validate_json((output / "tracks.json").read_text())
    assert all(not frame.tracks for frame in artifact.frames)
    summary = json.loads((output / "tracking_summary.json").read_text())
    assert summary["outside_court_detections"] == 7
    assert summary["unique_track_ids"] == 0
    repeat = pipeline.run_pipeline(video, weights, output, court_config=court)
    assert repeat["reused_tracking"]


@pytest.mark.skipif(not DEFAULT_WEIGHTS.is_file(), reason="Trained checkpoint not available")
def test_batched_detection_matches_single_frames():
    from tracking.detect import PersonDetector

    rng = np.random.default_rng(0)
    clip = DEFAULT_WEIGHTS.parent.parent / "runs" / "inputs" / "basketball-first30s.mp4"
    if clip.is_file():
        from tracking.decode import VideoReader

        with VideoReader(clip, DecodeConfig(max_frames=3)) as reader:
            images = [frame.image for frame in reader]
    else:
        images = [rng.integers(0, 255, (360, 640, 3), dtype=np.uint8) for _ in range(3)]
    detector = PersonDetector(DEFAULT_WEIGHTS, DetectorConfig(imgsz=640, batch_size=3))
    batched = detector.detect_batch(images)
    for image, rows in zip(images, batched):
        single = detector.detect(image)
        assert single.shape == rows.shape and single.shape[1] == 6 + len(detector.class_ids)
        np.testing.assert_array_equal(single[:, 5], rows[:, 5])
        # Batched CPU convolutions differ from single-image ones by float rounding only.
        np.testing.assert_allclose(single, rows, atol=1e-2)


def test_duplicate_class_rows_merge_into_one_person():
    from tracking.detect import merge_class_rows

    boxes = np.array([[0, 0, 10, 20], [5, 5, 15, 30], [0, 0, 10, 20]], np.float32)
    merged = merge_class_rows(boxes, np.array([0.3, 0.8, 0.6]), np.array([0, 0, 1]), [0, 1])
    np.testing.assert_allclose(
        merged, [[5, 5, 15, 30, 0.8, 0, 0.8, 0], [0, 0, 10, 20, 0.6, 1, 0.3, 0.6]], atol=1e-6
    )
    assert merge_class_rows(np.empty((0, 4)), [], [], [0, 1]).shape == (0, 8)


def test_reid_checkpoint_hash_is_part_of_the_tracking_key(sample_run, monkeypatch):
    from tracking.tests.test_track import FakeReID

    video, weights, output, _, detector = sample_run
    loaded = []
    monkeypatch.setattr(
        pipeline, "load_reid_model", lambda path, device: loaded.append(path) or FakeReID()
    )
    checkpoint = video.parent / "reid" / "osnet_x0_25_msmt17.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"first checkpoint")
    config = TrackerConfig(kind="deepocsort", min_hits=1)
    first = pipeline.run_pipeline(
        video, weights, output, tracker_config=config, reid_weights=checkpoint
    )
    assert first["provenance"]["reid_weights_sha256"] and len(loaded) == 1
    repeat = pipeline.run_pipeline(
        video, weights, output, tracker_config=config, reid_weights=checkpoint
    )
    assert repeat["reused_tracking"] and len(loaded) == 1
    checkpoint.write_bytes(b"retrained checkpoint")
    changed = pipeline.run_pipeline(
        video, weights, output, tracker_config=config, reid_weights=checkpoint
    )
    assert changed["reused_detection"] and not changed["reused_tracking"]
    assert len(loaded) == 2 and detector.starts == 1
    with pytest.raises(ValueError, match="name"):
        pipeline.run_pipeline(
            video, weights, output, tracker_config=config, reid_weights=weights
        )
