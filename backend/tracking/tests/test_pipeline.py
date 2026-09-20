import json
from fractions import Fraction

import av
import numpy as np
import pytest

from tracking import pipeline
from tracking.config import DecodeConfig, TrackerConfig
from tracking.schemas import TracksArtifact


@pytest.fixture
def sample_run(tmp_path, monkeypatch):
    video = tmp_path / "input.mp4"
    timestamps = [0, 0.033, 0.067, 0.11, 0.15, 0.18, 0.22, 0.26]
    with av.open(str(video), "w") as container:
        stream = container.add_stream("libx264", rate=30)
        stream.width = stream.height = 64
        stream.pix_fmt = "yuv420p"
        stream.codec_context.time_base = Fraction(1, 90000)
        for time in timestamps:
            frame = av.VideoFrame.from_ndarray(np.zeros((64, 64, 3), np.uint8), format="bgr24")
            frame.pts = round(time * 90000)
            frame.time_base = Fraction(1, 90000)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    weights = tmp_path / "weights.pt"
    weights.write_bytes(b"stub checkpoint; detector is replaced in this test")

    class FakeDetector:
        starts = 0
        player_id = 0

        def __init__(self, *_):
            FakeDetector.starts += 1
            self.index = -1

        def detect(self, image):
            self.index += 1
            if self.index == 3:
                return np.empty((0, 6), np.float32)
            return np.array([[10 + self.index, 10, 30 + self.index, 50, 0.9, 0]], np.float32)

    monkeypatch.setattr(pipeline, "PlayerDetector", FakeDetector)
    return video, weights, tmp_path / "output", timestamps, FakeDetector


def test_pipeline_preserves_timestamps_and_writes_playable_preview(sample_run):
    video, weights, output, timestamps, _ = sample_run
    result = pipeline.run_pipeline(video, weights, output, tracker_config=TrackerConfig(min_hits=1))
    artifact = TracksArtifact.model_validate_json((output / "tracks.json").read_text())
    assert result["frames"] == 8
    assert result["detections"] == 7
    assert [frame.timestamp_s for frame in artifact.frames] == pytest.approx(timestamps, abs=1e-5)
    assert {track.track_id for frame in artifact.frames for track in frame.tracks} == {0}
    assert artifact.frames[3].tracks == []
    with av.open(str(output / "annotated.mp4")) as preview:
        assert preview.streams.video[0].codec_context.name == "h264"
        assert [frame.time for frame in preview.decode(video=0)] == pytest.approx(
            timestamps, abs=1e-5
        )


def test_cache_reuse_and_tracker_only_rerun(sample_run):
    video, weights, output, _, detector = sample_run
    pipeline.run_pipeline(video, weights, output)
    second = pipeline.run_pipeline(video, weights, output)
    assert second["reused_detection"] and second["reused_tracking"]
    third = pipeline.run_pipeline(video, weights, output, tracker_config=TrackerConfig(max_age=15))
    assert third["reused_detection"] and not third["reused_tracking"]
    assert detector.starts == 1


def test_corrupt_artifacts_are_rebuilt(sample_run):
    video, weights, output, _, detector = sample_run
    pipeline.run_pipeline(video, weights, output)
    (output / "tracks.json").write_text("incomplete")
    result = pipeline.run_pipeline(video, weights, output)
    assert result["reused_detection"] and not result["reused_tracking"]
    TracksArtifact.model_validate_json((output / "tracks.json").read_text())
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
    video = video.rename(video.with_name("annotated.mp4"))
    with pytest.raises(ValueError, match="overwrite"):
        pipeline.run_pipeline(video, weights, video.parent)
