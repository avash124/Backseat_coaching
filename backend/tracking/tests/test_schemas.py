import pytest
from pydantic import ValidationError

from tracking.schemas import FrameTracks, PlayerTrack, TracksArtifact, VideoMetadata


def player(**changes):
    return PlayerTrack.model_validate(
        {
            "track_id": 1,
            "bbox_xyxy": [10, 20, 30, 40],
            "confidence": 0.8,
            "class_id": 0,
            **changes,
        }
    )


@pytest.mark.parametrize("box", [[10, 10, 10, 30], [-1, 0, 10, 10], [0, 0, float("nan"), 5]])
def test_invalid_box_rejected(box):
    with pytest.raises(ValidationError):
        player(bbox_xyxy=box)


@pytest.mark.parametrize(
    "frames",
    [
        [FrameTracks(frame_index=1, timestamp_s=1), FrameTracks(frame_index=0, timestamp_s=2)],
        [FrameTracks(frame_index=0, timestamp_s=1), FrameTracks(frame_index=1, timestamp_s=1)],
        [FrameTracks(frame_index=0, timestamp_s=0, tracks=[player(), player()])],
        [FrameTracks(frame_index=0, timestamp_s=0, tracks=[player(bbox_xyxy=[0, 0, 101, 50])])],
    ],
)
def test_invalid_sequence_rejected(frames):
    with pytest.raises(ValidationError):
        TracksArtifact(
            video=VideoMetadata(width=100, height=100, fps=30), provenance={}, frames=frames
        )
