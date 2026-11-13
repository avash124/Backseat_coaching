import pytest
from pydantic import ValidationError

from tracking.schemas import (
    FrameTracks,
    PlayerTrack,
    RawTracksArtifact,
    ShotCut,
    TrackInfo,
    TracksArtifact,
    VideoMetadata,
)

VIDEO = VideoMetadata(width=100, height=100, fps=30)


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


def info(track_id=1, role="player"):
    return TrackInfo(
        track_id=track_id,
        role=role,
        role_votes={"player": 0.8, "referee": 0.0},
        first_frame_index=0,
        last_frame_index=1,
        observed_boxes=1,
        interpolated_boxes=1,
    )


def artifact_1_1(frames=None, track_info=None, **changes):
    frames = frames or [
        FrameTracks(frame_index=0, timestamp_s=0, tracks=[player(role="player")]),
        FrameTracks(
            frame_index=1,
            timestamp_s=1 / 30,
            tracks=[player(role="player", observed=False, confidence=0)],
        ),
    ]
    return TracksArtifact(
        video=VIDEO,
        provenance={},
        class_names={0: "player", 1: "referee"},
        track_info=[info()] if track_info is None else track_info,
        frames=frames,
        **changes,
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
        TracksArtifact(schema_version="1.0", video=VIDEO, provenance={}, frames=frames)


def test_schema_1_0_artifacts_still_validate():
    frames = [FrameTracks(frame_index=0, timestamp_s=0, tracks=[player()])]
    artifact = TracksArtifact(schema_version="1.0", video=VIDEO, provenance={}, frames=frames)
    assert artifact.frames[0].tracks[0].role is None
    with pytest.raises(ValidationError, match="1.0"):
        TracksArtifact(
            schema_version="1.0",
            video=VIDEO,
            provenance={},
            frames=[FrameTracks(frame_index=0, timestamp_s=0, tracks=[player(role="player")])],
        )


def test_schema_1_1_round_trip_with_roles_cuts_and_track_info():
    artifact = artifact_1_1(shot_cuts=[ShotCut(frame_index=1, timestamp_s=1 / 30, distance=0.7)])
    parsed = TracksArtifact.model_validate_json(artifact.model_dump_json())
    assert parsed.schema_version == "1.1"
    assert parsed.class_names == {0: "player", 1: "referee"}
    assert not parsed.frames[1].tracks[0].observed
    assert parsed.track_info[0].role == "player"


@pytest.mark.parametrize(
    "changes",
    [
        {"track_info": [info(role="referee")]},  # Box role disagrees with track_info.
        {"track_info": []},  # Track missing from track_info.
        {"track_info": [info(), info(track_id=2)]},  # Listed track has no boxes.
        {"track_info": [info(), info()]},  # Duplicate entry.
        {"frames": [FrameTracks(frame_index=0, timestamp_s=0, tracks=[player()])]},  # No role.
        {"shot_cuts": [ShotCut(frame_index=5, timestamp_s=0.2, distance=0.7)]},  # Unknown frame.
    ],
)
def test_schema_1_1_consistency_rules(changes):
    with pytest.raises(ValidationError):
        artifact_1_1(**changes)


def test_raw_tracks_artifact_requires_scores_and_confirmation():
    payload = {
        "video": VIDEO.model_dump(),
        "provenance": {},
        "class_names": {"0": "player", "1": "referee"},
        "frames": [
            {
                "frame_index": 0,
                "timestamp_s": 0,
                "tracks": [
                    {
                        "track_id": 3,
                        "bbox_xyxy": [1, 2, 3, 4],
                        "confidence": 0.6,
                        "class_id": 0,
                        "class_scores": [0.6, 0.3],
                        "confirmed": False,
                    }
                ],
            }
        ],
    }
    raw = RawTracksArtifact.model_validate(payload)
    assert raw.class_names == {0: "player", 1: "referee"} and raw.frames[0].tracks[0].observed
    del payload["frames"][0]["tracks"][0]["confirmed"]
    with pytest.raises(ValidationError):
        RawTracksArtifact.model_validate(payload)
