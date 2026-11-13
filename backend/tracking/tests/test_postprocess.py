import pytest

from tracking.config import PostprocessConfig
from tracking.postprocess import assign_role, postprocess_tracks, proxy_metrics

FPS = 30
CLASSES = ["player", "referee"]
KEEP_ALL = PostprocessConfig(min_track_s=0, stitch_max_gap_s=0, interpolate_max_gap_s=0)
# Stitching is off by default; these tests opt in.
STITCH = PostprocessConfig(min_track_s=0, stitch_max_gap_s=1.0)


def empty_frames(count):
    return [{"frame_index": i, "timestamp_s": i / FPS, "tracks": []} for i in range(count)]


def add_track(
    frames,
    track_id,
    positions,
    x0=100.0,
    vx=0.0,
    y=300.0,
    height=100.0,
    scores=(0.8, 0.1),
    tentative=(),
):
    """Add a 40 px wide box moving vx px/s; positions in `tentative` are unconfirmed."""
    for position in positions:
        x = x0 + vx * position / FPS
        frames[position]["tracks"].append(
            {
                "track_id": track_id,
                "bbox_xyxy": [x, y - height, x + 40, y],
                "confidence": max(scores),
                "class_id": int(scores[1] > scores[0]),
                "observed": True,
                "class_scores": list(scores),
                "confirmed": position not in tentative,
            }
        )


def ids_by_frame(result):
    return [[track.track_id for track in frame.tracks] for frame in result.frames]


def test_role_margin_rule():
    assert assign_role({"player": 8, "referee": 2}, 0.1) == "player"
    assert assign_role({"player": 3, "referee": 7}, 0.1) == "referee"
    assert assign_role({"player": 6, "referee": 4}, 0.1) == "unknown"
    assert assign_role({"player": 5, "referee": 5}, 0.0) == "unknown"
    assert assign_role({"player": 4}, 0.1) == "player"
    # A calibrated boundary moves the decision; the margin band moves with it.
    assert assign_role({"player": 65, "referee": 35}, 0.05, boundary=0.25) == "referee"
    assert assign_role({"player": 75, "referee": 25}, 0.05, boundary=0.25) == "unknown"


def test_role_is_a_confidence_weighted_vote_over_the_track_lifetime():
    frames = empty_frames(30)
    # Per-frame argmax is "player" in 10 frames, but referee evidence dominates overall.
    add_track(frames, 0, range(10), scores=(0.35, 0.3))
    add_track(frames, 0, range(10, 30), scores=(0.2, 0.7))
    add_track(frames, 1, range(30), x0=400, scores=(0.9, 0.0))
    result = postprocess_tracks(frames, CLASSES, [], KEEP_ALL)
    roles = {info.track_id: info.role for info in result.track_info}
    assert roles == {0: "referee", 1: "player"}
    assert all(track.role == roles[track.track_id] for f in result.frames for track in f.tracks)
    votes = result.track_info[0].role_votes
    assert votes == pytest.approx({"player": 7.5, "referee": 17.0})


def test_interpolation_fills_short_gaps_only_and_marks_them_unobserved():
    frames = empty_frames(80)
    # A 5-frame gap (0.2 s) and a 40-frame gap (1.37 s between observations).
    add_track(frames, 0, [*range(10), *range(15, 30), *range(70, 80)], vx=60)
    config = PostprocessConfig(min_track_s=0, stitch_max_gap_s=0, interpolate_max_gap_s=1.0)
    result = postprocess_tracks(frames, CLASSES, [], config)
    filled = [frame.tracks[0] for frame in result.frames[10:15]]
    assert all(not track.observed and track.confidence == 0 for track in filled)
    assert filled[2].bbox_xyxy[0] == pytest.approx(100 + 60 * 12 / FPS)
    assert all(frame.tracks == [] for frame in result.frames[30:70])
    assert result.track_info[0].interpolated_boxes == 5
    assert result.stats["interpolated_boxes"] == 5


def test_backfill_restores_frames_hidden_by_min_hits():
    frames = empty_frames(30)
    add_track(frames, 0, range(30), tentative={0, 1})
    add_track(frames, 1, [3, 4], x0=500, tentative={3, 4})  # Never confirmed.
    result = postprocess_tracks(frames, CLASSES, [], KEEP_ALL)
    assert ids_by_frame(result)[:2] == [[0], [0]]
    assert all(track.observed for track in result.frames[0].tracks)
    assert result.stats["backfilled_boxes"] == 2
    assert result.stats["never_confirmed_tracks"] == 1
    no_backfill = postprocess_tracks(
        frames, CLASSES, [], KEEP_ALL.model_copy(update={"backfill": False})
    )
    assert ids_by_frame(no_backfill)[:3] == [[], [], [0]]


def test_stitching_merges_a_split_track_and_logs_it():
    frames = empty_frames(90)
    add_track(frames, 0, range(30), vx=90)
    add_track(frames, 7, range(45, 90), vx=90)  # Same motion after a 0.5 s gap.
    config = STITCH
    result = postprocess_tracks(frames, CLASSES, [], config)
    assert {track.track_id for frame in result.frames for track in frame.tracks} == {0}
    assert [record.merged_id for record in result.stitches] == [7]
    assert result.stitches[0].kept_id == 0 and result.stitches[0].cost < 0.1
    assert result.track_info[0].merged_ids == [0, 7]
    # The stitched gap is short enough to interpolate.
    assert all(len(frame.tracks) == 1 for frame in result.frames)


def test_stitching_is_off_by_default():
    frames = empty_frames(90)
    add_track(frames, 0, range(30), vx=90)
    add_track(frames, 7, range(45, 90), vx=90)
    result = postprocess_tracks(frames, CLASSES, [], PostprocessConfig(min_track_s=0))
    assert result.stitches == []
    assert {info.track_id for info in result.track_info} == {0, 7}


def test_stitching_requires_matching_torso_colors():
    frames = empty_frames(90)
    add_track(frames, 0, range(30), vx=90)
    add_track(frames, 7, range(45, 90), vx=90)
    blue, dark = [0.9, 0.1, 0.0], [0.0, 0.1, 0.9]
    config = STITCH
    same = {"0": {"start": blue, "end": blue}, "7": {"start": blue, "end": blue}}
    merged = postprocess_tracks(frames, CLASSES, [], config, same)
    assert merged.stitches[0].color_distance == pytest.approx(0)
    other = {"0": {"start": blue, "end": blue}, "7": {"start": dark, "end": dark}}
    assert postprocess_tracks(frames, CLASSES, [], config, other).stitches == []
    disabled = config.model_copy(update={"stitch_max_appearance": 1.0})
    assert len(postprocess_tracks(frames, CLASSES, [], disabled, other).stitches) == 1


def test_stitching_skips_wide_multi_person_boxes():
    frames = empty_frames(90)
    add_track(frames, 0, range(30), vx=90)
    add_track(frames, 7, range(45, 90), vx=90)
    # The last box of track 0 spans two people: wider than tall.
    last = frames[29]["tracks"][0]["bbox_xyxy"]
    frames[29]["tracks"][0]["bbox_xyxy"] = [last[0] - 40, last[1] + 20, last[2] + 40, last[3]]
    assert postprocess_tracks(frames, CLASSES, [], STITCH).stitches == []


def test_stitching_never_merges_concurrent_tracks():
    frames = empty_frames(60)
    add_track(frames, 0, range(40))
    add_track(frames, 1, range(30, 60), x0=110)  # Overlaps track 0 in time.
    result = postprocess_tracks(frames, CLASSES, [], STITCH)
    assert result.stitches == []
    assert {info.track_id for info in result.track_info} == {0, 1}


def test_stitching_respects_distance_gate_role_and_shot_cuts():
    config = STITCH
    frames = empty_frames(60)
    add_track(frames, 0, range(20))
    add_track(frames, 1, range(30, 60), x0=600)  # Too far for a 0.33 s gap.
    assert postprocess_tracks(frames, CLASSES, [], config).stitches == []
    frames = empty_frames(60)
    add_track(frames, 0, range(20))
    add_track(frames, 1, range(30, 60), scores=(0.1, 0.8))  # Referee starts there.
    assert postprocess_tracks(frames, CLASSES, [], config).stitches == []
    frames = empty_frames(60)
    add_track(frames, 0, range(20))
    add_track(frames, 1, range(30, 60))
    assert len(postprocess_tracks(frames, CLASSES, [], config).stitches) == 1
    cut = postprocess_tracks(frames, CLASSES, [25], config)
    assert cut.stitches == []
    # Interpolation does not bridge the cut either.
    assert all(frame.tracks == [] for frame in cut.frames[20:30])


def test_stitching_solves_crossing_candidates_globally():
    frames = empty_frames(70)
    # Two players run towards each other; both tracks break at the same time.
    add_track(frames, 0, range(30), x0=100, vx=150)
    add_track(frames, 1, range(30), x0=600, vx=-150)
    add_track(frames, 2, range(40, 70), x0=100, vx=150)
    add_track(frames, 3, range(40, 70), x0=600, vx=-150)
    result = postprocess_tracks(frames, CLASSES, [], STITCH)
    assert {(r.kept_id, r.merged_id) for r in result.stitches} == {(0, 2), (1, 3)}


def test_short_tracks_are_removed_after_stitching():
    frames = empty_frames(60)
    add_track(frames, 0, range(10))  # 0.3 s alone, but continues as track 1.
    add_track(frames, 1, range(14, 24))
    add_track(frames, 2, range(40, 45), x0=700)  # Isolated 0.13 s blip.
    config = PostprocessConfig(min_track_s=0.5, stitch_max_gap_s=1.0)
    result = postprocess_tracks(frames, CLASSES, [], config)
    assert {info.track_id for info in result.track_info} == {0}
    assert result.stats["short_tracks_removed"] == 1


def test_proxy_metrics_count_players_gaps_and_roles():
    frames = [
        {"timestamp_s": i / FPS, "tracks": []} for i in range(6)
    ]
    for i in (0, 1, 4, 5):
        frames[i]["tracks"].append({"track_id": 0, "role": "player", "observed": True})
    for i in (2, 3):
        frames[i]["tracks"].append({"track_id": 0, "role": "player", "observed": False})
    for i in range(6):
        frames[i]["tracks"].append({"track_id": 9, "role": "referee", "observed": True})
    metrics = proxy_metrics(frames, min_players=2, long_track_s=0.1)
    assert metrics["unique_player_ids"] == 1
    assert metrics["intra_track_gaps"] == 0
    assert metrics["observed_intra_track_gaps"] == 1
    assert metrics["observed_intra_track_gap_frames"] == 2
    assert metrics["frames_with_under_2_players"] == 6
    assert metrics["mean_players_per_frame"] == 1
    assert metrics["tracks_by_role"] == {"player": 1, "referee": 1, "unknown": 0}
    assert metrics["player_tracks_at_least_0.1s"] == 1
    assert metrics["interpolated_player_boxes"] == 2
