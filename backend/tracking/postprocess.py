"""Offline cleanup of raw tracker output: tracks_raw.json -> tracks.json.

Runs as its own cached stage, so changing these settings never reruns detection
or tracking. Steps, in order:

1. Backfill: tracks that were confirmed regain the observations hidden while they
   were tentative (min_hits); never-confirmed tracks are discarded.
2. Role vote per raw track: summed per-class detector scores of its boxes.
3. Stitching: a track's end is linked to a later track's start in the same shot
   when they do not overlap, the gap is short, both have the same role, neither
   boundary box is a wide multi-person box, box heights agree, torso colors
   match, and the start lies within a distance gate around the end's
   constant-velocity foot prediction. Links are solved globally (Hungarian) to
   maximize the summed gain ``1 - cost``, where cost averages the normalized
   distance and color terms. Roles are re-voted after merging.
4. Interpolation: gaps no longer than ``interpolate_max_gap_s`` between two boxes
   of a track are filled linearly with ``observed=false`` boxes (confidence 0).
5. Short tracks spanning less than ``min_track_s`` are removed.

``proxy_metrics`` computes label-free diagnostics from any tracks.json payload.
They are proxies (fragmentation, coverage), not identity accuracy.
"""

from bisect import bisect_left, bisect_right
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import pairwise

import numpy as np

from tracking.config import PostprocessConfig
from tracking.schemas import FrameTracks, PlayerTrack, Role, StitchRecord, TrackInfo


@dataclass
class Fragment:
    """All boxes of one track, sorted by processed-frame position."""

    track_id: int
    positions: list[int]
    boxes: list[tuple[float, float, float, float]]
    confidences: list[float]
    class_ids: list[int]
    votes: np.ndarray
    merged_ids: list[int] = field(default_factory=list)
    role: Role = "unknown"
    # Mean torso histograms of the first/last boxes (track_appearance.json).
    start_color: np.ndarray | None = None
    end_color: np.ndarray | None = None


@dataclass
class PostprocessResult:
    frames: list[FrameTracks]
    track_info: list[TrackInfo]
    stitches: list[StitchRecord]
    stats: dict
    raw_roles: dict[int, Role]


def assign_role(votes: dict[str, float], margin: float, boundary: float = 0.5) -> Role:
    """Decide from the referee share of the player+referee vote.

    Referee above ``boundary + margin``, player below ``boundary - margin``,
    otherwise unknown. Without a referee class every voted track is a player.
    """
    player, referee = votes.get("player", 0.0), votes.get("referee", 0.0)
    if "referee" not in votes:
        return "player" if player > 0 else "unknown"
    total = player + referee
    if total <= 0:
        return "unknown"
    share = referee / total
    if share > boundary + margin:
        return "referee"
    if share < boundary - margin:
        return "player"
    return "unknown"


def _votes(fragment: Fragment, class_names: list[str]) -> dict[str, float]:
    return {name: round(float(value), 4) for name, value in zip(class_names, fragment.votes)}


def _role(fragment: Fragment, class_names: list[str], config: PostprocessConfig) -> Role:
    return assign_role(_votes(fragment, class_names), config.role_margin, config.role_boundary)


def build_fragments(
    raw_frames: list[dict], n_classes: int, backfill: bool
) -> tuple[list[Fragment], dict]:
    observations = defaultdict(list)
    for position, frame in enumerate(raw_frames):
        for box in frame["tracks"]:
            observations[box["track_id"]].append((position, box))
    fragments = []
    stats = {"never_confirmed_tracks": 0, "backfilled_boxes": 0, "dropped_tentative_boxes": 0}
    for track_id in sorted(observations):
        items = observations[track_id]
        if not any(box["confirmed"] for _, box in items):
            stats["never_confirmed_tracks"] += 1
            stats["dropped_tentative_boxes"] += len(items)
            continue
        kept = [(p, box) for p, box in items if backfill or box["confirmed"]]
        tentative = sum(not box["confirmed"] for _, box in kept)
        stats["backfilled_boxes"] += tentative
        stats["dropped_tentative_boxes"] += len(items) - len(kept)
        scores = np.zeros(n_classes)
        for _, box in kept:
            scores += np.asarray(box["class_scores"][:n_classes], dtype=float)
        fragments.append(
            Fragment(
                track_id=track_id,
                positions=[p for p, _ in kept],
                boxes=[tuple(box["bbox_xyxy"]) for _, box in kept],
                confidences=[box["confidence"] for _, box in kept],
                class_ids=[box["class_id"] for _, box in kept],
                votes=scores,
                merged_ids=[track_id],
            )
        )
    return fragments, stats


def _foot(box) -> np.ndarray:
    return np.array(((box[0] + box[2]) / 2, box[3]), dtype=float)


def _height(box) -> float:
    return float(box[3] - box[1])


def _aspect(box) -> float:
    return float((box[2] - box[0]) / (box[3] - box[1]))


def color_distance(first: np.ndarray | None, second: np.ndarray | None) -> float | None:
    """Bhattacharyya distance of two normalized histograms; None if either is missing."""
    if first is None or second is None:
        return None
    coefficient = float(np.sqrt(np.clip(first, 0, None) * np.clip(second, 0, None)).sum())
    return float(np.sqrt(max(0.0, 1.0 - min(coefficient, 1.0))))


def _end_velocity(fragment: Fragment, times: np.ndarray, window_s: float) -> np.ndarray:
    """Least-squares foot velocity (px/s) over the fragment's final window."""
    end_time = times[fragment.positions[-1]]
    start = bisect_left([times[p] for p in fragment.positions], end_time - window_s)
    sample_times = np.array([times[p] for p in fragment.positions[start:]])
    if len(sample_times) < 2 or np.ptp(sample_times) <= 0:
        return np.zeros(2)
    feet = np.array([_foot(box) for box in fragment.boxes[start:]])
    centered = sample_times - sample_times.mean()
    return (centered[:, None] * (feet - feet.mean(axis=0))).sum(axis=0) / (centered**2).sum()


def stitch_fragments(
    fragments: list[Fragment],
    times: np.ndarray,
    shot_of_position: np.ndarray,
    config: PostprocessConfig,
) -> tuple[list[Fragment], list[StitchRecord]]:
    from scipy.optimize import linear_sum_assignment

    if not fragments or config.stitch_max_gap_s <= 0:
        return fragments, []
    order = sorted(range(len(fragments)), key=lambda i: fragments[i].positions[0])
    start_times = [times[fragments[i].positions[0]] for i in order]
    candidates = []  # (end fragment, start fragment, cost, gap, distance, gate, color)
    for a, fragment in enumerate(fragments):
        last = fragment.positions[-1]
        end_time = times[last]
        velocity = _end_velocity(fragment, times, config.stitch_velocity_window_s)
        low = bisect_right(start_times, end_time)
        high = bisect_right(start_times, end_time + config.stitch_max_gap_s + 1e-9)
        for b in (order[k] for k in range(low, high)):
            other = fragments[b]
            first = other.positions[0]
            if (
                b == a
                or first <= last
                or shot_of_position[first] != shot_of_position[last]
                or other.role != fragment.role
            ):
                continue
            end_box, start_box = fragment.boxes[-1], other.boxes[0]
            if max(_aspect(end_box), _aspect(start_box)) > config.stitch_max_aspect:
                continue
            heights = sorted((_height(end_box), _height(start_box)))
            if heights[1] > heights[0] * config.stitch_max_height_ratio:
                continue
            color = color_distance(fragment.end_color, other.start_color)
            if color is not None and color > config.stitch_max_appearance:
                continue
            gap = float(times[first] - end_time)
            gate = (sum(heights) / 2) * (config.stitch_gate_base + config.stitch_gate_per_s * gap)
            distance = float(np.linalg.norm(_foot(end_box) + velocity * gap - _foot(start_box)))
            if gate > 0 and distance <= gate:
                cost = distance / gate
                if color is not None:
                    cost = (cost + color / config.stitch_max_appearance) / 2
                candidates.append((a, b, cost, gap, distance, gate, color))
    if not candidates:
        return fragments, []
    ends = sorted({c[0] for c in candidates})
    starts = sorted({c[1] for c in candidates})
    row, column = {a: i for i, a in enumerate(ends)}, {b: j for j, b in enumerate(starts)}
    # Maximize total gain (1 - cost); infeasible pairs have zero gain.
    matrix = np.zeros((len(ends), len(starts)))
    details = {}
    for a, b, cost, *rest in candidates:
        matrix[row[a], column[b]] = min(cost, 1.0 - 1e-9) - 1
        details[(a, b)] = (cost, *rest)
    links = {}
    for i, j in zip(*linear_sum_assignment(matrix)):
        if matrix[i, j] < 0:
            links[ends[i]] = starts[j]
    # Follow chains from fragments that are not themselves a link target.
    targets = set(links.values())
    merged, records = [], []
    for a in order:
        if a in targets:
            continue
        chain = fragments[a]
        current = a
        while current in links:
            following = links[current]
            cost, gap, distance, gate, color = details[(current, following)]
            records.append(
                StitchRecord(
                    kept_id=chain.track_id,
                    merged_id=fragments[following].track_id,
                    previous_id=fragments[current].track_id,
                    gap_s=round(gap, 6),
                    distance_px=round(distance, 3),
                    gate_px=round(gate, 3),
                    color_distance=None if color is None else round(color, 4),
                    cost=round(cost, 6),
                )
            )
            chain = _concatenate(chain, fragments[following])
            current = following
        merged.append(chain)
    return merged, records


def _concatenate(first: Fragment, second: Fragment) -> Fragment:
    return Fragment(
        track_id=first.track_id,
        positions=first.positions + second.positions,
        boxes=first.boxes + second.boxes,
        confidences=first.confidences + second.confidences,
        class_ids=first.class_ids + second.class_ids,
        votes=first.votes + second.votes,
        merged_ids=first.merged_ids + second.merged_ids,
        role=first.role,
        start_color=first.start_color,
        end_color=second.end_color,
    )


def interpolate(
    fragment: Fragment, times: np.ndarray, shot_of_position: np.ndarray, max_gap_s: float
) -> list[tuple[int, tuple, float, int, bool]]:
    """Return (position, box, confidence, class_id, observed) including filled gaps."""
    out = []
    items = list(zip(fragment.positions, fragment.boxes, fragment.confidences, fragment.class_ids))
    for index, (position, box, confidence, class_id) in enumerate(items):
        out.append((position, box, confidence, class_id, True))
        if index + 1 == len(items):
            break
        next_position, next_box = items[index + 1][0], items[index + 1][1]
        span = times[next_position] - times[position]
        if (
            next_position - position < 2
            or span > max_gap_s + 1e-9
            or shot_of_position[next_position] != shot_of_position[position]
        ):
            continue
        start, end = np.asarray(box, float), np.asarray(next_box, float)
        for missing in range(position + 1, next_position):
            weight = (times[missing] - times[position]) / span
            filled = tuple(float(v) for v in (1 - weight) * start + weight * end)
            out.append((missing, filled, 0.0, class_id, False))
    return out


def postprocess_tracks(
    raw_frames: list[dict],
    class_names: list[str],
    cut_frame_indices: list[int],
    config: PostprocessConfig,
    appearance: dict | None = None,
) -> PostprocessResult:
    """Clean raw frames; ``appearance`` maps track IDs to {"start": hist, "end": hist}."""
    times = np.array([frame["timestamp_s"] for frame in raw_frames], dtype=float)
    cuts = set(cut_frame_indices)
    shot_of_position = np.cumsum([frame["frame_index"] in cuts for frame in raw_frames])
    fragments, stats = build_fragments(raw_frames, len(class_names), config.backfill)
    for fragment in fragments:
        colors = (appearance or {}).get(str(fragment.track_id), {})
        if colors:
            fragment.start_color = np.asarray(colors["start"], dtype=float)
            fragment.end_color = np.asarray(colors["end"], dtype=float)
    for fragment in fragments:
        fragment.role = _role(fragment, class_names, config)
    raw_roles = {fragment.track_id: fragment.role for fragment in fragments}
    fragments, stitches = stitch_fragments(fragments, times, shot_of_position, config)
    per_position = defaultdict(list)
    track_info = []
    stats.update(short_tracks_removed=0, short_track_boxes_removed=0, interpolated_boxes=0)
    for fragment in fragments:
        fragment.role = _role(fragment, class_names, config)
        span = times[fragment.positions[-1]] - times[fragment.positions[0]]
        if span < config.min_track_s:
            stats["short_tracks_removed"] += 1
            stats["short_track_boxes_removed"] += len(fragment.positions)
            continue
        boxes = interpolate(fragment, times, shot_of_position, config.interpolate_max_gap_s)
        filled = sum(not observed for *_, observed in boxes)
        stats["interpolated_boxes"] += filled
        for position, box, confidence, class_id, observed in boxes:
            per_position[position].append(
                PlayerTrack(
                    track_id=fragment.track_id,
                    bbox_xyxy=box,
                    confidence=confidence,
                    class_id=class_id,
                    observed=observed,
                    role=fragment.role,
                )
            )
        track_info.append(
            TrackInfo(
                track_id=fragment.track_id,
                role=fragment.role,
                role_votes=_votes(fragment, class_names),
                first_frame_index=raw_frames[fragment.positions[0]]["frame_index"],
                last_frame_index=raw_frames[fragment.positions[-1]]["frame_index"],
                observed_boxes=len(fragment.positions),
                interpolated_boxes=filled,
                merged_ids=fragment.merged_ids,
            )
        )
    frames = [
        FrameTracks(
            frame_index=frame["frame_index"],
            timestamp_s=frame["timestamp_s"],
            tracks=sorted(per_position.get(position, []), key=lambda track: track.track_id),
        )
        for position, frame in enumerate(raw_frames)
    ]
    track_info.sort(key=lambda info: info.track_id)
    stats["stitches"] = len(stitches)
    return PostprocessResult(frames, track_info, stitches, stats, raw_roles)


def _runs(positions: list[int]) -> tuple[int, int]:
    """Number of gaps and missing positions between the first and last position."""
    runs = missing = 0
    for previous, current in pairwise(positions):
        if current - previous > 1:
            runs += 1
            missing += current - previous - 1
    return runs, missing


def proxy_metrics(frames: list[dict], min_players: int = 8, long_track_s: float = 10.0) -> dict:
    """Label-free diagnostics over player boxes of a tracks.json ``frames`` list.

    Boxes without a ``role`` (schema 1.0) count as players. Gaps are counted
    inside each track's first..last processed frame: ``gaps`` are runs of
    missing frames, ``gap_frames`` the missing frames. ``observed_*`` ignores
    interpolated boxes, i.e. the value before interpolation.
    """
    times = [frame["timestamp_s"] for frame in frames]
    period = float(np.median(np.diff(times))) if len(times) > 1 else 0.0
    every, observed = defaultdict(list), defaultdict(list)
    roles = {}
    per_frame = []
    for position, frame in enumerate(frames):
        players = 0
        for box in frame["tracks"]:
            role = box.get("role") or "player"
            roles[box["track_id"]] = role
            if role != "player":
                continue
            players += 1
            every[box["track_id"]].append(position)
            if box.get("observed", True):
                observed[box["track_id"]].append(position)
        per_frame.append(players)
    gaps_all = [_runs(positions) for positions in every.values()]
    gaps_observed = [_runs(positions) for positions in observed.values()]
    role_counts = {role: 0 for role in ("player", "referee", "unknown")}
    for role in roles.values():
        role_counts[role] = role_counts.get(role, 0) + 1
    return {
        "frames": len(frames),
        "unique_player_ids": len(every),
        f"player_tracks_at_least_{long_track_s:g}s": sum(
            len(positions) * period >= long_track_s - 1e-9 for positions in every.values()
        ),
        "intra_track_gaps": sum(runs for runs, _ in gaps_all),
        "intra_track_gap_frames": sum(missing for _, missing in gaps_all),
        "observed_intra_track_gaps": sum(runs for runs, _ in gaps_observed),
        "observed_intra_track_gap_frames": sum(missing for _, missing in gaps_observed),
        f"frames_with_under_{min_players}_players": sum(count < min_players for count in per_frame),
        "mean_players_per_frame": round(sum(per_frame) / max(len(frames), 1), 4),
        "tracks_by_role": role_counts,
        "interpolated_player_boxes": sum(
            len(every[track]) - len(observed.get(track, [])) for track in every
        ),
    }
