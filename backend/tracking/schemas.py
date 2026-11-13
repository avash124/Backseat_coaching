"""Artifact contracts.

Schema 1.1 (tracks.json) keeps every 1.0 field and adds:
- ``role`` on every box: the track-level decision ``player``/``referee``/``unknown``.
  Referee and unknown tracks stay in the artifact; consumers that want players
  must filter on ``role == "player"``.
- ``observed=false`` boxes, which are interpolated inside short gaps.
- ``track_info``: one entry per track ID with its role vote and stitch history.
- ``shot_cuts``: frames where a hard cut was detected and the tracker restarted.
- ``class_names``: detector class IDs used by ``class_id`` and the role vote.

tracks_raw.json holds the tracker output before cleanup; its boxes carry per-class
scores and whether the track was confirmed (``min_hits``) at that frame.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Role = Literal["player", "referee", "unknown"]
Box = tuple[float, float, float, float]


class ArtifactModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class VideoMetadata(ArtifactModel):
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    fps: float = Field(gt=0)
    timestamp_origin: Literal["first_decoded_frame"] = "first_decoded_frame"


def _check_box(box: Box):
    x1, y1, x2, y2 = box
    if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1:
        raise ValueError("Expected a positive-area box in original-frame pixels")


class PlayerTrack(ArtifactModel):
    """One person box in one frame (the name predates referee tracking)."""

    track_id: int = Field(ge=0)
    bbox_xyxy: Box
    # Detector score of this box; 0 for interpolated (observed=false) boxes.
    confidence: float = Field(ge=0, le=1)
    # Detector class of this box (see class_names); the track decision is `role`.
    class_id: int = Field(ge=0)
    observed: bool = True
    role: Role | None = None

    @model_validator(mode="after")
    def valid_box(self):
        _check_box(self.bbox_xyxy)
        return self


class FrameTracks(ArtifactModel):
    frame_index: int = Field(ge=0)
    timestamp_s: float = Field(ge=0)
    tracks: list[PlayerTrack] = Field(default_factory=list)


class ShotCut(ArtifactModel):
    """The first frame of a new shot; the tracker restarted here."""

    frame_index: int = Field(ge=0)
    timestamp_s: float = Field(ge=0)
    distance: float = Field(ge=0, le=1)


class StitchRecord(ArtifactModel):
    """Raw track `merged_id` continues `previous_id` after a gap; it is now `kept_id`."""

    kept_id: int = Field(ge=0)
    merged_id: int = Field(ge=0)
    previous_id: int = Field(ge=0)
    gap_s: float = Field(ge=0)
    distance_px: float = Field(ge=0)
    gate_px: float = Field(ge=0)
    # Torso-color Bhattacharyya distance; None when appearance was unavailable.
    color_distance: float | None = Field(default=None, ge=0, le=1)
    cost: float = Field(ge=0)


class TrackInfo(ArtifactModel):
    track_id: int = Field(ge=0)
    role: Role
    # Confidence-weighted vote: summed per-class detector scores of observed boxes.
    role_votes: dict[str, float]
    first_frame_index: int = Field(ge=0)
    last_frame_index: int = Field(ge=0)
    observed_boxes: int = Field(ge=0)
    interpolated_boxes: int = Field(ge=0)
    # Raw tracker IDs merged into this one by stitching, in temporal order.
    merged_ids: list[int] = Field(default_factory=list)


class TracksArtifact(ArtifactModel):
    schema_version: Literal["1.0", "1.1"] = "1.1"
    video: VideoMetadata
    provenance: dict
    class_names: dict[int, str] = Field(default_factory=dict)
    shot_cuts: list[ShotCut] = Field(default_factory=list)
    track_info: list[TrackInfo] = Field(default_factory=list)
    frames: list[FrameTracks]

    @model_validator(mode="after")
    def chronological_frames(self):
        previous_index, previous_time = -1, -1.0
        version_1_0 = self.schema_version == "1.0"
        if version_1_0 and (self.class_names or self.shot_cuts or self.track_info):
            raise ValueError("Schema 1.0 artifacts have no class_names, shot_cuts, or track_info")
        roles = {info.track_id: info.role for info in self.track_info}
        if len(roles) != len(self.track_info):
            raise ValueError("Duplicate track_info entry")
        seen = set()
        for frame in self.frames:
            if frame.frame_index <= previous_index or frame.timestamp_s <= previous_time:
                raise ValueError("Frames and timestamps must increase strictly")
            ids = [track.track_id for track in frame.tracks]
            if len(ids) != len(set(ids)):
                raise ValueError("Duplicate track ID within a frame")
            for track in frame.tracks:
                if track.bbox_xyxy[2] > self.video.width or track.bbox_xyxy[3] > self.video.height:
                    raise ValueError("Track box exceeds source dimensions")
                if version_1_0:
                    if track.role is not None:
                        raise ValueError("Schema 1.0 boxes have no role")
                elif roles.get(track.track_id) != track.role or track.role is None:
                    raise ValueError("Each 1.1 box needs the role listed in track_info")
                seen.add(track.track_id)
            previous_index, previous_time = frame.frame_index, frame.timestamp_s
        if not version_1_0 and seen != set(roles):
            raise ValueError("track_info must list exactly the track IDs present in frames")
        frame_indices = {frame.frame_index for frame in self.frames}
        if any(cut.frame_index not in frame_indices for cut in self.shot_cuts):
            raise ValueError("Shot cuts must refer to processed frames")
        return self


class RawObservation(ArtifactModel):
    """A tracker-associated detection before cleanup (tracks_raw.json)."""

    track_id: int = Field(ge=0)
    bbox_xyxy: Box
    confidence: float = Field(ge=0, le=1)
    class_id: int = Field(ge=0)
    # Always true: the raw artifact contains detector boxes only.
    observed: Literal[True] = True
    # Detector score per class, ordered like the artifact's class_names keys.
    class_scores: tuple[float, ...]
    # False while the track had fewer than min_hits consecutive observations.
    confirmed: bool

    @model_validator(mode="after")
    def valid_box(self):
        _check_box(self.bbox_xyxy)
        return self


class RawFrameTracks(ArtifactModel):
    frame_index: int = Field(ge=0)
    timestamp_s: float = Field(ge=0)
    tracks: list[RawObservation] = Field(default_factory=list)


class RawTracksArtifact(ArtifactModel):
    schema_version: Literal["1.1"] = "1.1"
    video: VideoMetadata
    provenance: dict
    class_names: dict[int, str]
    frames: list[RawFrameTracks]
