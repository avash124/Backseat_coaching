from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ArtifactModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class VideoMetadata(ArtifactModel):
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    fps: float = Field(gt=0)
    timestamp_origin: Literal["first_decoded_frame"] = "first_decoded_frame"


class PlayerTrack(ArtifactModel):
    track_id: int = Field(ge=0)
    bbox_xyxy: tuple[float, float, float, float]
    confidence: float = Field(ge=0, le=1)
    class_id: int = Field(ge=0)
    observed: bool = True

    @model_validator(mode="after")
    def valid_box(self):
        x1, y1, x2, y2 = self.bbox_xyxy
        if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1:
            raise ValueError("Expected a positive-area box in original-frame pixels")
        return self


class FrameTracks(ArtifactModel):
    frame_index: int = Field(ge=0)
    timestamp_s: float = Field(ge=0)
    tracks: list[PlayerTrack] = Field(default_factory=list)


class TracksArtifact(ArtifactModel):
    schema_version: Literal["1.0"] = "1.0"
    video: VideoMetadata
    provenance: dict
    frames: list[FrameTracks]

    @model_validator(mode="after")
    def chronological_frames(self):
        previous_index, previous_time = -1, -1.0
        for frame in self.frames:
            if frame.frame_index <= previous_index or frame.timestamp_s <= previous_time:
                raise ValueError("Frames and timestamps must increase strictly")
            ids = [track.track_id for track in frame.tracks]
            if len(ids) != len(set(ids)):
                raise ValueError("Duplicate track ID within a frame")
            for track in frame.tracks:
                if track.bbox_xyxy[2] > self.video.width or track.bbox_xyxy[3] > self.video.height:
                    raise ValueError("Track box exceeds source dimensions")
            previous_index, previous_time = frame.frame_index, frame.timestamp_s
        return self
