"""Stream BGR frames with media timestamps, without expanding a video onto disk."""
from dataclasses import dataclass
from pathlib import Path

import av
import numpy as np

from tracking.config import DecodeConfig
from tracking.schemas import VideoMetadata


@dataclass(frozen=True)
class VideoFrame:
    index: int
    timestamp_s: float
    image: np.ndarray


class VideoReader:
    def __init__(self, path: Path, config: DecodeConfig):
        self.path = Path(path)
        self.config = config
        self.container = None

    def __enter__(self):
        self.container = av.open(str(self.path))
        try:
            if not self.container.streams.video:
                raise ValueError(f"No video stream in {self.path}")
            self.stream = self.container.streams.video[0]
            rate = self.stream.average_rate or self.stream.guessed_rate
            if rate is None:
                raise ValueError("Video has no usable frame rate; transcode it first")
            self.metadata = VideoMetadata(
                width=self.stream.width, height=self.stream.height, fps=float(rate)
            )
        except Exception:
            self.container.close()
            raise
        return self

    def __iter__(self):
        origin = None
        previous = -1.0
        emitted = 0
        for index, frame in enumerate(self.container.decode(self.stream)):
            timestamp = frame.time
            if timestamp is None:
                timestamp = (origin or 0.0) + index / self.metadata.fps
            if origin is None:
                origin = timestamp
            timestamp -= origin
            if timestamp <= previous:
                raise ValueError("Non-increasing video timestamps; transcode the clip first")
            previous = timestamp
            if self.config.max_seconds is not None and timestamp >= self.config.max_seconds:
                break
            if index % self.config.stride:
                continue
            if (frame.width, frame.height) != (self.metadata.width, self.metadata.height):
                raise ValueError("Frame dimensions changed within the video")
            yield VideoFrame(index, timestamp, frame.to_ndarray(format="bgr24"))
            emitted += 1
            if self.config.max_frames is not None and emitted >= self.config.max_frames:
                break

    def __exit__(self, *exc):
        self.container.close()
