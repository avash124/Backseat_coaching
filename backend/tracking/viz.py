"""A silent H.264 debug preview; product overlays will use tracks.json directly."""

from fractions import Fraction
from pathlib import Path

import av
import cv2

from tracking.schemas import FrameTracks, VideoMetadata


class PreviewWriter:
    def __init__(self, path: Path, video: VideoMetadata, stride: int):
        self.container = av.open(
            str(path), mode="w", format="mp4", options={"movflags": "+faststart"}
        )
        self.stream = self.container.add_stream(
            "libx264", rate=Fraction(str(video.fps / stride)).limit_denominator(100000)
        )
        self.stream.width = video.width + video.width % 2
        self.stream.height = video.height + video.height % 2
        self.stream.pix_fmt = "yuv420p"
        self.stream.time_base = Fraction(1, 90000)
        self.stream.codec_context.time_base = Fraction(1, 90000)
        self.stream.options = {"crf": "23", "preset": "fast", "bf": "0"}

    def __enter__(self):
        return self

    def write(self, image, tracks: FrameTracks):
        overlay = image.copy()
        for track in tracks.tracks:
            x1, y1, x2, y2 = (round(value) for value in track.bbox_xyxy)
            color = tuple(60 + (track.track_id * factor) % 196 for factor in (53, 97, 151))
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
            label = f"ID {track.track_id} {track.confidence:.2f}"
            if not track.observed:
                label += " predicted"
            cv2.putText(
                overlay,
                label,
                (x1, max(18, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )
        cv2.putText(
            overlay,
            f"Frame {tracks.frame_index} | {tracks.timestamp_s:.2f}s",
            (12, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        overlay = cv2.copyMakeBorder(
            overlay, 0, overlay.shape[0] % 2, 0, overlay.shape[1] % 2, cv2.BORDER_CONSTANT
        )
        frame = av.VideoFrame.from_ndarray(overlay, format="bgr24")
        frame.pts = round(tracks.timestamp_s * 90000)
        frame.time_base = Fraction(1, 90000)
        for packet in self.stream.encode(frame):
            self.container.mux(packet)

    def __exit__(self, exc_type, *_):
        try:
            if exc_type is None:
                for packet in self.stream.encode():
                    self.container.mux(packet)
        finally:
            self.container.close()
