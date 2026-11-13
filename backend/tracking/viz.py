"""A silent H.264 debug preview; product overlays will use tracks.json directly.

Players get a per-ID color. Referees are drawn in magenta and labeled REF;
unknown-role tracks in gray with a ``?``. Interpolated (observed=false) boxes use
a thin outline and the label ``interp``. The frame where a shot cut was detected
shows a red banner.
"""

from fractions import Fraction
from pathlib import Path

import av
import cv2
import numpy as np

from tracking.court import CourtFilter
from tracking.schemas import FrameTracks, ShotCut, VideoMetadata

ROLE_COLORS = {"referee": (255, 0, 255), "unknown": (160, 160, 160)}
ROLE_TAGS = {"player": "", "referee": " REF", "unknown": " ?"}


def track_color(track_id: int, role: str | None) -> tuple[int, int, int]:
    if role in ROLE_COLORS:
        return ROLE_COLORS[role]
    return tuple(60 + (track_id * factor) % 196 for factor in (53, 97, 151))


class PreviewWriter:
    def __init__(
        self, path: Path, video: VideoMetadata, stride: int, court: CourtFilter | None = None
    ):
        self.court = court
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

    def write(self, image, tracks: FrameTracks, cut: ShotCut | None = None):
        overlay = image.copy()
        if self.court is not None:
            cv2.polylines(
                overlay, [np.rint(self.court.polygon).astype(np.int32)], True, (0, 220, 255), 2
            )
        for track in tracks.tracks:
            x1, y1, x2, y2 = (round(value) for value in track.bbox_xyxy)
            color = track_color(track.track_id, track.role)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2 if track.observed else 1)
            label = f"ID {track.track_id}{ROLE_TAGS.get(track.role or 'player', '')}"
            label += f" {track.confidence:.2f}" if track.observed else " interp"
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
        header = f"Frame {tracks.frame_index} | {tracks.timestamp_s:.2f}s"
        cv2.putText(
            overlay, header, (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
        )
        if cut is not None:
            cv2.rectangle(overlay, (0, 32), (overlay.shape[1], 64), (0, 0, 200), -1)
            cv2.putText(
                overlay,
                f"SHOT CUT (distance {cut.distance:.2f}) - tracker restarted",
                (12, 56),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
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
