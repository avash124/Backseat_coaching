"""Optional court boundary for a continuous, fixed-camera shot.

Vertices follow the visible court perimeter in normalized image coordinates.
This is an image-space exclusion filter, not a calibrated court homography.
Panning/zooming footage needs updated boundaries before using this filter.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Annotated

from pydantic import BaseModel, ConfigDict, Field, model_validator

if TYPE_CHECKING:
    import numpy as np

Coordinate = Annotated[float, Field(ge=0, le=1)]


class CourtConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
    polygon: tuple[tuple[Coordinate, Coordinate], ...] = Field(min_length=3)
    # Tolerance for foot-point jitter/inbounders, as a fraction of image height.
    margin: float = Field(default=0.008, ge=0, le=0.1)

    @model_validator(mode="after")
    def valid_polygon(self):
        points = self.polygon
        twice_area = abs(
            sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(points, points[1:] + points[:1]))
        )
        if len(set(points)) != len(points) or twice_area < 2e-6:
            raise ValueError("Court polygon needs distinct vertices and nonzero area")

        def cross(a, b, c):
            return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))

        def on_segment(a, b, c):
            return (
                abs(cross(a, b, c)) < 1e-8
                and min(a[0], b[0]) <= c[0] <= max(a[0], b[0])
                and min(a[1], b[1]) <= c[1] <= max(a[1], b[1])
            )

        for i in range(len(points)):
            a, b = points[i], points[(i + 1) % len(points)]
            for j in range(i + 1, len(points)):
                if j == i + 1 or (i == 0 and j == len(points) - 1):
                    continue
                c, d = points[j], points[(j + 1) % len(points)]
                crossing = (
                    cross(a, b, c) * cross(a, b, d) < 0 and cross(c, d, a) * cross(c, d, b) < 0
                )
                if crossing or any(
                    (
                        on_segment(a, b, c),
                        on_segment(a, b, d),
                        on_segment(c, d, a),
                        on_segment(c, d, b),
                    )
                ):
                    raise ValueError("Court polygon edges must not intersect")
        return self

    @classmethod
    def from_file(cls, path: Path):
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))


class CourtFilter:
    def __init__(self, config: CourtConfig, width: int, height: int):
        import numpy as np

        self.polygon = np.asarray(config.polygon, dtype=np.float32) * (width, height)
        self.polygon = self.polygon.astype(np.float32)
        self.margin_pixels = config.margin * height

    def contains(self, detections: np.ndarray) -> np.ndarray:
        """Boolean mask of rows whose foot point lies inside the boundary plus margin."""
        import cv2
        import numpy as np

        # Bottom center estimates feet; the torso may overlap off-court seating.
        keep = [
            cv2.pointPolygonTest(self.polygon, (float((box[0] + box[2]) / 2), float(box[3])), True)
            >= -self.margin_pixels
            for box in detections
        ]
        return np.asarray(keep, dtype=bool).reshape(-1)

    def filter(self, detections: np.ndarray) -> np.ndarray:
        return detections[self.contains(detections)]
