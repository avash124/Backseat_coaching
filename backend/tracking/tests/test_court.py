import numpy as np
import pytest
from pydantic import ValidationError

from tracking.court import CourtConfig, CourtFilter


def test_feet_filter_keeps_boundary_and_margin_but_removes_spectators():
    court = CourtFilter(
        CourtConfig(polygon=((0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)), margin=0.02),
        width=200,
        height=100,
    )
    boxes = np.array(
        [
            [70, 0, 90, 40, 0.9, 0],  # Torso center at edge, feet inside.
            [70, 60, 90, 80, 0.9, 0],  # Exactly on boundary.
            [70, 60, 90, 81, 0.9, 0],  # Within two-pixel margin.
            [70, 60, 90, 95, 0.99, 0],  # Center inside, spectator's feet outside.
            [0, 30, 20, 60, 0.99, 0],  # Outside to the left.
        ],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(court.filter(boxes), boxes[:3])
    assert court.filter(np.empty((0, 6), np.float32)).shape == (0, 6)


@pytest.mark.parametrize(
    "polygon",
    [
        ((0, 0), (1, 1)),
        ((0, 0), (0.5, 0.5), (1, 1)),
        ((0, 0), (1, 0), (0, 1), (0, 0)),
        ((0, 0), (1, 0.8), (0, 1), (0.8, 0)),
        ((0, 0), (float("nan"), 0), (1, 1)),
        ((0, 0), (2, 0), (1, 1)),
    ],
)
def test_invalid_polygon_is_rejected(polygon):
    with pytest.raises(ValidationError):
        CourtConfig(polygon=polygon)
