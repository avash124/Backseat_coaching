"""Tiny synthetic own-footage frames for smoke tests and `training smoke`.

Writes images + canonical-id labels that follow the shared naming convention, so they go
through the real builder (own_footage.py) exactly like corrected CVAT exports would.
"""

import random
from pathlib import Path

from tracking.datasets.common import frame_stem
from tracking.datasets.taxonomy import CLASS_ID

COLORS = {"player": (220, 60, 60), "referee": (240, 240, 240),
          "basketball": (230, 120, 20), "hoop": (200, 30, 30)}


def write_frames(
    directory: Path,
    *,
    groups: tuple[str, ...] = ("synth-a", "synth-b", "synth-c", "synth-d"),
    frames_per_group: int = 2,
    size: int = 128,
    classes: tuple[str, ...] = ("player", "referee"),
    seed: int = 0,
) -> list[Path]:
    from PIL import Image, ImageDraw

    rng = random.Random(seed)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    written = []
    for group in groups:
        for index in range(frames_per_group):
            stem = frame_stem(group, index * 30)
            image = Image.new("RGB", (size, size), (70, 90, 60))
            draw = ImageDraw.Draw(image)
            lines = []
            for name in classes:
                w, h = rng.randint(size // 10, size // 4), rng.randint(size // 6, size // 3)
                x, y = rng.randint(0, size - w - 1), rng.randint(0, size - h - 1)
                draw.rectangle([x, y, x + w, y + h], fill=COLORS[name])
                lines.append(f"{CLASS_ID[name]} {(x + w / 2) / size:.6f} {(y + h / 2) / size:.6f} "
                             f"{w / size:.6f} {h / size:.6f}")
            image.save(directory / f"{stem}.jpg", quality=90)
            (directory / f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8",
                                                    newline="\n")
            written.append(directory / f"{stem}.jpg")
    return written
