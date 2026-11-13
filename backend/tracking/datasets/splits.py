"""Train/val/test split policy.

Split by game, never by frame — adjacent frames are near-duplicates, and a frame-level
split leaks them across the boundary and inflates every metric. The held-out test set
is our own footage only (see evaluate.py); public val splits measure the wrong domain.

The split unit is a *group*: a game, or a game segment while we only have one game.
Group ids match `[a-z0-9-]+`.

Two hard guarantees, enforced by `check_splits` and called by every builder right before
anything is written:
1. No group appears in two splits.
2. No quarantined group (the M0 eval games, `evalsets/quarantine.yaml`) enters train or
   val. `split_groups` is stricter still: a quarantined group in its input is an error,
   because eval frames should never reach a training export in the first place.
"""

import hashlib
import math
import random
import warnings
from collections.abc import Iterable, Mapping
from fractions import Fraction
from pathlib import Path

import yaml

from tracking.datasets.common import validate_group_id

QUARANTINE_PATH = Path(__file__).resolve().parents[1] / "evalsets" / "quarantine.yaml"
TRAINING_SPLITS = ("train", "val")


class SplitLeakError(ValueError):
    """A group appears in more than one split."""


class QuarantineError(ValueError):
    """A quarantined eval group would be trained or validated on."""


def _quarantine_path(path: Path | None) -> Path:
    return Path(path) if path is not None else QUARANTINE_PATH


def load_quarantine(path: Path | None = None) -> frozenset[str]:
    """Quarantined group ids. A missing file is an empty list, with a warning."""
    path = _quarantine_path(path)
    if not path.is_file():
        warnings.warn(
            f"Quarantine list {path} not found; treating it as empty. Eval-set groups are "
            f"NOT protected from training until it exists.",
            stacklevel=2,
        )
        return frozenset()
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data is None:
        return frozenset()
    if not isinstance(data, dict) or not isinstance(data.get("groups") or [], list):
        raise TypeError(f"{path} must be a mapping with a `groups:` list")
    groups = set()
    for entry in data.get("groups") or []:
        if isinstance(entry, dict):  # tolerate `- id: game-1` style entries
            entry = next((entry[key] for key in ("id", "group", "name") if key in entry), None)
        groups.add(validate_group_id(entry))
    return frozenset(groups)


def quarantine_record(path: Path | None = None) -> dict:
    """What a manifest records about the quarantine list it was checked against."""
    path = _quarantine_path(path).resolve()
    exists = path.is_file()
    groups = sorted(load_quarantine(path)) if exists else []
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest() if exists else None,
        "groups": groups,
    }


def _held_out(count: int, fraction: float) -> int:
    if fraction == 0:
        return 0
    return max(1, math.floor(count * Fraction(str(fraction))))


def check_splits(splits: Mapping[str, Iterable[str]], quarantine: Iterable[str] = ()) -> None:
    """Raise on a group in two splits or a quarantined group in train/val."""
    seen: dict[str, str] = {}
    for split, groups in splits.items():
        for group in groups:
            validate_group_id(group)
            if seen.get(group, split) != split:
                raise SplitLeakError(f"Group {group!r} appears in both {seen[group]} and {split}")
            seen[group] = split
    leaked = sorted(g for g in set(quarantine) if seen.get(g) in TRAINING_SPLITS)
    if leaked:
        raise QuarantineError(
            f"Quarantined eval groups would enter train/val: {leaked}. "
            f"Remove them from the training inputs."
        )


def split_groups(
    groups: Iterable[str],
    *,
    seed: int = 42,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    quarantine: Iterable[str] = (),
) -> dict[str, list[str]]:
    """Seeded group-wise split. Each non-zero fraction gets at least one group.

    Sorted groups are shuffled with `random.Random(seed)`; the first groups go to val,
    the next to test, the rest to train. With the defaults this reproduces the original
    prepare_ebard split exactly.
    """
    unique = sorted({validate_group_id(group) for group in groups})
    quarantined = sorted(set(unique) & set(quarantine))
    if quarantined:
        raise QuarantineError(
            f"Quarantined eval groups are present in the training inputs: {quarantined}. "
            f"They must never enter train/val; remove them from the inputs."
        )
    for name, fraction in (("val", val_fraction), ("test", test_fraction)):
        if not 0 <= fraction < 1:
            raise ValueError(f"{name}_fraction must be in [0, 1), got {fraction}")
    if val_fraction == 0:
        raise ValueError("val_fraction must be > 0: training selects best.pt on val")
    val_count = _held_out(len(unique), val_fraction)
    test_count = _held_out(len(unique), test_fraction)
    if len(unique) - val_count - test_count < 1:
        raise ValueError(
            f"{len(unique)} groups cannot fill train plus {val_count} val and {test_count} "
            f"test groups; label more groups (games or segments) or lower the fractions"
        )
    shuffled = list(unique)
    random.Random(seed).shuffle(shuffled)
    splits = {
        "train": shuffled[val_count + test_count:],
        "val": shuffled[:val_count],
        "test": shuffled[val_count:val_count + test_count],
    }
    check_splits(splits, quarantine)
    return splits
