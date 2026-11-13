"""The canonical class list, and per-source remapping into it.

Public sets disagree: some label `player`, some split `player`/`referee`, some add
`hoop`/`net`. Merging without an explicit remap silently trains on scrambled class
ids. Classes we do not use get dropped here, not ignored downstream.

Rules:
- `CANONICAL_CLASSES` is FROZEN. Index = class id, forever. Append-only would still change
  nothing already written, but reordering or renaming corrupts every label file on disk.
- Every source has a remap table: source class name -> canonical name, or `None` to drop
  it explicitly. A source class missing from its table raises `TaxonomyError`; there is
  no silent pass-through.
- A dataset may select a subset of the canonical classes (`select_classes`). The
  selection must be in canonical order, so a canonical *prefix* (`player,referee` or all
  four) keeps output ids equal to canonical ids. Any other subset (e.g. `basketball`
  alone for a ball-only detector) is compacted to 0..n-1 and the dataset's `names` list
  is authoritative.
"""

from collections.abc import Iterable, Mapping, Sequence
from types import MappingProxyType

CANONICAL_CLASSES: tuple[str, ...] = ("player", "referee", "basketball", "hoop")
CLASS_ID: Mapping[str, int] = MappingProxyType(
    {name: index for index, name in enumerate(CANONICAL_CLASSES)}
)
# Backward compatible default: the current detector (players-v1) is player/referee.
DEFAULT_CLASSES: tuple[str, ...] = ("player", "referee")

# Source name -> {source class name -> canonical class name | None (drop)}.
# Source names are the `remap` keys referenced from sources.yaml.
SOURCE_REMAPS: Mapping[str, Mapping[str, str | None]] = MappingProxyType({
    # eBAR export order is [basketball, hoop, player, referee]; prepare_ebard checks it.
    "ebard": MappingProxyType({
        "basketball": "basketball",
        "hoop": "hoop",
        "player": "player",
        "referee": "referee",
    }),
    # Our own labels (CVAT exports / labeling tools) use the canonical names directly.
    # Coaches, bench and crowd are not classes: they are left unlabeled per the
    # labeling guide. Add them here explicitly (-> None) if a labeling export ever
    # carries them, rather than letting them pass through.
    "own-footage": MappingProxyType({name: name for name in CANONICAL_CLASSES}),
})


class TaxonomyError(ValueError):
    """A class list or remap table that would scramble class ids."""


def _check_tables() -> None:
    for source, table in SOURCE_REMAPS.items():
        bad = sorted(
            f"{name}->{target}" for name, target in table.items()
            if target is not None and target not in CLASS_ID
        )
        if bad:
            raise TaxonomyError(f"Remap table {source!r} targets non-canonical classes: {bad}")


_check_tables()


def source_remap(source: str) -> Mapping[str, str | None]:
    try:
        return SOURCE_REMAPS[source]
    except KeyError:
        raise TaxonomyError(
            f"No remap table for source {source!r}; add one to taxonomy.SOURCE_REMAPS"
        ) from None


def validate_source_classes(source: str, names: Sequence[str]) -> None:
    """Raise unless every source class has an explicit remap entry."""
    names = list(names)
    if not names:
        raise TaxonomyError(f"Source {source!r} declares no classes")
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise TaxonomyError(f"Source {source!r} declares duplicate classes: {duplicates}")
    table = source_remap(source)
    unmapped = [name for name in names if name not in table]
    if unmapped:
        raise TaxonomyError(
            f"Source {source!r} has classes with no remap entry: {unmapped}. Map each to a "
            f"canonical class {list(CANONICAL_CLASSES)} or to None (drop) in "
            f"taxonomy.SOURCE_REMAPS[{source!r}]."
        )


def parse_class_list(text: str) -> tuple[str, ...]:
    """Parse a CLI value such as `player,referee`."""
    return select_classes(part.strip() for part in text.split(",") if part.strip())


def select_classes(classes: Iterable[str]) -> tuple[str, ...]:
    """Validate a dataset's class selection: canonical names, no duplicates, canonical order."""
    selected = tuple(classes)
    if not selected:
        raise TaxonomyError("Select at least one class")
    unknown = [name for name in selected if name not in CLASS_ID]
    if unknown:
        raise TaxonomyError(f"Not canonical classes: {unknown}; choose from {CANONICAL_CLASSES}")
    if len(set(selected)) != len(selected):
        raise TaxonomyError(f"Duplicate classes in selection: {list(selected)}")
    ordered = tuple(sorted(selected, key=CLASS_ID.__getitem__))
    if ordered != selected:
        raise TaxonomyError(
            f"Classes must be listed in canonical order {list(ordered)}, got {list(selected)}"
        )
    return selected


def output_ids(classes: Iterable[str]) -> dict[str, int]:
    """Canonical name -> class id in a dataset that selected `classes`."""
    return {name: index for index, name in enumerate(select_classes(classes))}


def is_canonical_prefix(classes: Iterable[str]) -> bool:
    selected = select_classes(classes)
    return selected == CANONICAL_CLASSES[: len(selected)]


def class_map(source: str, names: Sequence[str], classes: Iterable[str]) -> dict[int, int | None]:
    """Source class id -> output class id (None = dropped). Raises on unmapped classes."""
    validate_source_classes(source, names)
    table = source_remap(source)
    ids = output_ids(classes)
    return {
        index: (None if table[name] is None else ids.get(table[name]))
        for index, name in enumerate(names)
    }
