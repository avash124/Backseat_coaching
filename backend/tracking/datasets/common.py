"""Shared dataset plumbing: naming convention, label remapping, data.yaml, dataset hash.

Every builder (prepare_ebard, own_footage, merge, roboflow) writes the same layout:

    <name>/data.yaml                  path: /datasets/<name> (the Modal volume mount)
    <name>/{train,val,test}/images/   frames
    <name>/{train,val,test}/labels/   YOLO txt, same stem as the image
    <name>/train.txt                  optional image list (merge.py oversampling)
    <name>/dataset_manifest.json      provenance; its dataset_sha256 = dataset_sha256(<name>)

`dataset_sha256` is THE dataset identity used by training runs and the weights registry.
It covers data.yaml bytes, every image's bytes and every label's bytes (ordered by
stem), and any image-list files data.yaml references. It ignores everything else, in
particular the `labels.cache` files Ultralytics writes next to label directories.
The algorithm is byte-compatible with the original prepare_ebard digest.
"""

import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import yaml

MANIFEST_NAME = "dataset_manifest.json"
SPLITS = ("train", "val", "test")
IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png"})
# Formats Ultralytics would also train on. Refuse them rather than train on unhashed files.
_UNHASHED_IMAGE_SUFFIXES = frozenset(
    {".bmp", ".dng", ".mpo", ".tif", ".tiff", ".webp", ".pfm", ".heic", ".avif", ".jp2"}
)

# Shared with the labeling tools: split unit = group (a game, or a game segment).
GROUP_ID_PATTERN = r"[a-z0-9-]+"
GROUP_ID_RE = re.compile(GROUP_ID_PATTERN)
FRAME_STEM_RE = re.compile(rf"(?P<group>{GROUP_ID_PATTERN})_(?P<frame>\d{{7}})")


def validate_group_id(group: str) -> str:
    if not isinstance(group, str) or not GROUP_ID_RE.fullmatch(group):
        raise ValueError(f"Invalid group id {group!r}; expected {GROUP_ID_PATTERN}")
    return group


def frame_stem(group: str, frame_index: int) -> str:
    """`{group_id}_{frame_index:07d}` — the stem of an own-footage frame and its label."""
    validate_group_id(group)
    if not 0 <= frame_index <= 9_999_999:
        raise ValueError(f"Frame index out of range: {frame_index}")
    return f"{group}_{frame_index:07d}"


def parse_frame_stem(stem: str) -> tuple[str, int]:
    """Inverse of `frame_stem`; raises on anything that does not follow the convention."""
    match = FRAME_STEM_RE.fullmatch(stem)
    if match is None:
        raise ValueError(
            f"{stem!r} does not follow the naming convention {{group_id}}_{{frame:07d}} "
            f"with group_id matching {GROUP_ID_PATTERN}"
        )
    return match["group"], int(match["frame"])


def file_sha256(path: Path) -> str:
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


@dataclass
class RemappedLabels:
    text: str
    counts: Counter = field(default_factory=Counter)  # output class id -> boxes
    dropped: Counter = field(default_factory=Counter)  # source class id -> boxes
    excluded: list = field(default_factory=list)


def remap_label_text(
    raw: str, class_map: dict[int, int | None], *, label_ref: str
) -> RemappedLabels:
    """Validate one YOLO label file and rewrite its class ids.

    Coordinates are kept as the original strings (no float round-trip), so a remap that
    maps every id to itself reproduces the input byte for byte. Zero-area boxes are
    excluded and reported; anything malformed raises.
    """
    result = RemappedLabels(text="")
    lines = []
    for line_number, line in enumerate(raw.splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"Expected five label fields: {label_ref}:{line_number}")
        try:
            class_id = int(fields[0])
            coordinates = [float(value) for value in fields[1:]]
        except ValueError:
            raise ValueError(f"Non-numeric label field: {label_ref}:{line_number}") from None
        if class_id not in class_map:
            raise ValueError(f"Unknown class {class_id}: {label_ref}:{line_number}")
        if not all(math.isfinite(value) and 0 <= value <= 1 for value in coordinates):
            raise ValueError(f"Invalid normalized coordinates: {label_ref}:{line_number}")
        if coordinates[2] == 0 or coordinates[3] == 0:
            result.excluded.append({
                "file": label_ref,
                "line": line_number,
                "annotation": line,
                "reason": "zero-area box",
            })
            continue
        target = class_map[class_id]
        if target is None:
            result.dropped[class_id] += 1
            continue
        lines.append(" ".join([str(target), *fields[1:]]))
        result.counts[target] += 1
    result.text = "\n".join(lines) + ("\n" if lines else "")
    return result


def data_yaml_text(
    dataset_name: str,
    names: list[str],
    *,
    train: str = "train/images",
    val: str = "val/images",
    test: str | None = "test/images",
) -> str:
    config = {"path": f"/datasets/{dataset_name}", "train": train, "val": val}
    if test is not None:
        config["test"] = test
    config["names"] = list(names)
    return yaml.safe_dump(config, sort_keys=False)


def load_data_yaml(directory: Path) -> dict:
    config = yaml.safe_load((Path(directory) / "data.yaml").read_text(encoding="utf-8"))
    if not isinstance(config, dict) or "names" not in config:
        raise ValueError(f"{directory}/data.yaml has no class names")
    return config


def dataset_names(config: dict) -> list[str]:
    """`names` from a data.yaml as an id-ordered list (accepts list or {id: name})."""
    names = config["names"]
    if isinstance(names, dict):
        ids = sorted(int(key) for key in names)
        if ids != list(range(len(ids))):
            raise ValueError(f"Class ids must be contiguous from 0: {ids}")
        return [str(names[key]) if key in names else str(names[str(key)]) for key in ids]
    return [str(name) for name in names]


def warn_missing_provenance(source_version: str | None, license: str | None) -> list[str]:
    """Loud console warning when a builder records a null version or license."""
    missing = [name for name, value in (("--source-version", source_version),
                                        ("--license", license)) if not value]
    if missing:
        bar = "!" * 78
        print(
            f"{bar}\nWARNING: {', '.join(missing)} not set; the manifest records null.\n"
            f"Unknown license is NOT permissive. Resolve it (sources.yaml) before this data\n"
            f"trains a model that ships.\n{bar}",
            flush=True,
        )
    return missing


def read_manifest(directory: Path) -> dict:
    path = Path(directory) / MANIFEST_NAME
    if not path.is_file():
        raise FileNotFoundError(f"No {MANIFEST_NAME} in {directory}; build it with tracking.datasets")
    return json.loads(path.read_text(encoding="utf-8"))


def write_manifest(directory: Path, manifest: dict) -> Path:
    path = Path(directory) / MANIFEST_NAME
    # Same call as the original prepare_ebard (platform newlines) so its default output
    # stays byte-identical. The manifest is not part of the dataset hash.
    path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return path


def list_files(config: dict) -> list[str]:
    """Image-list files (e.g. train.txt) referenced by a data.yaml, root-relative."""
    found = set()
    for key in SPLITS:
        value = config.get(key)
        values = value if isinstance(value, list) else [value]
        for item in values:
            if isinstance(item, str) and item.endswith(".txt"):
                if Path(item).is_absolute() or ".." in Path(item).parts:
                    raise ValueError(f"Image list must be inside the dataset: {item}")
                found.add(Path(item).as_posix())
    return sorted(found)


def iter_dataset_images(directory: Path) -> list[tuple[str, str, Path]]:
    """(stem, split, image path) for every image in `<split>/images/`, sorted by stem."""
    directory = Path(directory)
    records = []
    for split_dir in sorted(p for p in directory.iterdir() if (p / "images").is_dir()):
        for image in (split_dir / "images").iterdir():
            if not image.is_file():
                continue
            suffix = image.suffix.lower()
            if suffix in IMAGE_SUFFIXES:
                records.append((image.stem, split_dir.name, image))
            elif suffix in _UNHASHED_IMAGE_SUFFIXES:
                raise ValueError(f"Unsupported image format would be trained on unhashed: {image}")
    stems = Counter(stem for stem, _, _ in records)
    duplicates = sorted(stem for stem, count in stems.items() if count > 1)
    if duplicates:
        raise ValueError(f"Duplicate image stems across splits: {duplicates[:10]}")
    return sorted(records)


def dataset_sha256(directory: Path) -> str:
    """Content hash of a prepared dataset (see module docstring for what it covers)."""
    directory = Path(directory)
    data_yaml = (directory / "data.yaml").read_bytes()
    digest = hashlib.sha256(data_yaml)
    for stem, split, image in iter_dataset_images(directory):
        label = directory / split / "labels" / f"{stem}.txt"
        if not label.is_file():
            raise FileNotFoundError(f"Missing label for {image}")
        digest.update(f"{split}/{image.name}\0{file_sha256(image)}\0".encode())
        digest.update(label.read_bytes() + b"\0")
    for name in list_files(yaml.safe_load(data_yaml)):
        digest.update(f"list/{name}\0".encode() + (directory / name).read_bytes() + b"\0")
    return digest.hexdigest()


def verify_dataset(directory: Path, expected_sha256: str | None = None) -> dict:
    """Recompute the hash; raise if it disagrees with the manifest or `expected_sha256`."""
    directory = Path(directory)
    manifest = read_manifest(directory)
    actual = dataset_sha256(directory)
    recorded = manifest.get("dataset_sha256")
    if recorded != actual:
        raise ValueError(
            f"{directory} changed after it was built: manifest says {recorded}, content is {actual}"
        )
    if expected_sha256 is not None and expected_sha256 != actual:
        raise ValueError(f"{directory} is dataset {actual}, but {expected_sha256} was requested")
    return manifest


def count_boxes(directory: Path) -> dict[str, dict[str, int]]:
    """Per split, per class name, box counts read back from disk (unique images)."""
    directory = Path(directory)
    names = dataset_names(load_data_yaml(directory))
    counts: dict[str, Counter] = {}
    for stem, split, _ in iter_dataset_images(directory):
        text = (directory / split / "labels" / f"{stem}.txt").read_text(encoding="utf-8")
        split_counts = counts.setdefault(split, Counter())
        for line in text.splitlines():
            if line.strip():
                split_counts[names[int(line.split()[0])]] += 1
    return {split: dict(sorted(c.items())) for split, c in sorted(counts.items())}
