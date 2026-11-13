"""Build a YOLO dataset from our own corrected labels.

Inputs (`--input`, repeatable) are either:
- CVAT "YOLO 1.1" exports: `obj.names` (class order), `obj.data`, `train.txt`, and
  `obj_train_data/*.txt` (+ the images when exported with "save images"), or
- plain folders of images + YOLO labels (any layout), with class names in `obj.names`,
  `classes.txt` or `data.yaml`; with none of those the labels must already use the
  canonical class ids (taxonomy.CANONICAL_CLASSES).

Frames that were exported without images are found in `--images` folders (the frames the
labeling tools extracted). Every file must follow the shared naming convention
`{group_id}_{frame_index:07d}.jpg` + `.txt` with the same stem; group = game (or game
segment). An empty label file is an explicit "nothing here" frame; a labeled input frame
with no label file is an error, never a silent background frame.

Run from backend:
    python -m tracking.datasets.own_footage --input ../exports/game1-cvat \
        --images ../frames --destination ../datasets/own/players_own_v1 --license internal

Output: the standard layout (see common.py) with groups split by `splits.py` (seeded,
quarantine enforced) and a `dataset_manifest.json` recording inputs, per-class counts,
groups per split and the dataset sha256.
"""

import argparse
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import yaml

from tracking.datasets import taxonomy
from tracking.datasets.common import (
    IMAGE_SUFFIXES,
    data_yaml_text,
    dataset_names,
    dataset_sha256,
    file_sha256,
    parse_frame_stem,
    remap_label_text,
    warn_missing_provenance,
    write_manifest,
)
from tracking.datasets.splits import load_quarantine, quarantine_record, split_groups

SOURCE = "own-footage"
NAMES_FILES = ("obj.names", "classes.txt", "data.yaml")
# Export metadata that is not a frame label.
METADATA_FILES = frozenset({
    "obj.data", "obj.names", "classes.txt", "data.yaml", "dataset_manifest.json",
    "train.txt", "val.txt", "valid.txt", "test.txt",
})
OS_JUNK = frozenset({"thumbs.db", "desktop.ini", ".ds_store"})


@dataclass(frozen=True)
class LabelInput:
    root: Path
    kind: str
    names: tuple[str, ...]
    names_file: str | None


def read_input_names(root: Path) -> LabelInput:
    for filename in NAMES_FILES:
        path = root / filename
        if not path.is_file():
            continue
        if filename == "data.yaml":
            names = dataset_names(yaml.safe_load(path.read_text(encoding="utf-8")))
            kind = "yolo-data-yaml"
        else:
            names = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
            names = [name for name in names if name]
            kind = "cvat-yolo-1.1" if filename == "obj.names" else "yolo-classes-txt"
        return LabelInput(root, kind, tuple(names), filename)
    return LabelInput(root, "canonical-ids", taxonomy.CANONICAL_CLASSES, None)


def _scan(root: Path, *, labels: bool) -> tuple[dict[str, Path], dict[str, Path], list[str]]:
    """(label stem -> path, image stem -> path, naming problems) under `root`."""
    found_labels: dict[str, Path] = {}
    found_images: dict[str, Path] = {}
    problems = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name.lower() in OS_JUNK:
            continue
        suffix = path.suffix.lower()
        is_label = labels and suffix == ".txt" and path.name not in METADATA_FILES
        if not is_label and suffix not in IMAGE_SUFFIXES:
            continue
        try:
            parse_frame_stem(path.stem)
        except ValueError as error:
            problems.append(f"{path}: {error}")
            continue
        target = found_labels if is_label else found_images
        if path.stem in target:
            problems.append(f"{path}: duplicate stem (also {target[path.stem]})")
            continue
        target[path.stem] = path
    return found_labels, found_images, problems


def _fail(message: str, items: list[str]) -> None:
    if items:
        shown = "\n  ".join(items[:20])
        more = f"\n  ... and {len(items) - 20} more" if len(items) > 20 else ""
        raise ValueError(f"{message} ({len(items)}):\n  {shown}{more}")


def collect(
    inputs: list[Path], image_dirs: list[Path] = ()
) -> tuple[dict[str, tuple[Path, Path, LabelInput]], list[LabelInput]]:
    """stem -> (image, label, input) for every labeled frame, after strict validation."""
    label_inputs = [read_input_names(Path(root).resolve()) for root in inputs]
    labels: dict[str, tuple[Path, LabelInput]] = {}
    images: dict[str, Path] = {}
    problems, conflicts, unlabeled = [], [], []

    def add_image(stem: str, path: Path):
        if stem in images and images[stem] != path:
            if file_sha256(images[stem]) != file_sha256(path):
                conflicts.append(f"{stem}: {images[stem]} differs from {path}")
            return
        images[stem] = path

    for label_input in label_inputs:
        if not label_input.root.is_dir():
            raise FileNotFoundError(label_input.root)
        found_labels, found_images, found_problems = _scan(label_input.root, labels=True)
        problems += found_problems
        for stem, path in found_labels.items():
            if stem in labels:
                conflicts.append(f"{stem}: labeled in {labels[stem][0]} and {path}")
            labels[stem] = (path, label_input)
        for stem, path in found_images.items():
            add_image(stem, path)
            if stem not in found_labels:
                unlabeled.append(str(path))
    for directory in image_dirs:
        directory = Path(directory).resolve()
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        _, found_images, found_problems = _scan(directory, labels=False)
        problems += found_problems
        for stem, path in found_images.items():
            add_image(stem, path)

    _fail("Files that break the {group_id}_{frame:07d} naming convention", problems)
    _fail("Conflicting duplicate frames", conflicts)
    _fail("Frames in a labeled export without a label file (empty .txt = no objects)", unlabeled)
    _fail("Labels without an image (pass the frames folder with --images)",
          sorted(str(path) for stem, (path, _) in labels.items() if stem not in images))
    if not labels:
        raise ValueError(f"No labeled frames found in {[str(p) for p in inputs]}")
    frames = {stem: (images[stem], path, li) for stem, (path, li) in sorted(labels.items())}
    return frames, label_inputs


def build(
    inputs: list[Path],
    destination: Path,
    *,
    image_dirs: list[Path] = (),
    classes: tuple[str, ...] = taxonomy.DEFAULT_CLASSES,
    seed: int = 42,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    quarantine_path: Path | None = None,
    source_version: str | None = None,
    license: str | None = None,
) -> dict:
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    names = list(taxonomy.select_classes(classes))
    frames, label_inputs = collect(inputs, image_dirs)
    class_maps = {li.root: taxonomy.class_map(SOURCE, li.names, names) for li in label_inputs}
    table = taxonomy.source_remap(SOURCE)
    for li in label_inputs:
        missing = [name for name in names if name not in {table[n] for n in li.names}]
        if missing:
            raise ValueError(
                f"{li.root} does not label {missing}; selecting them would teach the model "
                f"that every {missing[0]} in these frames is background. Drop them from "
                f"--classes or label them."
            )

    records = {}
    excluded = []
    dropped = Counter()
    frames_per_group = Counter()
    for stem, (image, label, label_input) in frames.items():
        group, _ = parse_frame_stem(stem)
        remapped = remap_label_text(
            label.read_text(encoding="utf-8"),
            class_maps[label_input.root],
            label_ref=str(label),
        )
        excluded.extend(remapped.excluded)
        dropped.update({label_input.names[i]: n for i, n in remapped.dropped.items()})
        records[stem] = (image, group, remapped.text, remapped.counts)
        frames_per_group[group] += 1

    quarantine = load_quarantine(quarantine_path)
    splits = split_groups(
        frames_per_group, seed=seed, val_fraction=val_fraction, test_fraction=test_fraction,
        quarantine=quarantine,
    )
    group_to_split = {group: split for split, groups in splits.items() for group in groups}

    destination.mkdir(parents=True)
    for split in splits:
        (destination / split / "images").mkdir(parents=True)
        (destination / split / "labels").mkdir()
    image_counts = {split: 0 for split in splits}
    box_counts = {split: Counter() for split in splits}
    for stem, (image, group, labels, counts) in records.items():
        split = group_to_split[group]
        shutil.copy2(image, destination / split / "images" / f"{stem}{image.suffix.lower()}")
        (destination / split / "labels" / f"{stem}.txt").write_text(
            labels, encoding="utf-8", newline="\n"
        )
        image_counts[split] += 1
        box_counts[split].update({names[class_id]: n for class_id, n in counts.items()})
    (destination / "data.yaml").write_text(
        data_yaml_text(destination.name, names, test="test/images" if splits["test"] else None),
        encoding="utf-8", newline="\n",
    )
    manifest = {
        "kind": "own-footage",
        "builder": "tracking.datasets.own_footage",
        "sources": [
            {
                "path": str(li.root),
                "kind": li.kind,
                "names_file": li.names_file,
                "names": list(li.names),
                "class_map": class_maps[li.root],
                "frames": sum(1 for *_, owner in frames.values() if owner.root == li.root),
            }
            for li in label_inputs
        ],
        "image_dirs": [str(Path(p).resolve()) for p in image_dirs],
        "source_version": source_version,
        "source_license": license,
        "seed": seed,
        "split_unit": "group",
        "split_policy": f"whole groups; val {val_fraction}, test {test_fraction}",
        "evaluation_domain": "own footage; quarantined eval groups excluded",
        "names": names,
        "quarantine": quarantine_record(quarantine_path),
        "groups": {split: sorted(groups) for split, groups in splits.items()},
        "frames_per_group": dict(sorted(frames_per_group.items())),
        "images": image_counts,
        "boxes": {split: dict(sorted(c.items())) for split, c in box_counts.items()},
        "dropped_boxes": dict(sorted(dropped.items())),
        "excluded_annotations": excluded,
        "oversampling": None,
        "dataset_sha256": dataset_sha256(destination),
    }
    write_manifest(destination, manifest)
    return manifest


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, action="append", required=True,
                        help="CVAT YOLO 1.1 export or images+labels folder (repeatable)")
    parser.add_argument("--images", type=Path, action="append", default=[],
                        help="folder of frames to resolve label stems against (repeatable)")
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--classes", type=taxonomy.parse_class_list,
                        default=taxonomy.DEFAULT_CLASSES,
                        help="canonical classes to keep, canonical order (default player,referee)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--quarantine", type=Path, default=None,
                        help="override evalsets/quarantine.yaml")
    parser.add_argument("--source-version", default=None)
    parser.add_argument("--license", default=None)
    args = parser.parse_args(argv)
    warn_missing_provenance(args.source_version, args.license)
    manifest = build(
        args.input, args.destination, image_dirs=args.images, classes=args.classes,
        seed=args.seed, val_fraction=args.val_fraction, test_fraction=args.test_fraction,
        quarantine_path=args.quarantine, source_version=args.source_version,
        license=args.license,
    )
    print(f"Built {args.destination.resolve()}")
    print(f"Groups: { {k: len(v) for k, v in manifest['groups'].items()} }")
    print(f"Images: {manifest['images']}")
    print(f"Boxes: {manifest['boxes']}")
    if manifest["dropped_boxes"]:
        print(f"Dropped (class not selected): {manifest['dropped_boxes']}")
    print(f"Dataset SHA256: {manifest['dataset_sha256']}")
    return manifest


if __name__ == "__main__":
    main()
