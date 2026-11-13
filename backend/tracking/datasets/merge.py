"""Combine prepared datasets (public + own) into one YOLO dataset directory + data.yaml.

Also where per-source sample weighting lives: public broadcast footage should not
outvote our own tripod frames just because there is more of it. Weighting is explicit
oversampling: own-footage train frames are listed `--oversample-own` times in
`train.txt` (Ultralytics keeps duplicate list entries), and the factor is recorded in
the manifest rather than buried in code.

Run from backend:
    python -m tracking.datasets.merge --public ../datasets/ebard/players_v1 \
        --own ../datasets/own/players_own_v1 --oversample-own 3 \
        --destination ../datasets/merged/players_merged_v1

Policy:
- Every source must be a dataset built by tracking.datasets (manifest + matching hash).
- Each source keeps its own group-wise split. Own-footage train/val/test are kept;
  public train is kept, public val only with `--public-val keep` (default drop: best.pt is
  selected on our domain), public test never (the test split is our footage only).
- Every source must label every selected class; missing classes would become background.
- Groups are re-checked across sources (no group in two splits, quarantine).
"""

import argparse
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from tracking.datasets import taxonomy
from tracking.datasets.common import (
    data_yaml_text,
    dataset_names,
    dataset_sha256,
    iter_dataset_images,
    load_data_yaml,
    remap_label_text,
    verify_dataset,
    write_manifest,
)
from tracking.datasets.splits import check_splits, load_quarantine, quarantine_record

TRAIN_LIST = "train.txt"


@dataclass(frozen=True)
class MergeSource:
    path: Path
    role: Literal["public", "own"]


def _kept_splits(role: str, public_val: str) -> tuple[str, ...]:
    if role == "own":
        return ("train", "val", "test")
    return ("train", "val") if public_val == "keep" else ("train",)


def merge(
    sources: list[MergeSource],
    destination: Path,
    *,
    classes: tuple[str, ...] | None = None,
    oversample_own: int = 1,
    public_val: Literal["drop", "keep"] = "drop",
    quarantine_path: Path | None = None,
) -> dict:
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    if oversample_own < 1:
        raise ValueError("oversample_own must be >= 1")
    if public_val not in ("drop", "keep"):
        raise ValueError("public_val must be 'drop' or 'keep'")
    if not any(source.role == "own" for source in sources):
        raise ValueError("Merge needs at least one own-footage source (val/test come from it)")

    loaded = []
    for source in sources:
        path = Path(source.path).resolve()
        config = load_data_yaml(path)
        if any(isinstance(config.get(key), str) and config[key].endswith(".txt")
               for key in ("train", "val", "test")):
            raise ValueError(f"{path} is itself list-based (merged); merge its inputs instead")
        manifest = verify_dataset(path)
        loaded.append((source, path, dataset_names(config), manifest))

    if classes is None:
        classes = tuple(next(names for s, _, names, _ in loaded if s.role == "own"))
    names = list(taxonomy.select_classes(classes))
    ids = taxonomy.output_ids(names)
    for _, path, source_names, _ in loaded:
        taxonomy.select_classes(source_names)  # sources must already be canonical
        missing = [name for name in names if name not in source_names]
        if missing:
            raise ValueError(
                f"{path} does not label {missing}; merged frames from it would teach the "
                f"model those objects are background"
            )

    splits: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}
    copies = []  # (source index, split, image, label)
    for index, (source, path, _, manifest) in enumerate(loaded):
        kept = _kept_splits(source.role, public_val)
        source_groups = manifest.get("groups") or manifest.get("games") or {}
        for split in kept:
            splits[split].update(source_groups.get(split, []))
        for stem, split, image in iter_dataset_images(path):
            if split in kept:
                copies.append((index, split, image, path / split / "labels" / f"{stem}.txt"))
    check_splits(splits, load_quarantine(quarantine_path))
    stems = Counter(image.stem for _, _, image, _ in copies)
    collisions = sorted(stem for stem, count in stems.items() if count > 1)
    if collisions:
        raise ValueError(f"Frame stems collide across sources: {collisions[:10]}")
    if not any(split == "val" for _, split, _, _ in copies):
        raise ValueError("Merged dataset has no val frames")

    destination.mkdir(parents=True)
    for split in splits:
        (destination / split / "images").mkdir(parents=True)
        (destination / split / "labels").mkdir()
    train_lines = []
    images = Counter()
    boxes = {split: Counter() for split in splits}
    effective_train = Counter()
    per_source = [Counter() for _ in loaded]
    for index, split, image, label in sorted(copies, key=lambda c: (c[1], c[2].name)):
        source, path, source_names, _ = loaded[index]
        class_map = {i: ids.get(name) for i, name in enumerate(source_names)}
        remapped = remap_label_text(
            label.read_text(encoding="utf-8"), class_map, label_ref=str(label)
        )
        shutil.copy2(image, destination / split / "images" / image.name)
        (destination / split / "labels" / f"{image.stem}.txt").write_text(
            remapped.text, encoding="utf-8", newline="\n"
        )
        images[split] += 1
        per_source[index][split] += 1
        counts = Counter({names[c]: n for c, n in remapped.counts.items()})
        boxes[split].update(counts)
        if split == "train":
            factor = oversample_own if source.role == "own" else 1
            train_lines += [f"./train/images/{image.name}"] * factor
            effective_train.update({name: n * factor for name, n in counts.items()})
    (destination / TRAIN_LIST).write_text(
        "\n".join(sorted(train_lines)) + "\n", encoding="utf-8", newline="\n"
    )
    (destination / "data.yaml").write_text(
        data_yaml_text(destination.name, names, train=TRAIN_LIST,
                       test="test/images" if images["test"] else None),
        encoding="utf-8", newline="\n",
    )
    manifest = {
        "kind": "merged",
        "builder": "tracking.datasets.merge",
        "sources": [
            {
                "path": str(path),
                "role": source.role,
                "dataset_sha256": manifest["dataset_sha256"],
                "names": source_names,
                "source_version": manifest.get("source_version"),
                "source_license": manifest.get("source_license"),
                "splits_kept": list(_kept_splits(source.role, public_val)),
                "images": dict(per_source[index]),
            }
            for index, (source, path, source_names, manifest) in enumerate(loaded)
        ],
        "names": names,
        "oversampling": {"own": oversample_own, "public": 1},
        "public_val": public_val,
        "split_unit": "group",
        "quarantine": quarantine_record(quarantine_path),
        "groups": {split: sorted(groups) for split, groups in splits.items()},
        "images": dict(images),
        "train_list_entries": len(train_lines),
        "boxes": {split: dict(sorted(c.items())) for split, c in boxes.items()},
        "effective_train_boxes": dict(sorted(effective_train.items())),
        "dataset_sha256": dataset_sha256(destination),
    }
    write_manifest(destination, manifest)
    return manifest


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--public", type=Path, action="append", default=[])
    parser.add_argument("--own", type=Path, action="append", default=[])
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--classes", type=taxonomy.parse_class_list, default=None)
    parser.add_argument("--oversample-own", type=int, default=1)
    parser.add_argument("--public-val", choices=("drop", "keep"), default="drop")
    parser.add_argument("--quarantine", type=Path, default=None)
    args = parser.parse_args(argv)
    sources = [MergeSource(p, "public") for p in args.public]
    sources += [MergeSource(p, "own") for p in args.own]
    manifest = merge(
        sources, args.destination, classes=args.classes, oversample_own=args.oversample_own,
        public_val=args.public_val, quarantine_path=args.quarantine,
    )
    print(f"Merged into {args.destination.resolve()}")
    print(f"Images: {manifest['images']}; train list entries: {manifest['train_list_entries']}")
    print(f"Effective train boxes: {manifest['effective_train_boxes']}")
    print(f"Dataset SHA256: {manifest['dataset_sha256']}")
    return manifest


if __name__ == "__main__":
    main()
