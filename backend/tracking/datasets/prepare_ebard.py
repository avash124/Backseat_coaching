"""Prepare the downloaded eBAR YOLO export for detector training.

Run from backend:
    python -m tracking.datasets.prepare_ebard                       # players_v1 (player, referee)
    python -m tracking.datasets.prepare_ebard --classes player,referee,basketball,hoop \
        --destination ../datasets/ebard/full_v1 \
        --source-version "hf:GabrieleGiudici/E-BARD-detection@00563215490c9a9642797b1495ea178535b3f59c" \
        --license "labels CC BY 4.0; frame rights unresolved (NBA broadcast) — internal experiments only, no shipped weights without legal sign-off"

The original export is unchanged. The destination must not already exist. Classes go
through `taxonomy.py` (source table "ebard"); output ids follow the selected classes in
canonical order, so the default reproduces players_v1 byte for byte.

The export's own train/val/test assignment is ignored (its data.yaml even swaps the val
and test paths): every frame is pooled and re-split by whole game, 80/10/10.
Public validation/test images are diagnostic only; promotion needs our own footage.
`--source-version` / `--license` are recorded verbatim in the manifest (default null, with a
loud warning). The verified eBAR values are in sources.yaml and shown above: labels are
CC BY 4.0 but the NBA broadcast frames have no stated rights, so weights trained on this
data are internal experiments only until legal sign-off.
"""

import argparse
import shutil
from collections import Counter
from pathlib import Path

from tracking.datasets import taxonomy
from tracking.datasets.common import (
    IMAGE_SUFFIXES,
    data_yaml_text,
    dataset_sha256,
    load_data_yaml,
    remap_label_text,
    warn_missing_provenance,
    write_manifest,
)
from tracking.datasets.splits import load_quarantine, split_groups

REPO_ROOT = Path(__file__).resolve().parents[3]
SOURCE = "ebard"
SOURCE_NAMES = ["basketball", "hoop", "player", "referee"]
TARGET_NAMES = list(taxonomy.DEFAULT_CLASSES)


def prepare(
    source: Path,
    destination: Path,
    seed: int = 42,
    classes: tuple[str, ...] = taxonomy.DEFAULT_CLASSES,
    source_version: str | None = None,
    license: str | None = None,
    quarantine_path: Path | None = None,
) -> dict:
    source = source.resolve()
    destination = destination.resolve()
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")

    config = load_data_yaml(source)
    if config["names"] != SOURCE_NAMES:
        raise ValueError("Source class order changed; review the eBAR remap before proceeding")
    names = list(taxonomy.select_classes(classes))
    class_map = taxonomy.class_map(SOURCE, SOURCE_NAMES, names)

    records = {}
    excluded = []
    for original_split in ("train", "val", "test"):
        images = sorted(
            p for p in (source / original_split / "images").iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
        )
        label_dir = source / original_split / "labels"
        orphan_labels = {p.stem for p in label_dir.glob("*.txt")} - {p.stem for p in images}
        if orphan_labels:
            raise ValueError(f"Orphan labels in {original_split}: {sorted(orphan_labels)}")
        for image in images:
            if image.stem in records:
                raise ValueError(f"Duplicate image stem across source splits: {image.stem}")
            game, separator, frame = image.stem.rpartition("_")
            if not game or not separator or not frame.isdigit():
                raise ValueError(f"Cannot infer game ID from {image.name}")
            label = label_dir / f"{image.stem}.txt"
            remapped = remap_label_text(
                label.read_text(encoding="utf-8"),
                class_map,
                label_ref=label.relative_to(source).as_posix(),
            )
            excluded.extend(remapped.excluded)
            records[image.stem] = (image, game, remapped.text, remapped.counts)

    games = sorted({record[1] for record in records.values()})
    if len(games) < 3:
        raise ValueError("At least three games are required for disjoint splits")
    splits = split_groups(games, seed=seed, quarantine=load_quarantine(quarantine_path))
    game_to_split = {game: split for split, group in splits.items() for game in group}
    image_counts = Counter()
    class_counts = {split: Counter() for split in splits}

    destination.mkdir(parents=True)
    for split in splits:
        (destination / split / "images").mkdir(parents=True)
        (destination / split / "labels").mkdir()
    for stem, (image, game, labels, counts) in sorted(records.items()):
        split = game_to_split[game]
        shutil.copy2(image, destination / split / "images" / image.name)
        (destination / split / "labels" / f"{stem}.txt").write_text(
            labels, encoding="utf-8", newline="\n"
        )
        image_counts[split] += 1
        class_counts[split].update(counts)
    (destination / "data.yaml").write_text(
        data_yaml_text(destination.name, names), encoding="utf-8", newline="\n"
    )
    manifest = {
        "source_directory": str(source),
        "source_version": source_version,
        "source_license": license,
        "seed": seed,
        "split_policy": "whole games, 80/10/10",
        "evaluation_domain": "public NBA footage; not the product evaluation set",
        "class_map": class_map,
        "names": names,
        "games": {split: sorted(group) for split, group in splits.items()},
        "images": dict(image_counts),
        "boxes": {split: dict(counts) for split, counts in class_counts.items()},
        "excluded_annotations": excluded,
        "dataset_sha256": dataset_sha256(destination),
    }
    write_manifest(destination, manifest)
    return manifest


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, default=REPO_ROOT / "datasets/ebard/extracted/yolo")
    parser.add_argument("--destination", type=Path, default=REPO_ROOT / "datasets/ebard/players_v1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--classes", type=taxonomy.parse_class_list, default=taxonomy.DEFAULT_CLASSES,
        help="comma-separated canonical classes in canonical order (default: player,referee)",
    )
    parser.add_argument("--source-version", default=None, help="pinned eBAR version/revision")
    parser.add_argument("--license", default=None, help="eBAR license identifier")
    parser.add_argument("--quarantine", type=Path, default=None,
                        help="override evalsets/quarantine.yaml")
    args = parser.parse_args(argv)
    warn_missing_provenance(args.source_version, args.license)
    manifest = prepare(
        args.source, args.destination, args.seed, args.classes,
        args.source_version, args.license, args.quarantine,
    )
    print(f"Prepared {args.destination.resolve()}")
    print(f"Classes: {manifest['names']}")
    print(f"Images: {manifest['images']}")
    print(f"Boxes: {manifest['boxes']}")
    print(f"Excluded zero-area annotations: {len(manifest['excluded_annotations'])}")
    print(f"Dataset SHA256: {manifest['dataset_sha256']}")
    return manifest


if __name__ == "__main__":
    main()
