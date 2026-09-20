"""Prepare the downloaded eBAR YOLO export for the player/referee experiment.

Run from backend: python -m tracking.datasets.prepare_ebard
The original export is unchanged. The destination must not already exist.
Public validation/test images are diagnostic only; promotion needs our own footage.
"""

import argparse
import hashlib
import json
import math
import random
import shutil
from collections import Counter
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
SOURCE_NAMES = ["basketball", "hoop", "player", "referee"]
TARGET_NAMES = ["player", "referee"]
CLASS_MAP = {0: None, 1: None, 2: 0, 3: 1}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def prepare(source: Path, destination: Path, seed: int = 42) -> dict:
    source = source.resolve()
    destination = destination.resolve()
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")

    config = yaml.safe_load((source / "data.yaml").read_text(encoding="utf-8"))
    if config["names"] != SOURCE_NAMES:
        raise ValueError("Source class order changed; review CLASS_MAP before proceeding")

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
            raw_labels = label.read_text(encoding="utf-8")
            remapped = []
            counts = Counter()
            for line_number, line in enumerate(raw_labels.splitlines(), 1):
                if not line.strip():
                    continue
                fields = line.split()
                if len(fields) != 5:
                    raise ValueError(f"Expected five label fields: {label}:{line_number}")
                class_id = int(fields[0])
                if class_id not in CLASS_MAP:
                    raise ValueError(f"Unknown class {class_id}: {label}:{line_number}")
                coordinates = [float(value) for value in fields[1:]]
                if not all(math.isfinite(value) and 0 <= value <= 1 for value in coordinates):
                    raise ValueError(f"Invalid normalized coordinates: {label}:{line_number}")
                if coordinates[2] == 0 or coordinates[3] == 0:
                    excluded.append({
                        "file": label.relative_to(source).as_posix(),
                        "line": line_number,
                        "annotation": line,
                        "reason": "zero-area box",
                    })
                    continue
                target = CLASS_MAP[class_id]
                if target is not None:
                    remapped.append(" ".join([str(target), *fields[1:]]))
                    counts[target] += 1
            text = "\n".join(remapped) + ("\n" if remapped else "")
            records[image.stem] = (image, game, text, counts)

    games = sorted({record[1] for record in records.values()})
    if len(games) < 3:
        raise ValueError("At least three games are required for disjoint splits")
    random.Random(seed).shuffle(games)
    held_out_count = max(1, len(games) // 10)
    splits = {
        "train": games[2 * held_out_count:],
        "val": games[:held_out_count],
        "test": games[held_out_count:2 * held_out_count],
    }
    game_to_split = {game: split for split, group in splits.items() for game in group}
    image_counts = Counter()
    class_counts = {split: Counter() for split in splits}

    destination.mkdir(parents=True)
    for split in splits:
        (destination / split / "images").mkdir(parents=True)
        (destination / split / "labels").mkdir()
    data_yaml = yaml.safe_dump({
        "path": f"/datasets/{destination.name}",
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "names": TARGET_NAMES,
    }, sort_keys=False)
    digest = hashlib.sha256(data_yaml.encode())
    for stem, (image, game, labels, counts) in sorted(records.items()):
        split = game_to_split[game]
        output_image = destination / split / "images" / image.name
        shutil.copy2(image, output_image)
        (destination / split / "labels" / f"{stem}.txt").write_text(
            labels, encoding="utf-8", newline="\n"
        )
        with output_image.open("rb") as handle:
            image_hash = hashlib.file_digest(handle, "sha256").hexdigest()
        digest.update(f"{split}/{image.name}\0{image_hash}\0{labels}\0".encode())
        image_counts[split] += 1
        class_counts[split].update(counts)
    (destination / "data.yaml").write_text(data_yaml, encoding="utf-8", newline="\n")
    manifest = {
        "source_directory": str(source),
        "source_version": None,
        "source_license": None,
        "seed": seed,
        "split_policy": "whole games, 80/10/10",
        "evaluation_domain": "public NBA footage; not the product evaluation set",
        "class_map": CLASS_MAP,
        "names": TARGET_NAMES,
        "games": {split: sorted(group) for split, group in splits.items()},
        "images": dict(image_counts),
        "boxes": {split: dict(counts) for split, counts in class_counts.items()},
        "excluded_annotations": excluded,
        "dataset_sha256": digest.hexdigest(),
    }
    (destination / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=REPO_ROOT / "datasets/ebard/extracted/yolo")
    parser.add_argument("--destination", type=Path, default=REPO_ROOT / "datasets/ebard/players_v1")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    manifest = prepare(args.source, args.destination, args.seed)
    print(f"Prepared {args.destination.resolve()}")
    print(f"Images: {manifest['images']}")
    print(f"Excluded zero-area annotations: {len(manifest['excluded_annotations'])}")
    print(f"Dataset SHA256: {manifest['dataset_sha256']}")
