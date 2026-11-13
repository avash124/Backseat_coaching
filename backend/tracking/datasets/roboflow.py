"""Pull Roboflow datasets by pinned (workspace, project, version).

Always pin the version — Roboflow projects change under you, and an unpinned pull
makes a training run unreproducible. The API key comes from the `ROBOFLOW_API_KEY`
environment variable (a Modal Secret if this ever runs remotely), never a committed file.

Steps (`pull` runs all three):
1. License gate: an entry in sources.yaml with `license_status: unknown` is refused before
   anything is downloaded; `restricted` (internal experiments only) is pulled with a warning.
2. `download`: fetch the export zip into a content-addressed raw directory named by a
   hash of (workspace, project, version, format). Exists with its completion marker ⇒
   skip. Uses the REST export endpoint via urllib (no SDK dependency); the fetch function
   is injectable so tests never touch the network.
3. `import_export`: re-derive class ids by NAME through taxonomy.py (the export's
   data.yaml order is only used to name its ids, never trusted as our ids), recover game
   groups with the entry's `group_pattern` if it has one, and write the standard layout
   + manifest. Without a group pattern there is no game identity, so every frame goes to
   train (a frame-level split would leak) and the set must be merged with own footage.

Run from backend (network; needs ROBOFLOW_API_KEY):
    python -m tracking.datasets.roboflow --source <name in sources.yaml> \
        --destination ../datasets/<name>-v<version>
"""

import argparse
import hashlib
import io
import json
import os
import re
import shutil
import zipfile
from collections import Counter
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from urllib.parse import quote
from urllib.request import urlopen
from uuid import uuid4

from tracking.datasets import taxonomy
from tracking.datasets.common import (
    IMAGE_SUFFIXES,
    data_yaml_text,
    dataset_names,
    dataset_sha256,
    load_data_yaml,
    remap_label_text,
    validate_group_id,
    write_manifest,
)
from tracking.datasets.sources import (
    SOURCES_PATH,
    RoboflowCoordinates,
    SourceEntry,
    load_sources,
)
from tracking.datasets.splits import (
    check_splits,
    load_quarantine,
    quarantine_record,
    split_groups,
)

API_URL = "https://api.roboflow.com"
DOWNLOAD_MARKER = ".download.json"
DEFAULT_RAW_ROOT = Path(__file__).resolve().parents[3] / "datasets" / "roboflow-raw"
EXPORT_SPLITS = {"train": "train", "valid": "val", "val": "val", "test": "test"}
Fetch = Callable[[str], bytes]


class LicenseError(RuntimeError):
    """The source's license is unresolved; it must not be downloaded for training."""


def coordinates_key(coords: RoboflowCoordinates) -> str:
    text = f"{coords.workspace}/{coords.project}/{coords.version}/{coords.export_format}"
    return hashlib.sha256(text.encode()).hexdigest()


def raw_dir(root: Path, coords: RoboflowCoordinates) -> Path:
    return Path(root) / f"{coords.project}-v{coords.version}-{coordinates_key(coords)[:12]}"


def _http_fetch(url: str) -> bytes:
    with urlopen(url, timeout=120) as response:
        return response.read()


def _safe_extract(archive: bytes, target: Path) -> int:
    with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
        members = bundle.infolist()
        for member in members:
            parts = PurePosixPath(member.filename.replace("\\", "/")).parts
            if member.filename.startswith(("/", "\\")) or ".." in parts or (
                parts and ":" in parts[0]
            ):
                raise ValueError(f"Unsafe path in archive: {member.filename}")
        bundle.extractall(target)
    return len(members)


def download(
    coords: RoboflowCoordinates,
    root: Path = DEFAULT_RAW_ROOT,
    *,
    api_key: str | None = None,
    fetch: Fetch = _http_fetch,
) -> Path:
    target = raw_dir(root, coords)
    if (target / DOWNLOAD_MARKER).is_file():
        return target
    if target.exists():
        raise FileExistsError(f"{target} has no {DOWNLOAD_MARKER} (interrupted?); delete it")
    api_key = api_key or os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise RuntimeError("Set ROBOFLOW_API_KEY to download from Roboflow")
    url = (f"{API_URL}/{quote(coords.workspace)}/{quote(coords.project)}/{coords.version}/"
           f"{quote(coords.export_format)}?api_key={quote(api_key)}")
    meta = json.loads(fetch(url))
    link = (meta.get("export") or {}).get("link")
    if not link:
        raise RuntimeError(
            f"No export link for {coords.workspace}/{coords.project} v{coords.version} "
            f"({coords.export_format}); generate that export in Roboflow first"
        )
    archive = fetch(link)
    Path(root).mkdir(parents=True, exist_ok=True)
    partial = Path(root) / f".{target.name}.partial-{uuid4().hex}"
    try:
        files = _safe_extract(archive, partial)
        (partial / DOWNLOAD_MARKER).write_text(json.dumps({
            "coordinates": coords.model_dump(),
            "coordinates_sha256": coordinates_key(coords),
            "archive_sha256": hashlib.sha256(archive).hexdigest(),
            "files": files,
        }, indent=2) + "\n", encoding="utf-8")
        partial.rename(target)
    finally:
        if partial.exists():
            shutil.rmtree(partial)
    return target


def _export_root(raw: Path) -> Path:
    if (raw / "data.yaml").is_file():
        return raw
    candidates = [p for p in raw.iterdir() if p.is_dir() and (p / "data.yaml").is_file()]
    if len(candidates) != 1:
        raise FileNotFoundError(f"No unique data.yaml in {raw}")
    return candidates[0]


def import_export(
    raw: Path,
    entry: SourceEntry,
    destination: Path,
    *,
    classes: tuple[str, ...] = taxonomy.DEFAULT_CLASSES,
    seed: int = 42,
    quarantine_path: Path | None = None,
) -> dict:
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    export = _export_root(Path(raw))
    export_names = dataset_names(load_data_yaml(export))
    if tuple(export_names) != entry.classes:
        raise ValueError(
            f"{entry.name}: export classes {export_names} differ from sources.yaml "
            f"{list(entry.classes)}; review the remap table before importing"
        )
    names = list(taxonomy.select_classes(classes))
    class_map = taxonomy.class_map(entry.remap, export_names, names)
    pattern = re.compile(entry.roboflow.group_pattern) if (
        entry.roboflow and entry.roboflow.group_pattern) else None

    records = {}
    excluded = []
    for export_split in EXPORT_SPLITS:
        image_dir = export / export_split / "images"
        if not image_dir.is_dir():
            continue
        for image in sorted(p for p in image_dir.iterdir()
                            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES):
            if image.stem in records:
                raise ValueError(f"Duplicate stem in export: {image.stem}")
            label = export / export_split / "labels" / f"{image.stem}.txt"
            if not label.is_file():
                raise FileNotFoundError(f"Export frame without a label file: {image}")
            original = image.stem.split(".rf.")[0]
            if pattern is None:
                group = entry.name
            else:
                match = pattern.search(original)
                if match is None:
                    raise ValueError(f"group_pattern does not match {original!r}")
                group = validate_group_id(match["group"].lower())
            remapped = remap_label_text(
                label.read_text(encoding="utf-8"), class_map,
                label_ref=label.relative_to(export).as_posix(),
            )
            excluded.extend(remapped.excluded)
            records[image.stem] = (image, group, remapped.text, remapped.counts)
    if not records:
        raise ValueError(f"No frames found in {export}")

    quarantine = load_quarantine(quarantine_path)
    groups = sorted({group for _, group, _, _ in records.values()})
    if pattern is None:
        splits = {"train": groups, "val": [], "test": []}
        check_splits(splits, quarantine)
    else:
        splits = split_groups(groups, seed=seed, quarantine=quarantine)
    group_to_split = {group: split for split, members in splits.items() for group in members}

    destination.mkdir(parents=True)
    for split in splits:
        (destination / split / "images").mkdir(parents=True)
        (destination / split / "labels").mkdir()
    images = Counter()
    boxes = {split: Counter() for split in splits}
    for stem, (image, group, text, counts) in sorted(records.items()):
        split = group_to_split[group]
        shutil.copy2(image, destination / split / "images" / image.name)
        (destination / split / "labels" / f"{stem}.txt").write_text(
            text, encoding="utf-8", newline="\n"
        )
        images[split] += 1
        boxes[split].update({names[c]: n for c, n in counts.items()})
    (destination / "data.yaml").write_text(
        data_yaml_text(destination.name, names, test="test/images" if splits["test"] else None),
        encoding="utf-8", newline="\n",
    )
    marker_path = Path(raw) / DOWNLOAD_MARKER
    manifest = {
        "kind": "roboflow",
        "builder": "tracking.datasets.roboflow",
        "source": entry.model_dump(mode="json"),
        "raw_directory": str(Path(raw).resolve()),
        "download": json.loads(marker_path.read_text(encoding="utf-8"))
        if marker_path.is_file() else None,
        "source_version": str(entry.version),
        "source_license": entry.license,
        "source_license_status": entry.license_status,
        "seed": seed,
        "split_unit": "group",
        "split_policy": "whole groups 80/10/10" if pattern else "no game ids: train only",
        "evaluation_domain": f"public {entry.domain} footage; not the product evaluation set",
        "class_map": class_map,
        "names": names,
        "quarantine": quarantine_record(quarantine_path),
        "groups": {split: sorted(members) for split, members in splits.items()},
        "images": dict(images),
        "boxes": {split: dict(sorted(c.items())) for split, c in boxes.items()},
        "excluded_annotations": excluded,
        "oversampling": None,
        "dataset_sha256": dataset_sha256(destination),
    }
    write_manifest(destination, manifest)
    return manifest


def pull(
    name: str,
    destination: Path,
    *,
    raw_root: Path = DEFAULT_RAW_ROOT,
    classes: tuple[str, ...] = taxonomy.DEFAULT_CLASSES,
    sources_path: Path = SOURCES_PATH,
    api_key: str | None = None,
    fetch: Fetch = _http_fetch,
    allow_unresolved_license: bool = False,
    seed: int = 42,
    quarantine_path: Path | None = None,
) -> dict:
    entry = load_sources(sources_path)[name]
    if entry.source != "roboflow" or entry.roboflow is None:
        raise ValueError(f"{name} is not a Roboflow source")
    if entry.license_status == "unknown" and not allow_unresolved_license:
        raise LicenseError(f"{name}: license {entry.license!r} is unresolved; not downloading")
    if not entry.shippable:
        print(f"WARNING: {name} is {entry.license_status} ({entry.license}); weights trained "
              f"on it must not ship.", flush=True)
    raw = download(entry.roboflow, raw_root, api_key=api_key, fetch=fetch)
    return import_export(raw, entry, destination, classes=classes, seed=seed,
                         quarantine_path=quarantine_path)


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required=True, help="entry name in sources.yaml")
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--classes", type=taxonomy.parse_class_list,
                        default=taxonomy.DEFAULT_CLASSES)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quarantine", type=Path, default=None)
    parser.add_argument("--allow-unresolved-license", action="store_true",
                        help="research only; the result must not train a shipped model")
    args = parser.parse_args(argv)
    manifest = pull(
        args.source, args.destination, raw_root=args.raw_root, classes=args.classes,
        allow_unresolved_license=args.allow_unresolved_license, seed=args.seed,
        quarantine_path=args.quarantine,
    )
    print(f"Imported {args.destination.resolve()}: {manifest['images']}")
    print(f"Dataset SHA256: {manifest['dataset_sha256']}")
    return manifest


if __name__ == "__main__":
    main()
