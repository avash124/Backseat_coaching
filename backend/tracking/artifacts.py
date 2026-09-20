"""Local artifact IO. The worker can later upload these using app.storage keys."""

import hashlib
import json
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4


def sha256_file(path: Path) -> str:
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def fingerprint(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


@contextmanager
def atomic_path(target: Path):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.stem}-{uuid4().hex}.partial{target.suffix}")
    try:
        yield temporary
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path: Path, payload: dict):
    with atomic_path(path) as temporary:
        temporary.write_text(
            json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )


def stage_is_complete(directory: Path, stage: str, key: str, filenames: tuple[str, ...]) -> bool:
    try:
        marker = json.loads((directory / f"{stage}.complete.json").read_text(encoding="utf-8"))
        return marker["key"] == key and all(
            sha256_file(directory / name) == marker["files"][name] for name in filenames
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


def complete_stage(directory: Path, stage: str, key: str, filenames: tuple[str, ...]):
    write_json(
        directory / f"{stage}.complete.json",
        {
            "key": key,
            "files": {name: sha256_file(directory / name) for name in filenames},
        },
    )


@contextmanager
def tracks_writer(path: Path, header: dict):
    """Stream the frames array; memory use does not grow with video duration."""
    with atomic_path(path) as temporary, temporary.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(header, allow_nan=False)[:-1] + ', "frames": [')
        first = True

        def append(frame):
            nonlocal first
            if not first:
                handle.write(",")
            handle.write("\n" + frame.model_dump_json())
            first = False

        yield append
        handle.write("\n]}\n")
