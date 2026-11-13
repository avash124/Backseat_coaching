"""Weights registry: which detector checkpoints exist, what produced them, which one ships.

`weights_registry.json` (tracked in git) holds one record per model: where the file is
(local path relative to backend/, volume key, or S3 key), its sha256, the imgsz it was
trained at, its classes, the dataset hash behind it, the training run id, the eval
report that justified it, and a status:

    candidate  produced by a training run; never used by the pipeline
    promoted   the one model per role the pipeline should use
    retired    previously promoted

Training jobs only ever add candidates. `promote` is a deliberate human CLI step (PLAN.md
M4) that requires an eval report and typing the model name back:

    python -m tracking.registry list
    python -m tracking.registry verify
    python -m tracking.registry add-candidate --manifest tracking/weights/<run>/weights_manifest.json \
        --path tracking/weights/<run>/best.pt
    python -m tracking.registry promote <name> --eval-report <report.json> --confirm <name> --by "<you>"

After promoting, point `tracking/config.py` at the new weights and bump DETECT_VERSION
(the pipeline's cache keys depend on it).
"""

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tracking.datasets.common import file_sha256

BACKEND_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = Path(__file__).resolve().parent / "weights_registry.json"
SHA256 = r"^[0-9a-f]{64}$"
Status = Literal["candidate", "promoted", "retired"]


class WeightsIntegrityError(ValueError):
    """A weights file is missing or does not match its recorded sha256."""


class WeightsRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    name: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$")
    role: str = Field(default="player-detector", pattern=r"^[a-z0-9-]+$")
    status: Status = "candidate"
    sha256: str = Field(pattern=SHA256)
    path: str | None = None  # POSIX, relative to backend/
    volume_key: str | None = None  # "<volume>:/<path>"
    s3_key: str | None = None
    imgsz: int = Field(ge=32)
    classes: tuple[str, ...]
    base_model: str | None = None
    dataset_sha256: str | None = Field(default=None, pattern=SHA256)
    dataset_manifest: str | None = None
    training_run_id: str | None = None
    weights_manifest: str | None = None
    eval_report: str | None = None
    ultralytics_version: str | None = None
    created_at: str | None = None
    promoted_at: str | None = None
    promoted_by: str | None = None
    notes: str | None = None

    @model_validator(mode="after")
    def _has_location(self):
        if not (self.path or self.volume_key or self.s3_key):
            raise ValueError(f"{self.name}: record needs a path, volume_key or s3_key")
        return self


class Registry(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    schema_version: Literal[1] = 1
    models: tuple[WeightsRecord, ...] = ()

    @model_validator(mode="after")
    def _invariants(self):
        names = [record.name for record in self.models]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"Duplicate model names: {duplicates}")
        roles = [record.role for record in self.models if record.status == "promoted"]
        doubled = sorted({role for role in roles if roles.count(role) > 1})
        if doubled:
            raise ValueError(f"More than one promoted model for roles: {doubled}")
        return self

    def get(self, name: str) -> WeightsRecord:
        for record in self.models:
            if record.name == name:
                return record
        raise KeyError(f"No model named {name!r} in the registry")

    def promoted(self, role: str = "player-detector") -> WeightsRecord | None:
        return next((r for r in self.models if r.role == role and r.status == "promoted"), None)

    def replace(self, *records: WeightsRecord) -> "Registry":
        updated = {record.name: record for record in records}
        models = [updated.pop(record.name, record) for record in self.models]
        return Registry(models=(*models, *updated.values()))


def load_registry(path: Path = REGISTRY_PATH) -> Registry:
    return Registry.model_validate_json(Path(path).read_text(encoding="utf-8"))


def save_registry(registry: Registry, path: Path = REGISTRY_PATH) -> None:
    path = Path(path)
    temporary = path.with_name(f".{path.name}.partial")
    temporary.write_text(registry.model_dump_json(indent=2) + "\n", encoding="utf-8",
                         newline="\n")
    temporary.replace(path)


def local_path(record: WeightsRecord, base: Path = BACKEND_ROOT) -> Path | None:
    return (Path(base) / record.path).resolve() if record.path else None


def verify_record(record: WeightsRecord, base: Path = BACKEND_ROOT) -> Path:
    """Return the local weights path after checking its sha256."""
    path = local_path(record, base)
    if path is None:
        raise WeightsIntegrityError(f"{record.name} has no local path (download it first)")
    if not path.is_file():
        raise WeightsIntegrityError(f"{record.name}: {path} does not exist")
    actual = file_sha256(path)
    if actual != record.sha256:
        raise WeightsIntegrityError(f"{record.name}: {path} is {actual}, expected {record.sha256}")
    return path


def promoted_weights(role: str = "player-detector", *, path: Path = REGISTRY_PATH,
                     base: Path = BACKEND_ROOT, verify: bool = True) -> Path:
    """Local path of the promoted model for `role` (sha256-verified by default)."""
    record = load_registry(path).promoted(role)
    if record is None:
        raise KeyError(f"No promoted model for role {role!r}")
    return verify_record(record, base) if verify else local_path(record, base)


def _relative(path: Path, base: Path) -> str:
    path = Path(path).resolve()
    try:
        return path.relative_to(Path(base).resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def record_from_manifest(manifest: dict, *, weights_path: Path | None = None,
                         manifest_path: Path | None = None, name: str | None = None,
                         role: str = "player-detector",
                         base: Path = BACKEND_ROOT) -> WeightsRecord:
    """A candidate record from a remote/train.py weights_manifest.json."""
    best = manifest["weights"]["best"]
    if weights_path is not None and file_sha256(weights_path) != best["sha256"]:
        raise WeightsIntegrityError(f"{weights_path} is not the best.pt in the manifest")
    start = manifest.get("start_weights") or {}
    return WeightsRecord(
        name=name or manifest["name"],
        role=role,
        status="candidate",
        sha256=best["sha256"],
        path=_relative(weights_path, base) if weights_path else None,
        volume_key=best.get("volume_key"),
        imgsz=manifest["imgsz"],
        classes=tuple(manifest["classes"]),
        base_model=start.get("model"),
        dataset_sha256=manifest["dataset"]["sha256"],
        training_run_id=manifest.get("run_id"),
        weights_manifest=_relative(manifest_path, base) if manifest_path else None,
        ultralytics_version=(manifest.get("versions") or {}).get("ultralytics"),
        created_at=manifest.get("created_at"),
    )


def add_candidate(record: WeightsRecord, *, path: Path = REGISTRY_PATH) -> Registry:
    registry = load_registry(path)
    if any(existing.name == record.name for existing in registry.models):
        raise ValueError(f"{record.name} is already registered")
    registry = registry.replace(record.model_copy(update={"status": "candidate",
                                                          "promoted_at": None,
                                                          "promoted_by": None}))
    save_registry(registry, path)
    return registry


def promote(name: str, *, eval_report: Path, promoted_by: str, path: Path = REGISTRY_PATH,
            base: Path = BACKEND_ROOT, now: str | None = None) -> Registry:
    """Make `name` the promoted model of its role; the previous one becomes retired."""
    if not promoted_by.strip():
        raise ValueError("promoted_by is required: promotion is a human decision")
    registry = load_registry(path)
    record = registry.get(name)
    if record.status != "candidate":
        raise ValueError(f"{name} is {record.status}; only candidates can be promoted")
    report = Path(eval_report)
    report = report if report.is_absolute() else Path(base) / report
    if not report.is_file():
        raise FileNotFoundError(f"Eval report {report} not found; promotion needs one (M4)")
    verify_record(record, base)
    now = now or datetime.now(UTC).isoformat(timespec="seconds")
    updates = [record.model_copy(update={
        "status": "promoted", "eval_report": _relative(report, base),
        "promoted_at": now, "promoted_by": promoted_by,
    })]
    previous = registry.promoted(record.role)
    if previous is not None:
        updates.append(previous.model_copy(update={"status": "retired"}))
    registry = registry.replace(*updates)  # re-validates: one promoted model per role
    save_registry(registry, path)
    return registry


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--registry", type=Path, default=REGISTRY_PATH)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list")
    verify = commands.add_parser("verify", help="check local files against recorded sha256")
    verify.add_argument("--name", default=None)
    add = commands.add_parser("add-candidate")
    add.add_argument("--manifest", type=Path, required=True)
    add.add_argument("--path", type=Path, required=True, help="downloaded best.pt")
    add.add_argument("--name", default=None)
    add.add_argument("--role", default="player-detector")
    promote_cmd = commands.add_parser("promote")
    promote_cmd.add_argument("name")
    promote_cmd.add_argument("--eval-report", type=Path, required=True)
    promote_cmd.add_argument("--confirm", required=True, help="type the model name again")
    promote_cmd.add_argument("--by", required=True, help="who is promoting")
    args = parser.parse_args(argv)

    if args.command == "list":
        for r in load_registry(args.registry).models:
            print(f"{r.status:9} {r.role:16} {r.name:24} imgsz={r.imgsz} {r.sha256[:12]} "
                  f"{r.path or r.volume_key or r.s3_key}")
    elif args.command == "verify":
        registry = load_registry(args.registry)
        records = [registry.get(args.name)] if args.name else list(registry.models)
        failed = 0
        for record in records:
            try:
                print(f"ok      {record.name}: {verify_record(record)}")
            except WeightsIntegrityError as error:
                failed += 1
                print(f"FAILED  {error}")
        return 1 if failed else 0
    elif args.command == "add-candidate":
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        record = record_from_manifest(manifest, weights_path=args.path,
                                      manifest_path=args.manifest, name=args.name,
                                      role=args.role)
        add_candidate(record, path=args.registry)
        print(f"Registered candidate {record.name} ({record.sha256[:12]})")
    elif args.command == "promote":
        if args.confirm != args.name:
            print("--confirm must repeat the model name exactly", file=sys.stderr)
            return 2
        registry = promote(args.name, eval_report=args.eval_report, promoted_by=args.by,
                           path=args.registry)
        record = registry.get(args.name)
        print(f"Promoted {record.name} for {record.role}.")
        print(f"Next (manual): point tracking/config.py at {record.path} and bump DETECT_VERSION.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
