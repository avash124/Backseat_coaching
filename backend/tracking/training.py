"""Detector training (roadmap 2.8 players, 2.3b ball) — the framework layer.

No Modal, no S3, no distributed setup: paths in, paths out. `remote/train.py` provisions
the GPU and calls in here; locally the same code trains a few synthetic frames on CPU
(`python -m tracking.training smoke`), which catches config errors before paying for a GPU.

Phases (PLAN.md §2-3):
- `train_detector(cfg)`: one run starting from `cfg.model` (stock weights or a model yaml).
- `pretrain(cfg)`: the same, tagged "pretrain" (merged public data; first half of run C).
- `finetune(cfg, weights)`: our frames from a starting checkpoint, with 1/10 the LR, milder
  augmentation and an optional backbone freeze unless `cfg` sets those explicitly.
  Run C is literally `finetune(cfg, weights=pretrain(...).best_weights)`.
- Resume: `train_detector(..., resume=True)` (or `resume_run(run_dir)`) continues
  `<run>/weights/last.pt` after a preemption, and returns the recorded result if the run
  already finished, so a retried GPU job is idempotent.

Each run directory gets, next to Ultralytics' own args.yaml / results.csv / weights/:
    train_config.json      the TrainConfig + phase as started (resume must match it)
    resolved_config.json   that, plus the exact Ultralytics arguments, dataset hash, start
                           weights sha256 and library versions
    data.yaml              the dataset's data.yaml with `path` set to the directory used
    dataset_manifest.json  copy of the dataset manifest (its hash is re-verified first)
    run_result.json        RunResult: weights paths + sha256, metrics

Ultralytics' `optimizer="auto"` silently ignores `lr0`, so it is not offered: the recorded
LR is the LR that ran. Per 2.8, do not fine-tune at all if the stock model already clears
the 0.2 bar — evaluate.py decides that on our footage, never on a public val split.
"""

import argparse
import json
import math
import platform
import shutil
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from tracking.datasets.common import (
    MANIFEST_NAME,
    dataset_names,
    file_sha256,
    load_data_yaml,
    verify_dataset,
)

Phase = Literal["train", "pretrain", "finetune"]
RunState = Literal["absent", "incomplete", "resumable", "finished", "unrecorded"]
RUN_NAME_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_-]{0,79}$"
SHA256_PATTERN = r"^[0-9a-f]{64}$"

DEFAULT_MODEL = "yolo26s.pt"
# Default batch for imgsz 1280 on a 24 GB L4/A10G (remote/train.py's GPU). Activation memory
# scales with pixels, so 8 images at 1280 ~ 32 at 640, where yolo26s trains in roughly
# 10-12 GB with AMP: about half the card, leaving headroom for mosaic outliers and the
# one-to-one head. A fixed batch (not -1 autobatch) keeps runs reproducible across GPU
# types; Ultralytics accumulates gradients to a nominal batch of 64 either way. Drop to 4
# on OOM (what the H100 players-v1 run used).
L4_BATCH_1280 = 8
FINETUNE_LR0 = 0.001  # 1/10 of the default lr0 (PLAN.md §3)

TRAIN_CONFIG_NAME = "train_config.json"
RESOLVED_CONFIG_NAME = "resolved_config.json"
RESULT_NAME = "run_result.json"
# Settings that may change on resume without changing what is being trained.
RESUMABLE_OVERRIDES = frozenset({"device", "workers"})


class Augmentation(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
    flipud: float = 0.0
    fliplr: float = Field(default=0.5, ge=0, le=1)
    hsv_h: float = Field(default=0.015, ge=0, le=1)
    hsv_s: float = Field(default=0.7, ge=0, le=1)
    hsv_v: float = Field(default=0.4, ge=0, le=1)
    degrees: float = Field(default=0.0, ge=0, le=180)
    translate: float = Field(default=0.1, ge=0, le=1)
    scale: float = Field(default=0.5, ge=0, lt=1)
    mosaic: float = Field(default=1.0, ge=0, le=1)
    mixup: float = Field(default=0.0, ge=0, le=1)

    @field_validator("flipud")
    @classmethod
    def _no_vertical_flip(cls, value: float) -> float:
        if value != 0:
            raise ValueError("flipud must be 0: basketball film has a fixed up direction")
        return value


# Milder than the pretraining defaults: ~1k own frames, starting from a good checkpoint.
FINETUNE_AUGMENTATION = Augmentation(mosaic=0.5, scale=0.3, translate=0.05)


class TrainConfig(BaseModel):
    """Everything that determines a training run. Unknown keys are rejected."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
    data: Path  # data.yaml of a dataset built by tracking.datasets (manifest beside it)
    model: str = DEFAULT_MODEL  # stock weights name, local .pt, or architecture .yaml
    run_name: str = Field(default="detector", pattern=RUN_NAME_PATTERN)
    output_dir: Path = Path("tracking/runs/train")
    imgsz: int = Field(default=1280, ge=32, multiple_of=32)
    epochs: int = Field(default=60, ge=1)
    batch: int = Field(default=L4_BATCH_1280, ge=-1)  # -1 = Ultralytics autobatch
    optimizer: Literal["MuSGD", "SGD", "AdamW"] = "MuSGD"  # what "auto" chose for players-v1
    lr0: float = Field(default=0.01, gt=0, lt=1)
    lrf: float = Field(default=0.01, gt=0, le=1)
    momentum: float = Field(default=0.9, ge=0, lt=1)
    weight_decay: float = Field(default=0.0005, ge=0, lt=1)
    warmup_epochs: float = Field(default=3.0, ge=0)
    cos_lr: bool = False
    freeze: int | None = Field(default=None, ge=1)  # first N layers; backbone = 11 in yolo26
    patience: int = Field(default=15, ge=0)
    close_mosaic: int = Field(default=10, ge=0)
    augmentation: Augmentation = Augmentation()
    device: str | None = None  # None = Ultralytics auto; Modal passes "0", smoke runs "cpu"
    workers: int = Field(default=8, ge=0)
    seed: int = 42
    deterministic: bool = True
    amp: bool = True
    cache: Literal[False, "ram", "disk"] = False
    fraction: float = Field(default=1.0, gt=0, le=1)
    plots: bool = True
    expected_dataset_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    start_weights_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)

    @field_validator("batch")
    @classmethod
    def _batch(cls, value: int) -> int:
        if value == 0:
            raise ValueError("batch must be >= 1, or -1 for autobatch")
        return value

    @property
    def run_dir(self) -> Path:
        return Path(self.output_dir) / self.run_name


FINETUNE_DEFAULTS = {
    "lr0": FINETUNE_LR0,
    "epochs": 30,
    "patience": 10,
    "augmentation": FINETUNE_AUGMENTATION,
}


class RunResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    run_id: str
    phase: Phase
    run_dir: Path
    best_weights: Path
    last_weights: Path
    best_sha256: str
    last_sha256: str
    classes: list[str]
    imgsz: int
    dataset_sha256: str
    epochs_completed: int
    metrics: dict[str, float | None]
    per_class_map50_95: dict[str, float | None]
    metrics_source: str
    resolved_config_path: Path


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _write_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.partial")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8", newline="\n")
    temporary.replace(path)


def _finite(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def library_versions() -> dict:
    import torch
    import ultralytics

    cuda = torch.cuda.is_available()
    return {
        "python": platform.python_version(),
        "ultralytics": ultralytics.__version__,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if cuda else None,
        "gpu": torch.cuda.get_device_name(0) if cuda else None,
        "platform": platform.platform(),
    }


def checkpoint_is_resumable(path: Path) -> bool:
    """Ultralytics strips the optimizer (and sets epoch -1) once a run completes."""
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return checkpoint.get("epoch", -1) >= 0 and checkpoint.get("optimizer") is not None


def run_state(run_dir: Path) -> RunState:
    run_dir = Path(run_dir)
    if not run_dir.exists():
        return "absent"
    if (run_dir / RESULT_NAME).is_file():
        return "finished"
    last = run_dir / "weights" / "last.pt"
    if not last.is_file():
        return "incomplete"
    # "unrecorded": training ended but final validation / run_result.json did not.
    return "resumable" if checkpoint_is_resumable(last) else "unrecorded"


def load_run_result(run_dir: Path) -> RunResult:
    return RunResult.model_validate_json((Path(run_dir) / RESULT_NAME).read_text("utf-8"))


def ultralytics_args(cfg: TrainConfig, data_yaml: Path, run_dir: Path) -> dict:
    """The exact keyword arguments passed to `YOLO.train` for a fresh run."""
    args = {
        "data": str(data_yaml),
        "epochs": cfg.epochs,
        "imgsz": cfg.imgsz,
        "batch": cfg.batch,
        "optimizer": cfg.optimizer,
        "lr0": cfg.lr0,
        "lrf": cfg.lrf,
        "momentum": cfg.momentum,
        "weight_decay": cfg.weight_decay,
        "warmup_epochs": cfg.warmup_epochs,
        "cos_lr": cfg.cos_lr,
        "patience": cfg.patience,
        "close_mosaic": cfg.close_mosaic,
        "workers": cfg.workers,
        "seed": cfg.seed,
        "deterministic": cfg.deterministic,
        "amp": cfg.amp,
        "cache": cfg.cache,
        "fraction": cfg.fraction,
        "plots": cfg.plots,
        "val": True,
        "project": str(run_dir.parent),
        "name": run_dir.name,
        "exist_ok": True,
        **cfg.augmentation.model_dump(),
    }
    if cfg.freeze is not None:
        args["freeze"] = cfg.freeze
    if cfg.device is not None:
        args["device"] = cfg.device
    return args


def _resolved_data_yaml(dataset_dir: Path, run_dir: Path) -> Path:
    config = load_data_yaml(dataset_dir)
    config["path"] = dataset_dir.as_posix()  # data.yaml says /datasets/<name> (Modal mount)
    path = run_dir / "data.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8", newline="\n")
    return path


def _check_same_config(run_dir: Path, cfg: TrainConfig, phase: Phase) -> None:
    path = run_dir / TRAIN_CONFIG_NAME
    if not path.is_file():
        raise FileNotFoundError(f"{path} missing: {run_dir} was not started by tracking.training")
    saved = json.loads(path.read_text(encoding="utf-8"))
    before, now = saved["config"], cfg.model_dump(mode="json")
    differs = sorted(key for key in set(before) | set(now)
                     if key not in RESUMABLE_OVERRIDES and before.get(key) != now.get(key))
    if saved["phase"] != phase:
        differs.append("phase")
    if differs:
        raise ValueError(f"Cannot resume {run_dir}: config differs from {path} in {differs}")


def _epochs_completed(run_dir: Path) -> int:
    results = run_dir / "results.csv"
    if not results.is_file():
        return 0
    return max(0, len(results.read_text(encoding="utf-8").strip().splitlines()) - 1)


def _metrics_from_results(results, names: list[str]) -> tuple[dict, dict]:
    metrics = {str(k): _finite(v) for k, v in (getattr(results, "results_dict", None) or {}).items()}
    maps = getattr(getattr(results, "box", None), "maps", None)
    per_class = {}
    if maps is not None:
        per_class = {name: _finite(maps[i]) if i < len(maps) else None
                     for i, name in enumerate(names)}
    return metrics, per_class


def _finish(run_dir: Path, cfg: TrainConfig, phase: Phase, dataset_sha256: str,
            names: list[str], results) -> RunResult:
    weights = run_dir / "weights"
    last = weights / "last.pt"
    best = weights / "best.pt" if (weights / "best.pt").is_file() else last
    if results is not None:
        metrics, per_class = _metrics_from_results(results, names)
        source = "final validation of best.pt"
    else:
        import torch

        checkpoint = torch.load(best, map_location="cpu", weights_only=False)
        metrics = {str(k): _finite(v) for k, v in (checkpoint.get("train_metrics") or {}).items()}
        per_class, source = {}, "best.pt train_metrics (final validation did not complete)"
    result = RunResult(
        run_id=cfg.run_name,
        phase=phase,
        run_dir=run_dir,
        best_weights=best,
        last_weights=last,
        best_sha256=file_sha256(best),
        last_sha256=file_sha256(last),
        classes=names,
        imgsz=cfg.imgsz,
        dataset_sha256=dataset_sha256,
        epochs_completed=_epochs_completed(run_dir),
        metrics=metrics,
        per_class_map50_95=per_class,
        metrics_source=source,
        resolved_config_path=run_dir / RESOLVED_CONFIG_NAME,
    )
    _write_json(run_dir / RESULT_NAME, result.model_dump(mode="json"))
    return result


def train_detector(
    cfg: TrainConfig,
    *,
    phase: Phase = "train",
    resume: bool = False,
    on_checkpoint: Callable[[Path], None] | None = None,
) -> RunResult:
    """Train (or resume) one run. `on_checkpoint(last_pt)` fires after every checkpoint save."""
    cfg = cfg.model_copy(update={
        "data": Path(cfg.data).resolve(),
        "output_dir": Path(cfg.output_dir).resolve(),
    })
    run_dir = cfg.run_dir
    state = run_state(run_dir)
    if state != "absent" and not resume:
        raise FileExistsError(f"{run_dir} exists ({state}); pass resume=True or a new run_name")
    if state in ("resumable", "finished", "unrecorded"):
        _check_same_config(run_dir, cfg, phase)
    dataset_dir = cfg.data.parent
    names = dataset_names(load_data_yaml(dataset_dir))
    if state == "finished":
        return load_run_result(run_dir)
    if state == "unrecorded":
        manifest = verify_dataset(dataset_dir, cfg.expected_dataset_sha256)
        return _finish(run_dir, cfg, phase, manifest["dataset_sha256"], names, None)

    # Re-hash the dataset so the recorded hash is what is actually trained on.
    manifest = verify_dataset(dataset_dir, cfg.expected_dataset_sha256)
    run_dir.mkdir(parents=True, exist_ok=True)
    data_yaml = _resolved_data_yaml(dataset_dir, run_dir)
    shutil.copy2(dataset_dir / MANIFEST_NAME, run_dir / MANIFEST_NAME)

    from ultralytics import YOLO

    if state == "resumable":
        resolved = json.loads((run_dir / RESOLVED_CONFIG_NAME).read_text(encoding="utf-8"))
        resolved.setdefault("resumes", []).append({"at": _now(), "device": cfg.device,
                                                   "versions": library_versions()})
        model = YOLO(str(run_dir / "weights" / "last.pt"))
        train_kwargs = {"resume": True, "data": str(data_yaml), "workers": cfg.workers}
        if cfg.device is not None:
            train_kwargs["device"] = cfg.device
    else:
        model = YOLO(cfg.model)
        start_path = Path(getattr(model, "ckpt_path", None) or cfg.model)
        start_sha256 = file_sha256(start_path) if start_path.is_file() else None
        if cfg.start_weights_sha256 and cfg.start_weights_sha256 != start_sha256:
            raise ValueError(f"Start weights {start_path} are {start_sha256}, "
                             f"expected {cfg.start_weights_sha256}")
        train_kwargs = ultralytics_args(cfg, data_yaml, run_dir)
        _write_json(run_dir / TRAIN_CONFIG_NAME,
                    {"phase": phase, "config": cfg.model_dump(mode="json")})
        resolved = {
            "run_id": cfg.run_name,
            "phase": phase,
            "started_at": _now(),
            "train_config": cfg.model_dump(mode="json"),
            "ultralytics_args": train_kwargs,
            "dataset": {
                "directory": dataset_dir.as_posix(),
                "data_yaml": cfg.data.as_posix(),
                "dataset_sha256": manifest["dataset_sha256"],
                "manifest_sha256": file_sha256(dataset_dir / MANIFEST_NAME),
                "names": names,
                "source_license": manifest.get("source_license"),
            },
            "start_weights": {"model": cfg.model, "path": start_path.as_posix(),
                              "sha256": start_sha256},
            "versions": library_versions(),
        }
    _write_json(run_dir / RESOLVED_CONFIG_NAME, resolved)

    if on_checkpoint is not None:
        model.add_callback("on_model_save", lambda trainer: on_checkpoint(Path(trainer.last)))
    results = model.train(**train_kwargs)
    return _finish(run_dir, cfg, phase, manifest["dataset_sha256"], names, results)


def pretrain(cfg: TrainConfig, **kwargs) -> RunResult:
    """Merged public data (PLAN.md M3, run C first half): the default schedule applies."""
    return train_detector(cfg, phase="pretrain", **kwargs)


def backbone_layers(weights: str | Path) -> int:
    from ultralytics import YOLO

    return len(YOLO(str(weights)).model.yaml["backbone"])


def finetune_config(cfg: TrainConfig, weights: str | Path, *,
                    freeze_backbone: bool = False) -> TrainConfig:
    """`cfg` with fine-tune defaults for every field it did not set explicitly."""
    if freeze_backbone and "freeze" in cfg.model_fields_set:
        raise ValueError("Set either cfg.freeze or freeze_backbone, not both")
    # Paths become absolute; a bare stock name ("yolo26s.pt") stays a name even after
    # Ultralytics downloads it into the cwd, so a resumed run sees the same config.
    weights = str(Path(weights).resolve()) if Path(weights).parent != Path(".") else str(weights)
    updates = {name: value for name, value in FINETUNE_DEFAULTS.items()
               if name not in cfg.model_fields_set}
    updates["model"] = weights
    if freeze_backbone:
        updates["freeze"] = backbone_layers(weights)
    payload = cfg.model_dump()
    payload.update({k: v.model_dump() if isinstance(v, BaseModel) else v
                    for k, v in updates.items()})
    return TrainConfig.model_validate(payload)


def finetune(cfg: TrainConfig, weights: str | Path, *, freeze_backbone: bool = False,
             **kwargs) -> RunResult:
    """Our own frames from `weights` (lower LR, milder augmentation, optional freeze)."""
    return train_detector(finetune_config(cfg, weights, freeze_backbone=freeze_backbone),
                          phase="finetune", **kwargs)


def resume_run(run_dir: Path, *, device: str | None = None, workers: int | None = None,
               on_checkpoint: Callable[[Path], None] | None = None) -> RunResult:
    """Continue a run from its own train_config.json (same config, maybe a new device)."""
    saved = json.loads((Path(run_dir) / TRAIN_CONFIG_NAME).read_text(encoding="utf-8"))
    overrides = {k: v for k, v in (("device", device), ("workers", workers)) if v is not None}
    cfg = TrainConfig.model_validate({**saved["config"], **overrides})
    return train_detector(cfg, phase=saved["phase"], resume=True, on_checkpoint=on_checkpoint)


def load_config(path: Path, **overrides) -> TrainConfig:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return TrainConfig.model_validate({**data, **overrides})


def smoke_config(data_yaml: Path, output_dir: Path, run_name: str = "smoke") -> TrainConfig:
    """1 epoch of a from-scratch yolo26n at 64 px on CPU: seconds, no downloads."""
    return TrainConfig(
        data=data_yaml, model="yolo26n.yaml", run_name=run_name, output_dir=output_dir,
        imgsz=64, epochs=1, batch=4, device="cpu", workers=0, amp=False, plots=False,
        close_mosaic=0, warmup_epochs=0,
    )


def build_smoke_dataset(directory: Path) -> Path:
    """Synthetic frames -> own_footage builder -> data.yaml (the real dataset path)."""
    from tracking.datasets import own_footage, synthetic

    directory = Path(directory)
    synthetic.write_frames(directory / "frames")
    quarantine = directory / "quarantine.yaml"
    quarantine.write_text("groups: []\n", encoding="utf-8")
    own_footage.build([directory / "frames"], directory / "smoke_v1", val_fraction=0.25,
                      test_fraction=0.25, quarantine_path=quarantine,
                      source_version="synthetic", license="synthetic")
    return directory / "smoke_v1" / "data.yaml"


def main(argv: list[str] | None = None) -> RunResult:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    smoke = commands.add_parser("smoke", help="1 CPU epoch on synthetic frames")
    smoke.add_argument("--output", type=Path, default=Path("tracking/runs/wp3-smoke"))
    train = commands.add_parser("train", help="train from a TrainConfig YAML/JSON file")
    train.add_argument("--config", type=Path, required=True)
    train.add_argument("--phase", choices=("train", "pretrain", "finetune"), default="train")
    train.add_argument("--weights", default=None, help="starting weights (finetune)")
    train.add_argument("--freeze-backbone", action="store_true")
    train.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "smoke":
        if args.output.exists():
            raise FileExistsError(f"{args.output} exists; choose a new --output")
        data_yaml = build_smoke_dataset(args.output)
        result = train_detector(smoke_config(data_yaml, args.output / "runs"))
    else:
        cfg = load_config(args.config)
        if args.phase == "finetune":
            if not args.weights:
                parser.error("--phase finetune needs --weights")
            result = finetune(cfg, args.weights, freeze_backbone=args.freeze_backbone,
                              resume=args.resume)
        else:
            result = train_detector(cfg, phase=args.phase, resume=args.resume)
    print(result.model_dump_json(indent=2))
    return result


if __name__ == "__main__":
    main()
