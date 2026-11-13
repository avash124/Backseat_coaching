"""GPU training job on Modal — the wrapper, not the training logic.

PAID GPU TIME. Run from backend/, detached so a closed laptop does not kill the run:

    .\\.venv\\Scripts\\modal.exe run --env main --detach -m tracking.remote.train \\
        --config path\\to\\request.yaml \\
        [--upload-dataset ..\\datasets\\own\\players_own_v1] \\
        [--weights tracking\\weights\\players-v1.pt] [--verify-only]

Request file (YAML or JSON; validated by `TrainRequest`):

    run_name: players-ft-own-v1        # -> nextup-checkpoints:/players-ft-own-v1/
    phase: finetune                    # train | pretrain | finetune
    dataset: players_own_v1            # directory on the nextup-datasets volume
    dataset_sha256: <dataset_manifest.json dataset_sha256>
    start_weights: yolo26s.pt          # stock name, or set by --weights (inputs/<sha>.pt)
    freeze_backbone: false
    train:                             # any TrainConfig field except paths/device/name
      epochs: 30

Flow:
1. local: optionally upload a locally built dataset (refused if its hash is not the
   requested one, or if the name already exists on the volume — omit --upload-dataset
   on re-runs) and starting weights (content-addressed: inputs/<sha256>.pt).
2. `prepare_dataset` (CPU): re-hash the dataset on the volume against the request.
   Dataset building/downloading never happens at GPU rates.
3. `train` (GPU, TRAIN_GPU): verify the dataset hash again, train via tracking.training
   with output on nextup-checkpoints, commit the volume after every checkpoint save, and
   resume from last.pt when Modal restarts a preempted/failed attempt. On success it writes
   `<run>/weights_manifest.json` (weights sha256, dataset hash, resolved config, metrics,
   library versions) and returns it.

It NEVER promotes weights. Promotion is the human step in tracking/registry.py after an
own-footage eval report (PLAN.md M4). The image is the shared inference image from
remote/app.py (same torch/ultralytics pins as inference; training needs nothing more).
"""

import json
import re
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, Literal

import modal
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from tracking.datasets.common import file_sha256
from tracking.remote.app import image
from tracking.training import RUN_NAME_PATTERN, SHA256_PATTERN, TrainConfig

# L4 (24 GB) is the right size for yolo26s at 1280 (PLAN.md §3); A100/H100 is waste here.
# training.L4_BATCH_1280 is the matching default batch.
TRAIN_GPU = "L4"
TRAIN_TIMEOUT_S = 12 * 60 * 60
DATASETS_VOLUME = "nextup-datasets"
CHECKPOINTS_VOLUME = "nextup-checkpoints"
DATASETS_ROOT = PurePosixPath("/datasets")
CHECKPOINTS_ROOT = PurePosixPath("/checkpoints")
INPUTS_DIR = "inputs"  # uploaded starting weights, named by sha256
WEIGHTS_MANIFEST = "weights_manifest.json"
STOCK_WEIGHTS = re.compile(r"yolo26[nsmlx]\.(pt|yaml)")
# Set by the job, never by the request.
RESERVED_TRAIN_KEYS = frozenset({
    "data", "model", "run_name", "output_dir", "device", "expected_dataset_sha256",
    "start_weights_sha256",
})

app = modal.App("nextup-detector-training")
datasets_volume = modal.Volume.from_name(DATASETS_VOLUME, create_if_missing=True)
checkpoints_volume = modal.Volume.from_name(CHECKPOINTS_VOLUME, create_if_missing=True)


class TrainRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    run_name: str = Field(pattern=RUN_NAME_PATTERN)
    phase: Literal["train", "pretrain", "finetune"] = "finetune"
    dataset: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,99}$")
    dataset_sha256: str = Field(pattern=SHA256_PATTERN)
    start_weights: str = "yolo26s.pt"
    start_weights_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    freeze_backbone: bool = False
    train: dict[str, Any] = Field(default_factory=dict)

    @field_validator("run_name")
    @classmethod
    def _not_reserved(cls, value: str) -> str:
        if value == INPUTS_DIR:
            raise ValueError(f"run_name {INPUTS_DIR!r} is reserved")
        return value

    @field_validator("train")
    @classmethod
    def _train_overrides(cls, value: dict) -> dict:
        reserved = sorted(RESERVED_TRAIN_KEYS & set(value))
        if reserved:
            raise ValueError(f"train: {reserved} are set by the job, not the request")
        return value

    @model_validator(mode="after")
    def _start_weights(self):
        if STOCK_WEIGHTS.fullmatch(self.start_weights):
            return self
        path = PurePosixPath(self.start_weights)
        if path.is_absolute() or ".." in path.parts or path.suffix != ".pt":
            raise ValueError("start_weights: a stock yolo26 name or a checkpoints-volume .pt path")
        if self.start_weights_sha256 is None:
            raise ValueError("start_weights_sha256 is required for checkpoint start weights")
        return self


def load_request(path: Path) -> TrainRequest:
    return TrainRequest.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


def start_weights_location(request: TrainRequest, checkpoints_root=CHECKPOINTS_ROOT) -> str:
    if STOCK_WEIGHTS.fullmatch(request.start_weights):
        return request.start_weights  # Ultralytics downloads it inside the container
    return str(PurePosixPath(checkpoints_root) / request.start_weights)


def build_train_config(
    request: TrainRequest,
    datasets_root=DATASETS_ROOT,
    checkpoints_root=CHECKPOINTS_ROOT,
) -> TrainConfig:
    """The TrainConfig the GPU job runs. Only request.train keys count as explicitly set,
    so `finetune` still applies its defaults (lower LR...) to everything else."""
    return TrainConfig.model_validate({
        **request.train,
        "data": Path(str(PurePosixPath(datasets_root) / request.dataset / "data.yaml")),
        "model": start_weights_location(request, checkpoints_root),
        "start_weights_sha256": request.start_weights_sha256,
        "run_name": request.run_name,
        "output_dir": Path(str(checkpoints_root)),
        "device": "0",
        "expected_dataset_sha256": request.dataset_sha256,
    })


def volume_key(volume: str, root: PurePosixPath, path: Path | str) -> str:
    relative = PurePosixPath(Path(path).as_posix()).relative_to(root)
    return f"{volume}:/{relative}"


def build_weights_manifest(request: TrainRequest, result, *, created_at: str | None = None,
                           checkpoints_root=CHECKPOINTS_ROOT) -> dict:
    """Pure: what the job records next to the weights. Status is always `candidate`."""
    resolved = json.loads(Path(result.resolved_config_path).read_text(encoding="utf-8"))
    return {
        "schema_version": 1,
        "status": "candidate",
        "name": request.run_name,
        "run_id": result.run_id,
        "phase": result.phase,
        "weights": {
            "best": {"volume_key": volume_key(CHECKPOINTS_VOLUME, checkpoints_root,
                                              result.best_weights),
                     "sha256": result.best_sha256},
            "last": {"volume_key": volume_key(CHECKPOINTS_VOLUME, checkpoints_root,
                                              result.last_weights),
                     "sha256": result.last_sha256},
        },
        "imgsz": result.imgsz,
        "classes": result.classes,
        "dataset": {"name": request.dataset, "sha256": result.dataset_sha256,
                    "source_license": resolved.get("dataset", {}).get("source_license")},
        "start_weights": resolved.get("start_weights"),
        "metrics": result.metrics,
        "per_class_map50_95": result.per_class_map50_95,
        "metrics_source": result.metrics_source,
        "epochs_completed": result.epochs_completed,
        "versions": resolved.get("versions"),
        "gpu": TRAIN_GPU,
        "request": request.model_dump(mode="json"),
        "resolved_config": resolved,
        "created_at": created_at or datetime.now(UTC).isoformat(timespec="seconds"),
        "promotion": "never automatic: python -m tracking.registry add-candidate / promote",
    }


@app.function(image=image, cpu=4.0, memory=8192, timeout=60 * 60,
              volumes={str(DATASETS_ROOT): datasets_volume})
def prepare_dataset(dataset: str, dataset_sha256: str) -> dict:
    """CPU: confirm the dataset on the volume is exactly the requested one."""
    from tracking.datasets.common import count_boxes, verify_dataset

    datasets_volume.reload()
    directory = Path(str(DATASETS_ROOT / dataset))
    manifest = verify_dataset(directory, dataset_sha256)
    return {
        "dataset": dataset,
        "dataset_sha256": manifest["dataset_sha256"],
        "names": manifest.get("names"),
        "images": manifest.get("images"),
        "boxes_on_disk": count_boxes(directory),
        "source_license": manifest.get("source_license"),
    }


@app.function(
    image=image,
    gpu=TRAIN_GPU,
    cpu=8.0,
    memory=32768,
    timeout=TRAIN_TIMEOUT_S,
    retries=modal.Retries(max_retries=2, initial_delay=30.0, backoff_coefficient=1.0),
    volumes={str(DATASETS_ROOT): datasets_volume, str(CHECKPOINTS_ROOT): checkpoints_volume},
)
def train(request: dict) -> dict:
    from tracking import training
    from tracking.datasets.common import verify_dataset

    req = TrainRequest.model_validate(request)
    datasets_volume.reload()
    checkpoints_volume.reload()
    cfg = build_train_config(req)
    run_dir = Path(str(CHECKPOINTS_ROOT / req.run_name))
    manifest_path = run_dir / "weights" / WEIGHTS_MANIFEST
    if manifest_path.is_file():  # a retried input whose previous attempt finished
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    verify_dataset(cfg.data.parent, req.dataset_sha256)
    start = Path(cfg.model)
    if req.start_weights_sha256 and file_sha256(start) != req.start_weights_sha256:
        raise ValueError(f"{start} does not match start_weights_sha256")

    # A preempted or failed attempt left a run dir: continue it rather than start over.
    resume = training.run_state(run_dir) != "absent"

    def commit(_last_pt: Path) -> None:  # after every last.pt/best.pt save
        checkpoints_volume.commit()

    try:
        if req.phase == "finetune":
            result = training.finetune(cfg, cfg.model, freeze_backbone=req.freeze_backbone,
                                       resume=resume, on_checkpoint=commit)
        else:
            result = training.train_detector(cfg, phase=req.phase, resume=resume,
                                             on_checkpoint=commit)
        manifest = build_weights_manifest(req, result)
        manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n",
                                 encoding="utf-8")
    finally:
        checkpoints_volume.commit()
    return manifest


@app.local_entrypoint()
def main(config: str, upload_dataset: str = "", weights: str = "", verify_only: bool = False):
    from tracking.datasets.common import dataset_sha256

    request = load_request(Path(config))
    if weights:
        local = Path(weights).resolve()
        sha = file_sha256(local)
        key = f"{INPUTS_DIR}/{sha}.pt"
        with checkpoints_volume.batch_upload(force=True) as upload:  # content-addressed
            upload.put_file(local, f"/{key}")
        request = TrainRequest.model_validate({**request.model_dump(), "start_weights": key,
                                               "start_weights_sha256": sha})
        print(f"Start weights: {CHECKPOINTS_VOLUME}:/{key}", flush=True)
    if upload_dataset:
        local = Path(upload_dataset).resolve()
        if local.name != request.dataset:
            raise ValueError(f"Local dataset dir {local.name!r} != request dataset "
                             f"{request.dataset!r} (data.yaml path is part of the hash)")
        local_sha = dataset_sha256(local)
        if local_sha != request.dataset_sha256:
            raise ValueError(f"{local} is {local_sha}, request says {request.dataset_sha256}")
        print(f"Uploading {local} -> {DATASETS_VOLUME}:/{request.dataset}", flush=True)
        with datasets_volume.batch_upload(force=False) as upload:  # never overwrite a dataset
            upload.put_directory(local, f"/{request.dataset}")
    summary = prepare_dataset.remote(request.dataset, request.dataset_sha256)
    print(json.dumps(summary, indent=2), flush=True)
    if verify_only:
        return
    print(f"Training {request.run_name} on {TRAIN_GPU}...", flush=True)
    manifest = train.remote(request.model_dump(mode="json"))
    print(json.dumps({k: manifest[k] for k in ("name", "weights", "metrics", "dataset")},
                     indent=2))
    print(
        f"Candidate only. Download: modal volume get {CHECKPOINTS_VOLUME} "
        f"/{request.run_name}/weights tracking/weights/{request.run_name} --env main\n"
        f"Then: python -m tracking.registry add-candidate --manifest "
        f"tracking/weights/{request.run_name}/{WEIGHTS_MANIFEST} "
        f"--path tracking/weights/{request.run_name}/best.pt"
    )
