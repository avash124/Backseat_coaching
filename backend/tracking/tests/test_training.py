import ast
import inspect
import json
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from tracking import training
from tracking.datasets.common import read_manifest
from tracking.training import TrainConfig


@pytest.fixture(scope="module", autouse=True)
def offline_ultralytics():
    """Ultralytics fetches Arial.ttf on first dataset check; keep tests offline."""
    pytest.importorskip("ultralytics")
    import ultralytics.data.utils as data_utils

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(data_utils, "check_font", lambda *args, **kwargs: None)
        yield


@pytest.fixture(scope="module")
def smoke_data(tmp_path_factory) -> Path:
    return training.build_smoke_dataset(tmp_path_factory.mktemp("smoke-data"))


@pytest.fixture(scope="module")
def smoke_run(smoke_data, tmp_path_factory):
    output = tmp_path_factory.mktemp("smoke-runs")
    return training.train_detector(training.smoke_config(smoke_data, output)), output


# --- config -------------------------------------------------------------------------


def test_train_config_defaults():
    cfg = TrainConfig(data="x/data.yaml")
    assert cfg.augmentation.flipud == 0
    assert cfg.batch == training.L4_BATCH_1280 == 8
    assert (cfg.imgsz, cfg.model, cfg.seed, cfg.optimizer) == (1280, "yolo26s.pt", 42, "MuSGD")
    assert (cfg.epochs, cfg.patience, cfg.close_mosaic, cfg.lr0) == (60, 15, 10, 0.01)
    assert cfg.freeze is None and cfg.deterministic


@pytest.mark.parametrize("overrides", [
    {"augmentation": {"flipud": 0.5}},
    {"optimizer": "auto"},  # would silently ignore lr0
    {"batch": 0},
    {"imgsz": 1000},
    {"run_name": "../escape"},
    {"unknown_key": 1},
])
def test_train_config_rejects_unsafe_values(overrides):
    with pytest.raises(ValidationError):
        TrainConfig(data="x/data.yaml", **overrides)


def test_finetune_defaults_only_fill_unset_fields():
    derived = training.finetune_config(TrainConfig(data="x/data.yaml"), "yolo26s.pt")
    assert derived.lr0 == training.FINETUNE_LR0 == pytest.approx(0.01 / 10)
    assert derived.epochs == 30 and derived.model == "yolo26s.pt"
    assert derived.augmentation.mosaic < TrainConfig(data="x").augmentation.mosaic
    assert derived.augmentation.flipud == 0
    explicit = training.finetune_config(TrainConfig(data="x/data.yaml", lr0=0.004, epochs=5), "w.pt")
    assert (explicit.lr0, explicit.epochs) == (0.004, 5)
    with pytest.raises(ValueError, match="not both"):
        training.finetune_config(TrainConfig(data="x", freeze=3), "w.pt", freeze_backbone=True)


def test_ultralytics_args_carry_the_resolved_config(tmp_path):
    cfg = TrainConfig(data="x/data.yaml", freeze=11, device="0")
    args = training.ultralytics_args(cfg, tmp_path / "data.yaml", tmp_path / "runs" / "r1")
    assert args["flipud"] == 0.0 and args["fliplr"] == 0.5
    assert (args["optimizer"], args["lr0"], args["freeze"], args["device"]) == (
        "MuSGD", 0.01, 11, "0")
    assert (args["project"], args["name"], args["exist_ok"]) == (
        str(tmp_path / "runs"), "r1", True)
    assert "device" not in training.ultralytics_args(TrainConfig(data="x"), tmp_path, tmp_path)


# --- CPU training -------------------------------------------------------------------


def test_smoke_training_produces_weights_metrics_and_provenance(smoke_run, smoke_data):
    result, _ = smoke_run
    assert result.best_weights.is_file() and result.last_weights.is_file()
    assert result.epochs_completed == 1
    assert "metrics/mAP50-95(B)" in result.metrics
    assert set(result.per_class_map50_95) == {"player", "referee"}
    manifest = read_manifest(smoke_data.parent)
    assert result.dataset_sha256 == manifest["dataset_sha256"]
    resolved = json.loads(result.resolved_config_path.read_text())
    assert resolved["dataset"]["dataset_sha256"] == manifest["dataset_sha256"]
    assert resolved["ultralytics_args"]["flipud"] == 0.0
    assert resolved["versions"]["ultralytics"]
    assert resolved["train_config"]["model"] == "yolo26n.yaml"
    run_dir = result.run_dir
    for name in ("train_config.json", "run_result.json", "dataset_manifest.json", "args.yaml"):
        assert (run_dir / name).is_file(), name
    assert yaml.safe_load((run_dir / "data.yaml").read_text())["path"] == (
        smoke_data.parent.resolve().as_posix())
    assert training.run_state(run_dir) == "finished"


def test_finished_run_is_not_overwritten_and_resume_is_idempotent(smoke_run, smoke_data):
    result, output = smoke_run
    cfg = training.smoke_config(smoke_data, output)
    with pytest.raises(FileExistsError, match="resume=True"):
        training.train_detector(cfg)
    assert training.train_detector(cfg, resume=True) == result
    with pytest.raises(ValueError, match="differs"):
        training.train_detector(cfg.model_copy(update={"lr0": 0.02}), resume=True)


def test_interrupted_run_resumes_from_last_checkpoint(smoke_data, tmp_path):
    cfg = training.smoke_config(smoke_data, tmp_path, run_name="interrupted")
    cfg = TrainConfig.model_validate({**cfg.model_dump(), "epochs": 2})

    class Preempted(Exception):
        pass

    def preempt(_last):
        raise Preempted

    with pytest.raises(Preempted):
        training.train_detector(cfg, on_checkpoint=preempt)
    assert training.run_state(cfg.run_dir) == "resumable"
    saves = []
    result = training.resume_run(cfg.run_dir, on_checkpoint=saves.append)
    assert result.epochs_completed == 2
    assert [path.name for path in saves] == ["last.pt"]  # only epoch 2 ran: a true resume
    assert json.loads(result.resolved_config_path.read_text())["resumes"]


def test_finetune_from_checkpoint_with_frozen_backbone(smoke_run, smoke_data, tmp_path):
    base, _ = smoke_run
    cfg = training.smoke_config(smoke_data, tmp_path, run_name="ft")
    result = training.finetune(cfg, base.best_weights, freeze_backbone=True)
    assert result.phase == "finetune" and result.best_weights.is_file()
    resolved = json.loads(result.resolved_config_path.read_text())
    assert resolved["ultralytics_args"]["freeze"] == 11  # yolo26 backbone layers 0-10
    assert resolved["ultralytics_args"]["lr0"] == training.FINETUNE_LR0
    assert resolved["start_weights"]["sha256"] == base.best_sha256


def test_dataset_hash_is_verified_before_training(smoke_data, tmp_path):
    cfg = training.smoke_config(smoke_data, tmp_path, run_name="wrong-hash")
    cfg = cfg.model_copy(update={"expected_dataset_sha256": "0" * 64})
    with pytest.raises(ValueError, match="was requested"):
        training.train_detector(cfg)
    assert not (tmp_path / "wrong-hash" / "weights").exists()


# --- remote/train.py (Modal wrapper; pure helpers only, no network) -----------------


@pytest.fixture
def remote():
    pytest.importorskip("modal")
    from tracking.remote import train as remote_train

    return remote_train


def request(**overrides):
    payload = {"run_name": "players-ft-own-v1", "dataset": "players_own_v1",
               "dataset_sha256": "c" * 64}
    return {**payload, **overrides}


def test_remote_defaults_to_l4_and_never_promotes(remote):
    assert remote.TRAIN_GPU == "L4"
    assert remote.app.name == "nextup-detector-training"
    tree = ast.parse(inspect.getsource(remote))
    imported = {alias.name for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
                for alias in node.names} | {
        node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    assert "tracking.registry" not in imported and "promote" not in imported


def test_remote_request_validation(remote):
    assert remote.TrainRequest.model_validate(request()).start_weights == "yolo26s.pt"
    bad = [
        {"train": {"device": "cpu"}},
        {"train": {"data": "/elsewhere"}},
        {"run_name": "inputs"},
        {"start_weights": "../escape.pt", "start_weights_sha256": "d" * 64},
        {"start_weights": "inputs/x.pt"},  # checkpoint weights need a sha256
        {"dataset": "../players_v1"},
        {"dataset_sha256": "short"},
    ]
    for overrides in bad:
        with pytest.raises(ValidationError):
            remote.TrainRequest.model_validate(request(**overrides))


def test_remote_builds_the_train_config_and_keeps_finetune_defaults(remote, tmp_path):
    path = tmp_path / "req.yaml"
    path.write_text(yaml.safe_dump(request(
        start_weights="inputs/abc.pt", start_weights_sha256="d" * 64, train={"epochs": 12})))
    req = remote.load_request(path)
    cfg = remote.build_train_config(req)
    assert cfg.data.as_posix() == "/datasets/players_own_v1/data.yaml"
    assert Path(cfg.output_dir).as_posix() == "/checkpoints" and cfg.device == "0"
    assert cfg.model == "/checkpoints/inputs/abc.pt" and cfg.expected_dataset_sha256 == "c" * 64
    derived = training.finetune_config(cfg, "yolo26s.pt")
    assert derived.epochs == 12 and derived.lr0 == training.FINETUNE_LR0
    assert derived.batch == training.L4_BATCH_1280


def test_remote_weights_manifest_is_a_candidate_with_provenance(remote, tmp_path):
    resolved = tmp_path / "resolved_config.json"
    resolved.write_text(json.dumps({
        "versions": {"ultralytics": "8.4.155", "torch": "2.8.0"},
        "start_weights": {"model": "yolo26s.pt", "sha256": None},
        "dataset": {"source_license": "internal"},
    }))
    run = "/checkpoints/players-ft-own-v1"
    result = training.RunResult(
        run_id="players-ft-own-v1", phase="finetune", run_dir=Path(run),
        best_weights=Path(f"{run}/weights/best.pt"), last_weights=Path(f"{run}/weights/last.pt"),
        best_sha256="e" * 64, last_sha256="f" * 64, classes=["player", "referee"], imgsz=1280,
        dataset_sha256="c" * 64, epochs_completed=30, metrics={"fitness": 0.5},
        per_class_map50_95={"player": 0.6, "referee": 0.4}, metrics_source="final validation",
        resolved_config_path=resolved,
    )
    req = remote.TrainRequest.model_validate(request())
    manifest = remote.build_weights_manifest(req, result, created_at="2026-11-11T00:00:00Z")
    assert manifest["status"] == "candidate"
    assert manifest["weights"]["best"] == {
        "volume_key": "nextup-checkpoints:/players-ft-own-v1/weights/best.pt",
        "sha256": "e" * 64}
    assert manifest["dataset"]["sha256"] == "c" * 64
    assert manifest["versions"]["torch"] == "2.8.0" and manifest["gpu"] == "L4"
    assert json.loads(json.dumps(manifest)) == manifest
    from tracking import registry

    record = registry.record_from_manifest(manifest)
    assert record.status == "candidate" and record.sha256 == "e" * 64
