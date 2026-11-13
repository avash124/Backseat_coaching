import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from tracking import registry
from tracking.datasets.common import file_sha256, read_manifest

PLAYERS_V1_DATASET = registry.BACKEND_ROOT.parent / "datasets" / "ebard" / "players_v1"


def record(name="cand-1", sha="a" * 64, **overrides) -> registry.WeightsRecord:
    fields = {"name": name, "sha256": sha, "path": f"tracking/weights/{name}.pt",
              "imgsz": 1280, "classes": ("player", "referee")}
    return registry.WeightsRecord(**{**fields, **overrides})


@pytest.fixture
def workspace(tmp_path):
    """A backend-like base dir with a weights file, an eval report and a registry."""
    weights = tmp_path / "tracking" / "weights"
    weights.mkdir(parents=True)
    old = weights / "old.pt"
    old.write_bytes(b"old weights")
    new = weights / "new.pt"
    new.write_bytes(b"new weights")
    report = tmp_path / "report.json"
    report.write_text("{}", encoding="utf-8")
    path = tmp_path / "weights_registry.json"
    registry.save_registry(registry.Registry(models=(
        record("old", file_sha256(old), status="promoted"),
        record("new", file_sha256(new)),
    )), path)
    return tmp_path, path, report


def test_seeded_registry_is_valid_and_promotes_players_v1():
    loaded = registry.load_registry()
    promoted = loaded.promoted("player-detector")
    assert promoted.name == "players-v1"
    assert promoted.sha256.startswith("8ea31d1e")
    assert promoted.imgsz == 1280 and promoted.classes == ("player", "referee")
    assert promoted.dataset_sha256 == (
        "9484779986800943513c4e143d37fe686260641cb6fc637efa20f6919ecae32a")
    raw = registry.REGISTRY_PATH.read_bytes()
    assert b"\r\n" not in raw and raw.endswith(b"}\n")


def test_config_default_weights_is_the_promoted_record():
    from tracking.config import DEFAULT_WEIGHTS

    promoted = registry.load_registry().promoted("player-detector")
    assert Path(DEFAULT_WEIGHTS).resolve() == registry.local_path(promoted)


def test_seeded_hashes_match_local_files_when_present():
    promoted = registry.load_registry().promoted("player-detector")
    weights = registry.local_path(promoted)
    if not weights.is_file():
        pytest.skip("players-v1.pt is not on this machine (weights are gitignored)")
    assert registry.verify_record(promoted) == weights
    if PLAYERS_V1_DATASET.is_dir():
        assert read_manifest(PLAYERS_V1_DATASET)["dataset_sha256"] == promoted.dataset_sha256


def test_round_trip_and_invariants(tmp_path):
    path = tmp_path / "r.json"
    original = registry.Registry(models=(record("a"), record("b", status="promoted")))
    registry.save_registry(original, path)
    assert registry.load_registry(path) == original
    with pytest.raises(ValidationError, match="Duplicate"):
        registry.Registry(models=(record("a"), record("a")))
    with pytest.raises(ValidationError, match="More than one promoted"):
        registry.Registry(models=(record("a", status="promoted"), record("b", status="promoted")))
    with pytest.raises(ValidationError, match="path, volume_key or s3_key"):
        record("x", path=None)
    with pytest.raises(ValidationError):
        record("x", sha="not-a-sha")
    with pytest.raises(KeyError):
        original.get("missing")


def test_promote_is_gated_and_retires_previous(workspace):
    base, path, report = workspace
    with pytest.raises(FileNotFoundError, match="Eval report"):
        registry.promote("new", eval_report=base / "nope.json", promoted_by="me",
                         path=path, base=base)
    with pytest.raises(ValueError, match="promoted_by"):
        registry.promote("new", eval_report=report, promoted_by=" ", path=path, base=base)
    (base / "tracking" / "weights" / "new.pt").write_bytes(b"tampered")
    with pytest.raises(registry.WeightsIntegrityError):
        registry.promote("new", eval_report=report, promoted_by="me", path=path, base=base)
    (base / "tracking" / "weights" / "new.pt").write_bytes(b"new weights")
    updated = registry.promote("new", eval_report=report, promoted_by="me", path=path,
                               base=base, now="2026-11-11T00:00:00+00:00")
    assert updated.get("new").status == "promoted"
    assert updated.get("new").eval_report == "report.json"
    assert updated.get("old").status == "retired"
    assert registry.load_registry(path) == updated
    with pytest.raises(ValueError, match="only candidates"):
        registry.promote("old", eval_report=report, promoted_by="me", path=path, base=base)


def test_promote_cli_requires_confirmation(workspace, capsys):
    _, path, report = workspace
    before = path.read_bytes()
    code = registry.main(["--registry", str(path), "promote", "new", "--eval-report",
                          str(report), "--confirm", "nwe", "--by", "me"])
    assert code == 2 and path.read_bytes() == before
    assert "--confirm" in capsys.readouterr().err


def test_add_candidate_from_weights_manifest(tmp_path):
    path = tmp_path / "r.json"
    registry.save_registry(registry.Registry(), path)
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"trained")
    manifest = {
        "name": "players-ft-own-v1", "run_id": "players-ft-own-v1", "imgsz": 1280,
        "classes": ["player", "referee"], "dataset": {"sha256": "b" * 64},
        "weights": {"best": {"sha256": file_sha256(weights),
                             "volume_key": "nextup-checkpoints:/players-ft-own-v1/weights/best.pt"}},
        "start_weights": {"model": "yolo26s.pt"}, "versions": {"ultralytics": "8.4.155"},
        "created_at": "2026-11-11T00:00:00+00:00", "status": "candidate",
    }
    manifest_path = tmp_path / "weights_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    code = registry.main(["--registry", str(path), "add-candidate", "--manifest",
                          str(manifest_path), "--path", str(weights)])
    assert code == 0
    added = registry.load_registry(path).get("players-ft-own-v1")
    assert added.status == "candidate" and added.dataset_sha256 == "b" * 64
    assert added.volume_key.endswith("/weights/best.pt") and added.base_model == "yolo26s.pt"
    with pytest.raises(ValueError, match="already registered"):
        registry.add_candidate(added, path=path)
    weights.write_bytes(b"other")
    with pytest.raises(registry.WeightsIntegrityError):
        registry.record_from_manifest(manifest, weights_path=weights)
