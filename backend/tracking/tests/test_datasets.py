import io
import json
import random
import zipfile
from pathlib import Path
from types import MappingProxyType

import pytest
import yaml
from PIL import Image

from tracking.datasets import merge, own_footage, prepare_ebard, roboflow, splits, taxonomy
from tracking.datasets.common import (
    data_yaml_text,
    dataset_sha256,
    frame_stem,
    parse_frame_stem,
    read_manifest,
    remap_label_text,
    verify_dataset,
)
from tracking.datasets.sources import SourceEntry, load_sources

REPO_ROOT = Path(__file__).resolve().parents[3]
PLAYERS_V1 = REPO_ROOT / "datasets" / "ebard" / "players_v1"


def write_image(path: Path, seed: int = 0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    Image.new("RGB", (32, 32), tuple(rng.randrange(256) for _ in range(3))).save(path)
    return path


def write_frame(directory: Path, stem: str, labels: str, seed: int = 0, image_dir=None):
    write_image((image_dir or directory) / f"{stem}.jpg", seed)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{stem}.txt").write_text(labels, encoding="utf-8", newline="\n")


@pytest.fixture
def no_quarantine(tmp_path):
    path = tmp_path / "quarantine.yaml"
    path.write_text("groups: []\n", encoding="utf-8")
    return path


def own_dataset(root: Path, name="own_v1", groups=("g-a", "g-b", "g-c", "g-d"), **kwargs):
    frames = root / f"{name}-frames"
    for index, group in enumerate(groups):
        for frame in (0, 90):
            write_frame(frames, frame_stem(group, frame),
                        "0 0.5 0.5 0.2 0.4\n1 0.2 0.2 0.1 0.3\n", seed=index * 7 + frame)
    kwargs.setdefault("val_fraction", 0.25)
    kwargs.setdefault("test_fraction", 0.25)
    return own_footage.build([frames], root / name, **kwargs)


# --- taxonomy -----------------------------------------------------------------------


def test_canonical_classes_are_frozen():
    assert taxonomy.CANONICAL_CLASSES == ("player", "referee", "basketball", "hoop")
    assert dict(taxonomy.CLASS_ID) == {"player": 0, "referee": 1, "basketball": 2, "hoop": 3}


def test_unmapped_source_class_raises():
    with pytest.raises(taxonomy.TaxonomyError, match="no remap entry.*ball"):
        taxonomy.validate_source_classes("own-footage", ["player", "ball"])
    with pytest.raises(taxonomy.TaxonomyError, match="No remap table"):
        taxonomy.class_map("some-new-source", ["player"], ["player"])
    with pytest.raises(taxonomy.TaxonomyError, match="duplicate"):
        taxonomy.validate_source_classes("ebard", ["player", "player"])


def test_class_selection_order_and_ids():
    assert taxonomy.class_map("ebard", prepare_ebard.SOURCE_NAMES, ["player", "referee"]) == {
        0: None, 1: None, 2: 0, 3: 1,
    }
    assert taxonomy.class_map("ebard", prepare_ebard.SOURCE_NAMES,
                              taxonomy.CANONICAL_CLASSES) == {0: 2, 1: 3, 2: 0, 3: 1}
    assert taxonomy.output_ids(["basketball"]) == {"basketball": 0}
    assert taxonomy.is_canonical_prefix(["player", "referee"])
    assert not taxonomy.is_canonical_prefix(["basketball"])
    assert taxonomy.parse_class_list("player, referee,basketball") == (
        "player", "referee", "basketball")
    with pytest.raises(taxonomy.TaxonomyError, match="canonical order"):
        taxonomy.select_classes(["referee", "player"])
    with pytest.raises(taxonomy.TaxonomyError, match="Not canonical"):
        taxonomy.select_classes(["ball"])


# --- naming convention + labels -----------------------------------------------------


@pytest.mark.parametrize("stem,expected", [
    ("game-1_0000123", ("game-1", 123)),
    ("2026-01-12-varsity-seg03_0045000", ("2026-01-12-varsity-seg03", 45000)),
])
def test_frame_stem_round_trip(stem, expected):
    assert parse_frame_stem(stem) == expected
    assert frame_stem(*expected) == stem


@pytest.mark.parametrize("stem", [
    "Game1_0000001", "game_1_0000001", "game1_000001", "game1_00000001", "frame_000001",
    "game1-0000001", "game1_0000001.rf",
])
def test_frame_stem_rejects_bad_names(stem):
    with pytest.raises(ValueError):
        parse_frame_stem(stem)


def test_remap_label_text_validates_and_reports():
    result = remap_label_text(
        "2 0.5 0.5 0.1 0.2\n\n0 0.1 0.1 0.01 0.02\n3 0.9 0.9 0 0.1\n",
        {0: None, 1: None, 2: 0, 3: 1}, label_ref="x.txt",
    )
    assert result.text == "0 0.5 0.5 0.1 0.2\n"
    assert result.dropped == {0: 1}
    assert result.excluded[0]["reason"] == "zero-area box"
    for bad in ("9 0.5 0.5 0.1 0.1", "0 0.5 0.5 0.1", "0 1.5 0.5 0.1 0.1", "0 nan 0.5 0.1 0.1"):
        with pytest.raises(ValueError):
            remap_label_text(bad, {0: 0}, label_ref="x.txt")


# --- splits + quarantine ------------------------------------------------------------


def legacy_split(games, seed):
    games = sorted(set(games))
    random.Random(seed).shuffle(games)
    held = max(1, len(games) // 10)
    return {"train": games[2 * held:], "val": games[:held], "test": games[held:2 * held]}


@pytest.mark.parametrize("count", [3, 9, 10, 31, 60])
def test_split_groups_reproduces_original_policy_and_is_disjoint(count):
    groups = [f"game-{i:02d}" for i in range(count)]
    result = splits.split_groups(groups, seed=42)
    assert result == legacy_split(groups, 42)
    members = [g for split in result.values() for g in split]
    assert sorted(members) == sorted(groups) and len(members) == len(set(members))
    assert splits.split_groups(reversed(groups), seed=42) == result


def test_split_groups_needs_enough_groups():
    with pytest.raises(ValueError, match="cannot fill train"):
        splits.split_groups(["a", "b"])
    with pytest.raises(ValueError, match="val_fraction"):
        splits.split_groups(["a", "b", "c"], val_fraction=0)
    assert splits.split_groups(["a", "b"], val_fraction=0.5, test_fraction=0)["test"] == []


def test_check_splits_rejects_leaks_and_quarantine():
    with pytest.raises(splits.SplitLeakError, match="game-1"):
        splits.check_splits({"train": ["game-1"], "val": ["game-1"]})
    for split in ("train", "val"):
        with pytest.raises(splits.QuarantineError, match="eval-1"):
            splits.check_splits({split: ["eval-1"], "test": []}, quarantine={"eval-1"})
    splits.check_splits({"train": ["game-1"], "test": ["eval-1"]}, quarantine={"eval-1"})
    with pytest.raises(splits.QuarantineError):
        splits.split_groups(["a", "b", "c", "eval-1"], quarantine={"eval-1"})


def test_quarantine_file_loading(tmp_path):
    assert splits.QUARANTINE_PATH == (
        Path(splits.__file__).resolve().parents[1] / "evalsets" / "quarantine.yaml")
    with pytest.warns(UserWarning, match="not found"):
        assert splits.load_quarantine(tmp_path / "missing.yaml") == frozenset()
    path = tmp_path / "q.yaml"
    path.write_text("groups:\n  - eval-a\n  - id: eval-b\n", encoding="utf-8")
    assert splits.load_quarantine(path) == {"eval-a", "eval-b"}
    assert splits.quarantine_record(path)["groups"] == ["eval-a", "eval-b"]
    path.write_text("groups: [Not_Valid]\n", encoding="utf-8")
    with pytest.raises(ValueError):
        splits.load_quarantine(path)
    path.write_text("", encoding="utf-8")
    assert splits.load_quarantine(path) == frozenset()


# --- own-footage builder ------------------------------------------------------------


def test_builder_parses_cvat_export_and_remaps_by_name(tmp_path, no_quarantine):
    export = tmp_path / "cvat"
    frames = tmp_path / "frames"
    (export / "obj_train_data").mkdir(parents=True)
    (export / "obj.names").write_text("referee\nplayer\n", encoding="utf-8")
    (export / "obj.data").write_text("classes = 2\n", encoding="utf-8")
    (export / "train.txt").write_text("data/obj_train_data/x.jpg\n", encoding="utf-8")
    groups = ["game-a", "game-b", "game-c"]
    for i, group in enumerate(groups):
        # CVAT ids: 0 = referee, 1 = player. Images come from the frames folder.
        write_frame(export / "obj_train_data", frame_stem(group, 30),
                    "1 0.5 0.5 0.2 0.4\n0 0.3 0.3 0.1 0.2\n", seed=i, image_dir=frames)
    write_image(frames / f"{frame_stem('unlabeled-pool', 1)}.jpg")  # pool extras are fine
    manifest = own_footage.build([export], tmp_path / "own_v1", image_dirs=[frames],
                                 quarantine_path=no_quarantine)
    out = tmp_path / "own_v1"
    assert manifest["names"] == ["player", "referee"]
    assert manifest["sources"][0]["kind"] == "cvat-yolo-1.1"
    assert manifest["sources"][0]["class_map"] == {0: 1, 1: 0}
    assert sorted(g for split in manifest["groups"].values() for g in split) == groups
    label = next((out / "train" / "labels").glob("*.txt")).read_text()
    assert label == "0 0.5 0.5 0.2 0.4\n1 0.3 0.3 0.1 0.2\n"
    assert manifest["boxes"]["train"] == {"player": 1, "referee": 1}
    assert manifest["dataset_sha256"] == dataset_sha256(out)
    assert yaml.safe_load((out / "data.yaml").read_text())["path"] == "/datasets/own_v1"


@pytest.mark.parametrize("problem,match", [
    ("bad-name", "naming convention"),
    ("unlabeled", "without a label file"),
    ("orphan-label", "Labels without an image"),
    ("unmapped-class", "no remap entry"),
    ("missing-class", "does not label"),
])
def test_builder_rejects_bad_inputs(tmp_path, no_quarantine, problem, match):
    export = tmp_path / "export"
    for i, group in enumerate(["game-a", "game-b", "game-c"]):
        write_frame(export, frame_stem(group, 0), "0 0.5 0.5 0.2 0.2\n", seed=i)
    classes = taxonomy.DEFAULT_CLASSES
    if problem == "bad-name":
        write_frame(export, "frame_000001", "")
    elif problem == "unlabeled":
        write_image(export / f"{frame_stem('game-a', 30)}.jpg")
    elif problem == "orphan-label":
        (export / f"{frame_stem('game-a', 60)}.txt").write_text("", encoding="utf-8")
    elif problem == "unmapped-class":
        (export / "obj.names").write_text("player\nball\n", encoding="utf-8")
    elif problem == "missing-class":
        (export / "obj.names").write_text("player\nreferee\n", encoding="utf-8")
        classes = ("player", "referee", "basketball")
    with pytest.raises(ValueError, match=match):
        own_footage.build([export], tmp_path / "out", classes=classes,
                          quarantine_path=no_quarantine)
    assert not (tmp_path / "out").exists()


def test_builder_enforces_quarantine(tmp_path):
    quarantine = tmp_path / "quarantine.yaml"
    quarantine.write_text("groups: [g-b]\n", encoding="utf-8")
    with pytest.raises(splits.QuarantineError, match="g-b"):
        own_dataset(tmp_path, quarantine_path=quarantine)
    assert not (tmp_path / "own_v1").exists()


def test_manifest_hash_is_stable_and_content_sensitive(tmp_path, no_quarantine):
    first = own_dataset(tmp_path / "a", quarantine_path=no_quarantine)
    second = own_dataset(tmp_path / "b", quarantine_path=no_quarantine)
    assert first["dataset_sha256"] == second["dataset_sha256"]
    directory = tmp_path / "a" / "own_v1"
    (directory / "train" / "labels.cache").write_bytes(b"ultralytics cache")
    assert verify_dataset(directory)["dataset_sha256"] == first["dataset_sha256"]
    label = next((directory / "val" / "labels").glob("*.txt"))
    label.write_text(label.read_text() + "1 0.5 0.5 0.1 0.1\n", encoding="utf-8")
    assert dataset_sha256(directory) != first["dataset_sha256"]
    with pytest.raises(ValueError, match="changed after it was built"):
        verify_dataset(directory)
    with pytest.raises(ValueError, match="was requested"):
        verify_dataset(tmp_path / "b" / "own_v1", "0" * 64)


@pytest.mark.skipif(not PLAYERS_V1.is_dir(), reason="eBAR players_v1 not on this machine")
def test_hash_matches_existing_players_v1_manifest():
    assert dataset_sha256(PLAYERS_V1) == read_manifest(PLAYERS_V1)["dataset_sha256"]


# --- prepare_ebard ------------------------------------------------------------------


def fake_ebard(root: Path, games: int = 10) -> Path:
    (root).mkdir(parents=True)
    (root / "data.yaml").write_text(yaml.safe_dump({
        "names": prepare_ebard.SOURCE_NAMES, "train": "train/images",
        "val": "test/images", "test": "val/images",
    }), encoding="utf-8")
    for index in range(games):
        split = ("train", "val", "test")[index % 3]
        stem = f"aaa-vs-bbb-{index:010d}_000{index:03d}"
        write_frame(root / split / "labels", stem,
                    "0 0.5 0.5 0.02 0.03\n1 0.5 0.1 0.05 0.05\n2 0.4 0.6 0.1 0.3\n"
                    "3 0.6 0.6 0.1 0.3\n2 0.9 0.9 0.1 0\n", seed=index,
                    image_dir=root / split / "images")
    return root


def test_prepare_ebard_default_keeps_legacy_output(tmp_path, no_quarantine):
    source = fake_ebard(tmp_path / "yolo")
    manifest = prepare_ebard.prepare(source, tmp_path / "players_v1",
                                     quarantine_path=no_quarantine)
    assert manifest["names"] == ["player", "referee"]
    assert manifest["class_map"] == {0: None, 1: None, 2: 0, 3: 1}
    assert manifest["source_version"] is None and manifest["source_license"] is None
    assert list(manifest) == [
        "source_directory", "source_version", "source_license", "seed", "split_policy",
        "evaluation_domain", "class_map", "names", "games", "images", "boxes",
        "excluded_annotations", "dataset_sha256",
    ]
    assert len(manifest["excluded_annotations"]) == 10
    label = next((tmp_path / "players_v1" / "train" / "labels").glob("*.txt")).read_text()
    assert label == "0 0.4 0.6 0.1 0.3\n1 0.6 0.6 0.1 0.3\n"
    assert manifest["dataset_sha256"] == dataset_sha256(tmp_path / "players_v1")


def test_prepare_ebard_full_classes_and_provenance(tmp_path, no_quarantine, capsys):
    source = fake_ebard(tmp_path / "yolo")
    manifest = prepare_ebard.main([
        "--source", str(source), "--destination", str(tmp_path / "full_v1"),
        "--classes", "player,referee,basketball,hoop", "--license", "CC-BY-4.0",
        "--quarantine", str(no_quarantine),
    ])
    assert "WARNING: --source-version not set" in capsys.readouterr().out
    assert manifest["names"] == list(taxonomy.CANONICAL_CLASSES)
    assert manifest["source_license"] == "CC-BY-4.0"
    label = next((tmp_path / "full_v1" / "val" / "labels").glob("*.txt")).read_text()
    assert label == ("2 0.5 0.5 0.02 0.03\n3 0.5 0.1 0.05 0.05\n0 0.4 0.6 0.1 0.3\n"
                     "1 0.6 0.6 0.1 0.3\n")
    assert yaml.safe_load((tmp_path / "full_v1" / "data.yaml").read_text())["names"] == list(
        taxonomy.CANONICAL_CLASSES)


# --- merge --------------------------------------------------------------------------


def test_merge_oversamples_own_footage_and_records_it(tmp_path, no_quarantine):
    public = prepare_ebard.prepare(fake_ebard(tmp_path / "yolo"), tmp_path / "public_v1",
                                   quarantine_path=no_quarantine)
    own = own_dataset(tmp_path, quarantine_path=no_quarantine)
    manifest = merge.merge(
        [merge.MergeSource(tmp_path / "public_v1", "public"),
         merge.MergeSource(tmp_path / "own_v1", "own")],
        tmp_path / "merged_v1", oversample_own=3, quarantine_path=no_quarantine,
    )
    out = tmp_path / "merged_v1"
    lines = (out / "train.txt").read_text().splitlines()
    assert len(lines) == manifest["train_list_entries"] == (
        public["images"]["train"] + 3 * own["images"]["train"])
    assert manifest["oversampling"] == {"own": 3, "public": 1}
    assert manifest["images"]["val"] == own["images"]["val"]  # public val dropped
    assert manifest["images"].get("test", 0) == own["images"]["test"]  # public test never
    assert manifest["effective_train_boxes"]["player"] == (
        public["boxes"]["train"][0] + 3 * own["boxes"]["train"]["player"])
    assert manifest["groups"]["val"] == own["groups"]["val"]
    assert verify_dataset(out)["dataset_sha256"] == dataset_sha256(out)
    (out / "train.txt").write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    assert dataset_sha256(out) != manifest["dataset_sha256"]  # the list is hashed


def test_ultralytics_reads_oversampled_train_list(tmp_path, no_quarantine):
    pytest.importorskip("ultralytics")
    from ultralytics.data.dataset import YOLODataset
    from ultralytics.data.utils import check_det_dataset

    own_dataset(tmp_path, quarantine_path=no_quarantine)
    manifest = merge.merge([merge.MergeSource(tmp_path / "own_v1", "own")],
                           tmp_path / "merged_v1", oversample_own=2,
                           quarantine_path=no_quarantine)
    data_yaml = tmp_path / "merged_v1" / "resolved.yaml"
    config = yaml.safe_load((tmp_path / "merged_v1" / "data.yaml").read_text())
    data_yaml.write_text(yaml.safe_dump({**config, "path": str(tmp_path / "merged_v1")}))
    data = check_det_dataset(str(data_yaml))
    dataset = YOLODataset(img_path=data["train"], data=data, imgsz=32, augment=False)
    assert len(dataset) == manifest["train_list_entries"] == 8


def test_merge_rejects_unsafe_sources(tmp_path, no_quarantine):
    prepare_ebard.prepare(fake_ebard(tmp_path / "yolo"), tmp_path / "public_v1",
                          quarantine_path=no_quarantine)
    own_dataset(tmp_path, quarantine_path=no_quarantine)
    sources = [merge.MergeSource(tmp_path / "public_v1", "public"),
               merge.MergeSource(tmp_path / "own_v1", "own")]
    with pytest.raises(ValueError, match="does not label"):
        merge.merge(sources, tmp_path / "m1", classes=("player", "referee", "basketball"),
                    quarantine_path=no_quarantine)
    with pytest.raises(ValueError, match="own-footage source"):
        merge.merge(sources[:1], tmp_path / "m2", quarantine_path=no_quarantine)
    quarantine = tmp_path / "q.yaml"
    quarantine.write_text("groups: [g-a, g-b, g-c, g-d]\n", encoding="utf-8")
    with pytest.raises(splits.QuarantineError):
        merge.merge(sources, tmp_path / "m3", quarantine_path=quarantine)
    label = next((tmp_path / "own_v1" / "train" / "labels").glob("*.txt"))
    label.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="changed after it was built"):
        merge.merge(sources, tmp_path / "m4", quarantine_path=no_quarantine)


# --- roboflow (network mocked) + sources.yaml ---------------------------------------


@pytest.fixture
def roboflow_remap(monkeypatch):
    tables = dict(taxonomy.SOURCE_REMAPS)
    tables["test-rf"] = MappingProxyType({"ball": "basketball", "player": "player",
                                          "ref": "referee", "crowd": None})
    monkeypatch.setattr(taxonomy, "SOURCE_REMAPS", MappingProxyType(tables))


def roboflow_zip(names=("ball", "player", "ref", "crowd"), extra=None) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as bundle:
        bundle.writestr("data.yaml", yaml.safe_dump({"names": list(names), "nc": len(names)}))
        for split, indices in (("train", range(4)), ("valid", range(4, 6))):
            for i in indices:
                stem = f"game{i % 3}-clip_{i:04d}_jpg.rf.{i:08x}"
                image = io.BytesIO()
                Image.new("RGB", (16, 16), (i * 20, 0, 0)).save(image, format="JPEG")
                bundle.writestr(f"{split}/images/{stem}.jpg", image.getvalue())
                bundle.writestr(f"{split}/labels/{stem}.txt",
                                "0 0.5 0.5 0.1 0.1\n1 0.3 0.3 0.2 0.4\n2 0.7 0.7 0.2 0.4\n"
                                "3 0.1 0.1 0.1 0.1\n")
        for name, content in (extra or {}).items():
            bundle.writestr(name, content)
    return buffer.getvalue()


def rf_entry(**overrides) -> SourceEntry:
    entry = {
        "name": "test-rf", "source": "roboflow", "license": "CC-BY-4.0",
        "license_status": "permissive", "version": 2,
        "remap": "test-rf", "classes": ["ball", "player", "ref", "crowd"], "domain": "broadcast",
        "roboflow": {"workspace": "ws", "project": "hoops", "version": 2},
    }
    return SourceEntry.model_validate({**entry, **overrides})


def test_roboflow_download_is_pinned_content_addressed_and_skipped(tmp_path, roboflow_remap):
    calls = []

    def fetch(url):
        calls.append(url)
        if url.startswith(roboflow.API_URL):
            return json.dumps({"export": {"link": "https://example.invalid/export.zip"}}).encode()
        return roboflow_zip()

    coords = rf_entry().roboflow
    raw = roboflow.download(coords, tmp_path / "raw", api_key="secret", fetch=fetch)
    assert raw == roboflow.raw_dir(tmp_path / "raw", coords)
    assert raw.name.startswith("hoops-v2-") and roboflow.coordinates_key(coords)[:12] in raw.name
    assert calls[0] == f"{roboflow.API_URL}/ws/hoops/2/yolov8?api_key=secret"
    assert json.loads((raw / roboflow.DOWNLOAD_MARKER).read_text())["coordinates"]["version"] == 2

    def no_network(url):
        raise AssertionError("already downloaded; must not fetch")

    assert roboflow.download(coords, tmp_path / "raw", api_key="secret", fetch=no_network) == raw


def test_roboflow_import_rederives_ids_through_taxonomy(tmp_path, roboflow_remap, no_quarantine):
    raw = tmp_path / "raw"
    roboflow._safe_extract(roboflow_zip(), raw)
    manifest = roboflow.import_export(raw, rf_entry(), tmp_path / "rf_v2",
                                      classes=taxonomy.CANONICAL_CLASSES[:3],
                                      quarantine_path=no_quarantine)
    assert manifest["names"] == ["player", "referee", "basketball"]
    assert manifest["class_map"] == {0: 2, 1: 0, 2: 1, 3: None}
    assert manifest["groups"] == {"train": ["test-rf"], "val": [], "test": []}
    label = next((tmp_path / "rf_v2" / "train" / "labels").glob("*.txt")).read_text()
    assert label == "2 0.5 0.5 0.1 0.1\n0 0.3 0.3 0.2 0.4\n1 0.7 0.7 0.2 0.4\n"
    grouped = rf_entry(roboflow={"workspace": "ws", "project": "hoops", "version": 2,
                                 "group_pattern": r"^(?P<group>game\d)-clip"})
    manifest = roboflow.import_export(raw, grouped, tmp_path / "rf_grouped",
                                      quarantine_path=no_quarantine)
    assert sorted(g for s in manifest["groups"].values() for g in s) == ["game0", "game1", "game2"]
    with pytest.raises(ValueError, match="differ from sources.yaml"):
        roboflow.import_export(raw, rf_entry(classes=["ball", "player", "ref"],
                                             remap="test-rf"), tmp_path / "x")


def test_roboflow_license_gate_and_unsafe_archive(tmp_path, roboflow_remap):
    sources = tmp_path / "sources.yaml"
    entry = rf_entry(license="unspecified on the project page",
                     license_status="unknown").model_dump(mode="json")
    sources.write_text(yaml.safe_dump({"sources": [entry]}), encoding="utf-8")

    def fetch(url):
        raise AssertionError("license gate must stop before any download")

    with pytest.raises(roboflow.LicenseError):
        roboflow.pull("test-rf", tmp_path / "out", raw_root=tmp_path / "raw",
                      sources_path=sources, api_key="k", fetch=fetch)
    with pytest.raises(ValueError, match="Unsafe path"):
        roboflow._safe_extract(roboflow_zip(extra={"../evil.txt": "x"}), tmp_path / "unsafe")
    with pytest.raises(ValueError):  # unmapped class fails at the sources.yaml entry
        rf_entry(classes=["ball", "player", "ref", "coach"])


def test_sources_yaml_records_ebard_provenance_as_restricted():
    ebard = load_sources()["ebard"]
    assert ebard.classes == tuple(prepare_ebard.SOURCE_NAMES)
    assert ebard.repo == "GabrieleGiudici/E-BARD-detection"
    assert ebard.revision == ebard.version == "00563215490c9a9642797b1495ea178535b3f59c"
    assert ebard.license_status == "restricted" and not ebard.shippable
    assert ebard.license.startswith("labels CC BY 4.0; frame rights unresolved (NBA broadcast)")
    assert data_yaml_text("x", ["player"]).startswith("path: /datasets/x\n")
