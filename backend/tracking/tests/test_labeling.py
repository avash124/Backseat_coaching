import json
import zipfile
from fractions import Fraction

import av
import cv2
import numpy as np
import pytest
import yaml

from tracking.labeling import (
    CLASS_NAMES,
    boxes_to_yolo_lines,
    check_registered_span,
    export_tracks,
    frame_stem,
    import_cvat_mot,
    import_cvat_yolo,
    max_sample_count,
    parse_frame_stem,
    plan_samples,
    read_mot,
    read_yolo_labels,
    sample,
)

GAP_TOLERANCE = 1e-6


def test_frame_names_follow_the_shared_convention():
    assert frame_stem("g001-b", 6152) == "g001-b_0006152"
    assert parse_frame_stem("g001-b_0006152") == ("g001-b", 6152)
    for bad in ("G001", "g001_b", "", "g 1"):
        with pytest.raises(ValueError):
            frame_stem(bad, 1)
    with pytest.raises(ValueError):
        parse_frame_stem("det-vs-gsw-0022400919_001103")  # six digits: public data only
    assert parse_frame_stem("det-vs-gsw-0022400919_001103", strict=False) == (
        "det-vs-gsw-0022400919",
        1103,
    )


def test_class_ids_are_frozen():
    assert CLASS_NAMES[:2] == ("player", "referee")


@pytest.mark.parametrize(("gap", "expected"), [(4.0, 26), (2.0, 52), (1.0, 104)])
def test_max_density_sampling_keeps_spacing(gap, expected):
    times = np.arange(3106) / 30 + 205.0  # g001-b: 103.5 s at 30 fps
    picks = plan_samples(times, gap)
    assert len(picks) == expected == max_sample_count(times, gap)
    assert np.diff(times[picks]).min() >= gap - GAP_TOLERANCE
    assert times[picks][0] < 205.0 + gap and times[picks][-1] > times[-1] - gap


def test_sampling_with_slack_prefers_novel_frames():
    times = np.arange(601) / 10  # 60 s at 10 fps
    signatures = np.zeros((len(times), 18, 32), np.float32)
    distinct = (times >= 14.0) & (times <= 14.5)
    signatures[distinct] = 100
    plain = times[plan_samples(times, 4.0, None, [(0, None, 5)])]
    diverse = times[plan_samples(times, 4.0, signatures, [(0, None, 5)])]
    assert len(plain) == len(diverse) == 5
    assert not np.any((plain >= 14.0) & (plain <= 14.5))
    assert np.any((diverse >= 14.0) & (diverse <= 14.5))
    for picked in (plain, diverse):
        assert np.diff(picked).min() >= 4.0 - GAP_TOLERANCE
        assert picked[-1] > 45  # spread over the range, not front-loaded


def test_quotas_respect_spacing_across_ranges():
    times = np.arange(1001) / 10  # 100 s
    picks = times[plan_samples(times, 4.0, None, [(0, 50, 3), (50, None, None)])]
    first = picks[picks < 50]
    assert len(first) == 3
    assert np.diff(picks).min() >= 4.0 - GAP_TOLERANCE
    assert len(picks) - 3 == max_sample_count(times[times >= max(50, first[-1] + 4)], 4.0)


def test_registered_span_guard():
    registry = {"g001-b": {"video_sha256": "abc", "start_s": 205.0, "end_s": None}}
    check_registered_span("g001-b", "abc", [(205.0, None)], registry)
    check_registered_span("g009-a", "zzz", [(0.0, 1.0)], registry)  # unregistered: allowed
    with pytest.raises(ValueError, match="different video"):
        check_registered_span("g001-b", "def", [(205.0, None)], registry)
    with pytest.raises(ValueError, match="span"):
        check_registered_span("g001-b", "abc", [(100.0, None)], registry)


def test_yolo_lines_round_trip_and_clip():
    boxes = np.array([[0, 10, 20, 110, 220], [1, -5, 900, 50, 1200], [0, 5, 5, 5, 9]], float)
    lines = boxes_to_yolo_lines(boxes, 1920, 1080)
    assert len(lines) == 2  # the zero-width box is dropped, the overflowing one clipped
    assert all(0 <= float(v) <= 1 for line in lines for v in line.split()[1:])


def test_yolo_label_file_round_trip(tmp_path):
    boxes = np.array([[0, 10.5, 20.25, 110, 220], [1, 300, 400, 340, 500]], float)
    path = tmp_path / "x.txt"
    path.write_text("\n".join(boxes_to_yolo_lines(boxes, 1920, 1080)) + "\n")
    assert np.allclose(read_yolo_labels(path, 1920, 1080), boxes, atol=0.01)
    path.write_text("0 0.5 0.5 0 0.1\n")
    with pytest.raises(ValueError, match="zero-area"):
        read_yolo_labels(path, 1920, 1080)


def _write_video(path, frames=120, fps=10, size=(64, 48)):
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.width, stream.height = size
        stream.pix_fmt = "yuv420p"
        stream.codec_context.time_base = Fraction(1, fps)
        for index in range(frames):
            level = 2 * index  # each frame's gray level encodes its index
            image = np.full((size[1], size[0], 3), level, np.uint8)
            frame = av.VideoFrame.from_ndarray(image, format="bgr24")
            frame.pts = index
            frame.time_base = Fraction(1, fps)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


class FakeDetector:
    def __init__(self):
        self.names = {0: "player", 1: "referee"}
        self.nms = False
        self.settings = {"imgsz": 64, "conf": 0.25, "device": "cpu", "max_det": 100}

    def __call__(self, paths):
        return [
            np.array([[4, 4, 20, 40, 0.9, 0], [30, 5, 40, 30, 0.6, 1]], np.float32)
            for _ in paths
        ]


def test_sample_writes_convention_named_frames_and_packages(tmp_path):
    video = tmp_path / "clip.mp4"
    _write_video(video)
    out = tmp_path / "batch"
    batch = sample(video, "t001-a", out, min_gap_s=4.0, detector=FakeDetector(),
                   group_registry={})
    frames = batch["frames"]
    assert len(frames) == 3 == batch["selection"]["max_frames_at_gap_s"]["4"]
    assert batch["selection"]["max_frames_at_gap_s"]["1"] == 12
    assert np.diff([f["timestamp_s"] for f in frames]).min() >= 4.0 - GAP_TOLERANCE
    for frame in frames:
        stem = frame_stem("t001-a", frame["frame_index"])
        assert frame["image"] == f"{stem}.jpg"
        image = cv2.imread(str(out / "images" / frame["image"]))
        assert abs(image.mean() - 2 * frame["frame_index"]) < 4  # right decoded frame
        labels = (out / "labels" / f"{stem}.txt").read_text().splitlines()
        assert [line.split()[0] for line in labels] == ["0", "1"]
    data = yaml.safe_load((out / "data.yaml").read_text())
    assert data["names"] == ["player", "referee"] and data["nc"] == 2
    with zipfile.ZipFile(out / "cvat_yolo11_annotations.zip") as archive:
        names = set(archive.namelist())
        assert {"obj.names", "obj.data", "train.txt"} <= names
        assert archive.read("obj.names").decode().split() == ["player", "referee"]
        assert {f"obj_train_data/{f['image'][:-4]}.txt" for f in frames} <= names
    with zipfile.ZipFile(out / "cvat_images.zip") as archive:
        assert sorted(archive.namelist()) == sorted(f["image"] for f in frames)
    assert json.loads((out / "batch.json").read_text())["prelabel"]["status"] == "prelabel"
    with pytest.raises(FileExistsError):
        sample(video, "t001-a", out, detector=FakeDetector(), group_registry={})


def _tracks_artifact():
    def track(track_id, x, role=None, observed=True):
        payload = {"track_id": track_id, "bbox_xyxy": [x, 10.0, x + 20.0, 60.0],
                   "confidence": 0.9, "class_id": 0, "observed": observed}
        if role:
            payload["role"] = role
        return payload

    frames = [
        {"frame_index": i, "timestamp_s": i / 30,
         "tracks": [track(0, 10.0 + i, "player"), track(1, 200.0, "referee"),
                    track(2, 400.0 - i)]}
        for i in range(3)
    ]
    return {"schema_version": "1.1", "video": {"width": 640, "height": 360, "fps": 30.0},
            "provenance": {}, "frames": frames}


def test_export_tracks_writes_mot_and_cvat_zip(tmp_path):
    tracks = tmp_path / "tracks.json"
    tracks.write_text(json.dumps(_tracks_artifact()))
    seed = export_tracks(tracks, tmp_path / "seed", "t001-a")
    rows = read_mot(tmp_path / "seed" / "gt" / "gt.txt")
    assert seed["rows"] == len(rows) == 6  # the referee-role track is excluded
    assert set(rows[:, 0]) == {1, 2, 3}  # 1-based frames
    assert set(rows[:, 1]) == {1, 3}  # track_id + 1
    first = rows[(rows[:, 0] == 1) & (rows[:, 1] == 1)][0]
    assert first.tolist() == [1, 1, 10, 10, 20, 50, 1, 1, 1]
    with zipfile.ZipFile(tmp_path / "seed" / "cvat_mot11.zip") as archive:
        assert set(archive.namelist()) == {"gt/gt.txt", "gt/labels.txt"}
    everyone = export_tracks(tracks, tmp_path / "all", "t001-a", include_all_roles=True)
    assert everyone["rows"] == 9


def test_import_cvat_mot_round_trip(tmp_path):
    tracks = tmp_path / "tracks.json"
    tracks.write_text(json.dumps(_tracks_artifact()))
    export_tracks(tracks, tmp_path / "seed", "t001-a")
    export = tmp_path / "export.zip"
    with zipfile.ZipFile(export, "w") as archive:
        gt = (tmp_path / "seed" / "gt" / "gt.txt").read_text()
        # CVAT numbers classes by labels.txt order; add a label we do not track.
        archive.writestr("gt/labels.txt", "ball\nplayer\n")
        archive.writestr("gt/gt.txt", gt.replace(",1,1\n", ",2,1\n") + "1,99,5,5,4,4,1,1,1\n")
    summary = import_cvat_mot(export, tmp_path / "gt" / "gt.txt")
    assert summary["rows_kept"] == 6 and summary["rows_dropped"] == 1
    imported = read_mot(tmp_path / "gt" / "gt.txt")
    seeded = read_mot(tmp_path / "seed" / "gt" / "gt.txt")
    assert np.array_equal(imported, seeded)


def test_import_cvat_yolo_remaps_classes_by_name(tmp_path):
    export = tmp_path / "export.zip"
    with zipfile.ZipFile(export, "w") as archive:
        archive.writestr("obj.names", "referee\nplayer\ncoach\n")
        archive.writestr("obj.data", "classes = 3\n")
        archive.writestr("obj_train_data/t001-a_0000010.txt",
                         "0 0.5 0.5 0.1 0.2\n1 0.2 0.2 0.1 0.1\n2 0.8 0.8 0.1 0.1\n")
        archive.writestr("obj_train_data/t001-a_0000050.txt", "")
    with pytest.raises(ValueError, match="coach"):
        import_cvat_yolo(export, tmp_path / "out")
    summary = import_cvat_yolo(export, tmp_path / "out", group_id="t001-a",
                               drop_labels=["coach"])
    assert summary["boxes"] == {"referee": 1, "player": 1, "dropped": 1}
    lines = (tmp_path / "out" / "labels" / "t001-a_0000010.txt").read_text().splitlines()
    assert [line.split()[0] for line in lines] == ["1", "0"]
    assert (tmp_path / "out" / "labels" / "t001-a_0000050.txt").read_text() == ""
    images = tmp_path / "images"
    images.mkdir()
    for stem in ("t001-a_0000010", "t001-a_0000050", "t001-a_0000090"):
        (images / f"{stem}.jpg").write_bytes(b"")
    with pytest.raises(ValueError, match="Images without labels"):
        import_cvat_yolo(export, tmp_path / "out2", images=images, drop_labels=["coach"])
    with pytest.raises(ValueError, match="expected t009-z"):
        import_cvat_yolo(export, tmp_path / "out3", group_id="t009-z", drop_labels=["coach"])
