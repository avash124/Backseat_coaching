import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from tracking.evaluate import (
    DetectionEvalConfig,
    EvalPolicyError,
    ImageRecord,
    choose_operating_point,
    coco_average_precision,
    compute_detection_metrics,
    match_detections,
    run_detection_eval,
    run_tracking_eval,
)
from tracking.labeling import boxes_to_yolo_lines, export_tracks


def record(gt, width=100, height=100):
    return ImageRecord(Path("x.jpg"), "t001-b", width, height,
                       np.asarray(gt, float).reshape(-1, 5))


def preds(*rows):
    return np.asarray(rows, float).reshape(-1, 6)


# --------------------------------------------------------------------------- AP


def test_ap_perfect_prediction():
    ap = coco_average_precision([record([[0, 10, 10, 30, 50]])],
                                [preds([10, 10, 30, 50, 0.9, 0])], [0], 100)
    assert ap[0] == {"ap50": pytest.approx(1.0), "ap50_95": pytest.approx(1.0)}


def test_ap_false_positive_ranked_above_true_positive():
    # Precision is 0.5 when recall reaches 1, and the envelope is 0.5 everywhere.
    ap = coco_average_precision(
        [record([[0, 10, 10, 30, 50]])],
        [preds([60, 60, 80, 90, 0.9, 0], [10, 10, 30, 50, 0.8, 0])],
        [0],
        100,
    )
    assert ap[0]["ap50"] == pytest.approx(0.5)


def test_ap_half_recall_uses_101_point_interpolation():
    ap = coco_average_precision(
        [record([[0, 10, 10, 30, 50], [0, 60, 10, 80, 50]])],
        [preds([10, 10, 30, 50, 0.9, 0])],
        [0],
        100,
    )
    assert ap[0]["ap50"] == pytest.approx(51 / 101)


def test_ap50_95_counts_iou_thresholds_met():
    # IoU 0.62 passes the 0.50, 0.55 and 0.60 thresholds: 3 of 10.
    ap = coco_average_precision([record([[0, 0, 0, 10, 10]])],
                                [preds([0, 0, 10, 6.2, 0.9, 0])], [0], 100)
    assert ap[0]["ap50"] == pytest.approx(1.0)
    assert ap[0]["ap50_95"] == pytest.approx(0.3)


def test_ap_is_none_without_ground_truth_and_zero_without_predictions():
    ap = coco_average_precision([record([[0, 10, 10, 30, 50]])],
                                [preds([10, 10, 30, 50, 0.9, 0])], [0, 1], 100)
    assert ap[1] == {"ap50": None, "ap50_95": None}
    ap = coco_average_precision([record([[0, 10, 10, 30, 50]])], [preds()], [0], 100)
    assert ap[0] == {"ap50": 0.0, "ap50_95": 0.0}


def _ap101(conf, is_tp, n_gt):
    order = np.argsort(-conf, kind="mergesort")
    tp = is_tp[order].astype(float)
    cum_tp, cum_fp = np.cumsum(tp), np.cumsum(1 - tp)
    recall, precision = cum_tp / n_gt, cum_tp / (cum_tp + cum_fp)
    for i in range(len(precision) - 1, 0, -1):
        precision[i - 1] = max(precision[i - 1], precision[i])
    picks = np.searchsorted(recall, np.linspace(0, 1, 101), side="left")
    return np.mean([precision[i] if i < len(precision) else 0.0 for i in picks])


def test_sweep_matching_agrees_with_cocoeval_on_random_data():
    rng = np.random.default_rng(7)
    records, predictions, confs, hits, n_gt = [], [], [], [], 0
    for _ in range(25):
        xy = rng.uniform(0, 400, (6, 2))
        gt = np.column_stack((np.zeros(6), xy, xy + rng.uniform(20, 80, (6, 2))))
        noisy = gt[:, 1:] + rng.normal(0, 6, (6, 4))
        false = np.column_stack((rng.uniform(0, 400, (3, 2)),) * 2) + [0, 0, 30, 60]
        boxes = np.vstack((noisy, false))
        conf = rng.uniform(0.01, 1, len(boxes))
        records.append(record(gt, 500, 500))
        predictions.append(np.column_stack((boxes, conf, np.zeros(len(boxes)))))
        matches = match_detections(gt[:, 1:], boxes, conf, 0.5)
        confs.append(conf)
        hits.append(matches >= 0)
        n_gt += len(gt)
    coco = coco_average_precision(records, predictions, [0], 100)[0]["ap50"]
    assert coco == pytest.approx(_ap101(np.concatenate(confs), np.concatenate(hits), n_gt))


def test_matching_is_confidence_greedy_one_to_one():
    gt = np.array([[0, 0, 10, 10], [5, 0, 15, 10]], float)
    boxes = np.array([[4, 0, 14, 10], [0, 0, 10, 10], [0, 0, 10, 10]], float)
    matches = match_detections(gt, boxes, np.array([0.9, 0.8, 0.7]), 0.5)
    # The top prediction takes its best-IoU GT (index 1); duplicates become false positives.
    assert matches.tolist() == [1, 0, -1]


# --------------------------------------------------------------------------- sweep / sizes


SWEEP_CONFIG = DetectionEvalConfig(thresholds=(0.25, 0.5, 0.75, 0.85),
                                   report_thresholds=(0.5,), recall_floors=())


def _sweep_case():
    gt = [[0, 0, 0, 10, 20], [0, 20, 0, 30, 20], [0, 40, 0, 50, 20], [0, 60, 0, 70, 20]]
    prediction = preds(
        [0, 0, 10, 20, 0.9, 0], [20, 0, 30, 20, 0.8, 0], [80, 50, 90, 70, 0.7, 0],
        [40, 0, 50, 20, 0.4, 0], [80, 80, 90, 99, 0.3, 0],
    )
    return [record(gt)], [prediction]


def test_threshold_sweep_counts():
    records, predictions = _sweep_case()
    result = compute_detection_metrics(records, predictions, ["player"], SWEEP_CONFIG)["player"]
    rows = {row["threshold"]: row for row in result["sweep"]}
    assert (rows[0.25]["tp"], rows[0.25]["fp"], rows[0.25]["fn"]) == (3, 2, 1)
    assert rows[0.25]["precision"] == pytest.approx(0.6)
    assert rows[0.25]["recall"] == pytest.approx(0.75)
    assert rows[0.5]["f1"] == pytest.approx(4 / 7)
    assert (rows[0.75]["tp"], rows[0.75]["fp"]) == (2, 0)
    assert rows[0.85]["recall"] == pytest.approx(0.25)
    assert result["at_thresholds"]["0.50"]["tp"] == 2


def test_operating_point_rule():
    records, predictions = _sweep_case()
    sweep = compute_detection_metrics(records, predictions, ["player"],
                                      SWEEP_CONFIG)["player"]["sweep"]
    met = choose_operating_point(sweep, 0.7)
    assert met["threshold"] == 0.25 and met["recall_floor_met"] is True
    # Unreachable floor: unconstrained max F1; 0.25 and 0.75 tie at 2/3 -> higher threshold.
    missed = choose_operating_point(sweep, 0.9)
    assert missed["threshold"] == 0.75 and missed["recall_floor_met"] is False
    assert choose_operating_point(sweep, None)["recall_floor_met"] is None


def test_size_buckets_use_1080p_equivalent_heights():
    # 540-px-tall frame: heights 40/60/90 px are 80/120/180 at 1080p -> small/medium/large.
    gt = [[0, 0, 0, 20, 40], [0, 100, 0, 120, 60], [0, 200, 0, 240, 90]]
    prediction = preds([0, 0, 20, 40, 0.9, 0], [200, 0, 240, 90, 0.9, 0],
                       [400, 0, 420, 55, 0.9, 0])
    config = DetectionEvalConfig(thresholds=(0.5,), report_thresholds=(0.5,))
    result = compute_detection_metrics([record(gt, 960, 540)], [prediction], ["player"], config)
    sizes = result["player"]["operating_point"]["by_size"]
    assert {k: (v["gt"], v["tp"], v["fp"]) for k, v in sizes.items()} == {
        "small": (1, 1, 0), "medium": (1, 0, 1), "large": (1, 1, 0),
    }
    assert sizes["medium"]["recall"] == 0.0 and sizes["medium"]["precision"] == 0.0
    edges = DetectionEvalConfig(size_edges_px=(70.0, 150.0), thresholds=(0.5,),
                                report_thresholds=(0.5,))
    resized = compute_detection_metrics([record(gt, 960, 540)], [prediction], ["player"], edges)
    assert resized["player"]["operating_point"]["by_size"]["small"]["gt"] == 0


# --------------------------------------------------------------------------- eval sets


def _evalset(tmp_path, name="t001-b", groups=("t001-b",), status="corrected",
             quarantined=("t001-b",), image_groups=None):
    evalsets = tmp_path / "evalsets"
    evalsets.mkdir(exist_ok=True)
    (evalsets / "quarantine.yaml").write_text(
        "groups:\n" + "".join(f"  - {g}\n" for g in quarantined) if quarantined
        else "groups: []\n"
    )
    images, labels = tmp_path / name / "images", tmp_path / name / "labels"
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    gt = np.array([[0, 10, 10, 40, 90], [1, 60, 20, 90, 95]], float)
    for index, group in enumerate(image_groups or [groups[0]] * 2):
        stem = f"{group}_{index:07d}"
        cv2.imwrite(str(images / f"{stem}.jpg"), np.zeros((100, 120, 3), np.uint8))
        (labels / f"{stem}.txt").write_text("\n".join(boxes_to_yolo_lines(gt, 120, 100)))
    (evalsets / f"{name}.yaml").write_text(
        f"name: {name}\ngroups: [{', '.join(groups)}]\nimages: {name}/images\n"
        f"labels: {name}/labels\nlabels_status: {status}\n"
    )
    return evalsets


def perfect_predictor(paths):
    return [preds([10, 10, 40, 90, 0.9, 0], [60, 20, 90, 95, 0.8, 1]) for _ in paths]


def _run(tmp_path, evalsets, name="t001-b", **kwargs):
    return run_detection_eval(
        name, tmp_path / "missing.pt", predictor=kwargs.pop("predictor", perfect_predictor),
        evalsets_dir=evalsets, base_dir=tmp_path,
        quarantine_file=evalsets / "quarantine.yaml", **kwargs,
    )


def test_quarantined_corrected_set_gives_a_valid_report(tmp_path):
    evalsets = _evalset(tmp_path)
    report = _run(tmp_path, evalsets, out=tmp_path / "report.json")
    assert report["diagnostic"] is False and report["warning"] is None
    assert report["summary"]["player.ap50"] == pytest.approx(1.0)
    assert report["summary"]["referee.op.recall"] == pytest.approx(1.0)
    assert json.loads((tmp_path / "report.json").read_text())["kind"] == "detection"


def test_non_quarantined_set_requires_diagnostic(tmp_path):
    evalsets = _evalset(tmp_path, quarantined=())
    with pytest.raises(EvalPolicyError, match="quarantine"):
        _run(tmp_path, evalsets)
    report = _run(tmp_path, evalsets, diagnostic=True)
    assert report["diagnostic"] is True
    assert "quarantine" in report["diagnostic_reasons"][0]
    assert report["warning"].startswith("DIAGNOSTIC ONLY")


def test_prelabels_require_diagnostic(tmp_path):
    evalsets = _evalset(tmp_path, status="prelabel")
    with pytest.raises(EvalPolicyError, match="pre-labels"):
        _run(tmp_path, evalsets)


def test_eval_set_contents_are_checked(tmp_path):
    evalsets = _evalset(tmp_path, image_groups=["t001-b", "t001-a"])
    with pytest.raises(ValueError, match="not in the manifest"):
        _run(tmp_path, evalsets)
    with pytest.raises(ValueError, match="must match"):
        _run(tmp_path, evalsets, name="../t001-b")
    evalsets = _evalset(tmp_path, name="t002-b", groups=("t002-b",), quarantined=("t002-b",))
    next((tmp_path / "t002-b" / "labels").glob("*.txt")).unlink()
    with pytest.raises(ValueError, match="label file"):
        _run(tmp_path, evalsets, name="t002-b")


def test_person_as_player_merges_referees_for_stock_weights(tmp_path):
    evalsets = _evalset(tmp_path)

    def stock_predictor(paths):  # a COCO model only knows 'person' (mapped to player)
        return [preds([10, 10, 40, 90, 0.9, 0], [60, 20, 90, 95, 0.8, 0]) for _ in paths]

    report = _run(tmp_path, evalsets, predictor=stock_predictor, person_as_player=True)
    assert list(report["classes"]) == ["player"]
    assert report["classes"]["player"]["gt_boxes"] == 4  # 2 players + 2 merged referees
    assert report["summary"]["player.op.recall"] == pytest.approx(1.0)


def test_baseline_deltas(tmp_path):
    evalsets = _evalset(tmp_path)
    _run(tmp_path, evalsets, out=tmp_path / "base.json")

    def miss_referee(paths):
        return [preds([10, 10, 40, 90, 0.9, 0]) for _ in paths]

    report = _run(tmp_path, evalsets, predictor=miss_referee, baseline=tmp_path / "base.json")
    baseline = report["baseline"]
    assert baseline["comparable"] is True
    assert baseline["deltas"]["player.ap50"]["delta"] == pytest.approx(0.0)
    assert baseline["deltas"]["referee.ap50"]["delta"] == pytest.approx(-1.0)


# --------------------------------------------------------------------------- tracking


FPS = 10


def _write_tracks(path, assign, frames=range(20), extra=None):
    """assign(frame, player) -> tracker id for GT player 0 (top) and 1 (bottom)."""
    payload = []
    for f in frames:
        tracks = [
            {"track_id": assign(f, p), "bbox_xyxy": [10.0 + 5 * f, 10.0 + 190 * p,
                                                      40.0 + 5 * f, 90.0 + 190 * p],
             "confidence": 0.9, "class_id": 0, "observed": True, "role": "player"}
            for p in (0, 1)
        ]
        tracks += extra(f) if extra else []
        payload.append({"frame_index": f, "timestamp_s": f / FPS, "tracks": tracks})
    path.write_text(json.dumps({"schema_version": "1.1",
                                "video": {"width": 640, "height": 360, "fps": FPS},
                                "frames": payload}))
    return path


def _write_gt(path, frames=range(20)):
    lines = [
        f"{f + 1},{p + 1},{10 + 5 * f},{10 + 190 * p},30,80,1,1,1"
        for f in frames for p in (0, 1)
    ]
    path.write_text("\n".join(lines) + "\n")
    return path


def test_perfect_tracks_score_one(tmp_path):
    gt = _write_gt(tmp_path / "gt.txt")
    tracks = _write_tracks(tmp_path / "tracks.json", lambda f, p: 7 + 2 * p)
    report = run_tracking_eval(gt, tracks, tmp_path / "report.json")
    metrics = report["metrics"]
    for key in ("HOTA", "DetA", "AssA", "IDF1", "MOTA"):
        assert metrics[key] == pytest.approx(1.0), key
    assert metrics["IDSW"] == 0 and metrics["Frag"] == 0
    assert report["diagnostic"] is False
    assert report["frames"]["evaluated"] == 20


def test_id_swap_is_counted(tmp_path):
    gt = _write_gt(tmp_path / "gt.txt")
    tracks = _write_tracks(tmp_path / "tracks.json",
                           lambda f, p: 7 + 2 * (p if f < 10 else 1 - p))
    metrics = run_tracking_eval(gt, tracks)["metrics"]
    assert metrics["IDSW"] == 2  # both GT players change tracker id once
    assert metrics["IDF1"] == pytest.approx(0.5)
    assert metrics["DetA"] == pytest.approx(1.0)
    assert metrics["AssA"] < 1.0 and metrics["HOTA"] < 1.0
    # 40 GT boxes x 0.1 s = 4 player-seconds = 1/15 player-minute.
    assert metrics["idsw_per_player_minute"] == pytest.approx(2 / (4 / 60))


def test_stride_scores_only_processed_frames(tmp_path):
    gt = _write_gt(tmp_path / "gt.txt")
    tracks = _write_tracks(tmp_path / "tracks.json", lambda f, p: p, frames=range(0, 20, 2))
    report = run_tracking_eval(gt, tracks)
    assert report["frames"]["evaluated"] == 10
    assert report["frames"]["tracker_stride"] == 2
    assert report["frames"]["gt_frames_without_tracker_output"] == 10
    assert report["frames"]["frame_duration_s"] == pytest.approx(0.2)
    assert report["metrics"]["HOTA"] == pytest.approx(1.0)


def test_non_player_roles_are_excluded_by_default(tmp_path):
    gt = _write_gt(tmp_path / "gt.txt")

    def referee(f):
        return [{"track_id": 50, "bbox_xyxy": [500.0, 100.0, 530.0, 180.0], "confidence": 0.9,
                 "class_id": 1, "observed": True, "role": "referee"}]

    tracks = _write_tracks(tmp_path / "tracks.json", lambda f, p: p, extra=referee)
    default = run_tracking_eval(gt, tracks)
    assert default["metrics"]["MOTA"] == pytest.approx(1.0)
    assert default["counts"]["tracker_boxes_excluded_by_role"] == 20
    everyone = run_tracking_eval(gt, tracks, include_all_roles=True)
    assert everyone["metrics"]["CLR_FP"] == 20


def test_self_export_round_trip_is_perfect_and_flagged_as_seed(tmp_path):
    tracks = _write_tracks(tmp_path / "tracks.json",
                           lambda f, p: 7 + 2 * (p if f < 10 else 1 - p))
    export_tracks(tracks, tmp_path / "seed", "t001-a")
    report = run_tracking_eval(tmp_path / "seed" / "gt" / "gt.txt", tracks)
    assert report["metrics"]["HOTA"] == pytest.approx(1.0)
    assert report["metrics"]["IDSW"] == 0
    assert report["diagnostic"] is True and "seed" in report["diagnostic_reasons"][0]


def test_tracking_baseline_comparison(tmp_path):
    gt = _write_gt(tmp_path / "gt.txt")
    good = _write_tracks(tmp_path / "good.json", lambda f, p: p)
    run_tracking_eval(gt, good, tmp_path / "base.json")
    swapped = _write_tracks(tmp_path / "swap.json", lambda f, p: p if f < 10 else 1 - p)
    report = run_tracking_eval(gt, swapped, baseline=tmp_path / "base.json")
    assert report["baseline"]["comparable"] is True
    assert report["baseline"]["deltas"]["IDSW"]["delta"] == 2
    assert report["baseline"]["deltas"]["IDF1"]["delta"] == pytest.approx(-0.5)
