"""Evaluation harness: detector and tracker accuracy on our own, quarantined footage.

The eval set is **our own footage**, never a public dataset's val split. Public sets are
broadcast-angle; scoring against them measures how well we do on someone else's camera. A
model can gain 6 points of mAP on a Roboflow val split and lose accuracy on real uploads, and
only this file would catch it. Public or uncorrected sets run only with ``--diagnostic``, and
the report then says so in ``diagnostic`` / ``diagnostic_reasons`` / ``warning``.

Commands (run from ``backend/``)::

    python -m tracking.evaluate detect --evalset NAME --weights PATH --out FILE [--baseline FILE]
    python -m tracking.evaluate track --gt gt.txt --tracks tracks.json --out FILE [--baseline FILE]

evalsets/ (README)
==================
``tracking/evalsets/`` holds small, durable text files. Images and labels live elsewhere.

- ``quarantine.yaml``: groups that must never enter any training set. A group is added
  **before** it is labeled. The dataset builder refuses these groups, and this module only
  scores non-diagnostic reports on them.
- ``<name>.yaml``: one manifest per detection eval set. ``detect --evalset <name>`` takes
  the name, not a path, so a public val split cannot be passed by accident.

  =================  =============================================================
  ``name``           must equal the file stem
  ``groups``         group ids the set covers; every image stem must belong to one
  ``images``         directory of ``{group_id}_{frame_index:07d}.jpg`` images
  ``labels``         directory of YOLO txt labels with the same stems (one per image)
  ``labels_status``  ``prelabel`` (uncorrected model output) or ``corrected``
  ``classes``        canonical class names to score (default ``[player, referee]``)
  ``frame_names``    ``strict`` (default, 7-digit frame index) or ``loose`` (public data)
  =================  =============================================================

  Relative paths resolve against ``backend/``. A report is non-diagnostic only when every
  group is quarantined **and** ``labels_status`` is ``corrected``.

Labeling workflow, step by step
-------------------------------
Conventions (group ids, file names, class ids, MOT columns) are in ``tracking/labeling.py``.

0. **Choose groups.** A group is a game, or a game segment while there is only one game. Add
   every eval group to ``quarantine.yaml`` first.
1. **Sample.** Choose frames >= 4 s apart, pre-label them with the current weights, and
   package them for annotation::

       python -m tracking.labeling sample --video "../Basketball Game(1)_cut.mp4" \\
           --group g001-b --start-s 205 --out ../datasets/own/labeling/g001-batch1/g001-b

   This writes ``images/``, ``labels/`` (pre-labels), ``data.yaml``, ``batch.json``,
   ``cvat_images.zip``, ``cvat_yolo11_annotations.zip`` and ``yolo_dataset.zip``.
2. **Import into CVAT.** Create a task with the labels ``player`` and ``referee`` (exact
   names; order does not matter) and upload ``cvat_images.zip`` as the data. Then open
   Actions -> Upload annotations -> "YOLO 1.1" and choose ``cvat_yolo11_annotations.zip``.
   For Roboflow, upload ``yolo_dataset.zip`` (images + YOLO txt + data.yaml) instead.
3. **Correct.** The rules are provisional; write the one-page labeling guide (PLAN.md M0)
   before scaling up.

   - Box every person on the game court: players as ``player``, officials as ``referee``.
     Include partly occluded and far-side people.
   - Do not box spectators, bench players, coaches, or people on an adjacent court.
   - Box the visible extent of the body, including feet.
   - Remove false positives. Fix classes, especially referees labeled as players.
   - Add every missed person. Small far-side players are what the size buckets measure.
4. **Export.** In CVAT, choose Actions -> Export task dataset -> "YOLO 1.1"; images are not
   needed.
5. **Import.** Convert the export back to our conventions::

       python -m tracking.labeling import-cvat --format yolo --zip export.zip --group g001-b \\
           --images ../datasets/own/labeling/g001-batch1/g001-b/images \\
           --out ../datasets/own/labeling/g001-batch1/g001-b/corrected

   Class ids are remapped by name. An unknown label name is an error.
6. **Activate.** In ``evalsets/<name>.yaml``, point ``labels:`` at ``.../corrected/labels``
   and set ``labels_status: corrected``.
7. **Evaluate**::

       python -m tracking.evaluate detect --evalset g001-b \\
           --weights tracking/weights/players-v1.pt --out tracking/runs/eval-g001-b/v1.json

   For a candidate, add ``--baseline`` with the incumbent's report. ``baseline.deltas``
   gives per-metric changes, e.g. ``player.op.recall.small``.

Tracking ground truth uses the same loop, with CVAT's video tracks:

a. Seed from tracker output::

       python -m tracking.labeling export-tracks --group g001-a \\
           --tracks tracking/runs/basketball-court-v2/tracks.json \\
           --video tracking/runs/inputs/basketball-first30s.mp4 \\
           --out ../datasets/own/labeling/g001-track-seed

b. Create a CVAT task from **the same video file**, with label ``player``, frame step 1 and
   start frame 0. Then open Actions -> Upload annotations -> "MOT 1.1" and choose
   ``cvat_mot11.zip``.
c. Correct the seed:

   - Fix boxes and fill gaps.
   - **Split** a track where its identity jumps to another player.
   - **Merge** tracks that are the same player.
   - Delete non-players.

   One GT id per player for the whole clip.
d. Export with "MOT 1.1", then convert::

       python -m tracking.labeling import-cvat --format mot --zip export.zip \\
           --out ../datasets/own/labeling/g001-track-gt/gt/gt.txt

e. Evaluate any run of the same clip::

       python -m tracking.evaluate track \\
           --gt ../datasets/own/labeling/g001-track-gt/gt/gt.txt \\
           --tracks RUN/tracks.json --out tracking/runs/eval-track/RUN.json

   Evaluating against an uncorrected seed is flagged diagnostic automatically: ``seed.json``
   sits next to it with the same hash.

Metric definitions
------------------
**Detection.**

- The model runs at ``--min-conf`` 0.001 with at most 100 detections per image and class.
  YOLO26's NMS-free head is used, as in ``detect.py``; ``--nms`` selects the other head.
- ``ap50`` and ``ap50_95`` are per-class COCO AP (101-point interpolation) from
  ``pycocotools.COCOeval``, over all box sizes.
- The threshold sweep and size buckets use COCO's matching at IoU 0.5: predictions in
  descending confidence each take the highest-IoU unmatched GT box with IoU >= 0.5. Because
  a match depends only on higher-confidence predictions, P/R/F1 at any threshold are exact.
- **Operating point:** maximize F1 over the swept thresholds subject to recall >= the
  class's floor (player 0.90, matching the Phase 0.2 "~90% of players found" bar; referee
  has no floor). Ties go to the higher threshold. If no threshold reaches the floor, the
  point maximizes F1 unconstrained and reports ``recall_floor_met: false``.
- **Size buckets** use GT box height normalized to a 1080-px-tall frame (``h * 1080 / H``).
  Defaults: small < 100, medium 100-150, large >= 150 (``--size-edges``). On the g001
  camera, tracked players are about 80-210 px tall. Far-baseline players are 80-100 px,
  and near-side players reach 160-210 px.
  - At imgsz 1280 a 100 px player is 67 model pixels; at imgsz 640 it is 33. "small" is
    therefore where a lower imgsz fails first.
  - A matched prediction counts in its GT's bucket; an unmatched one counts in the bucket of
    its own height.

**Tracking.**

- HOTA (+ DetA/AssA/LocA, averaged over alpha 0.05..0.95), CLEAR (MOTA, MOTP, IDSW, Frag,
  MT/PT/ML) and Identity (IDF1/IDP/IDR) come from TrackEval's own metric classes, at IoU
  similarity with threshold 0.5. These are the standard definitions.
- Scored frames are the frames tracks.json processed that fall inside the GT range
  (``--gt-step`` for sparse GT). GT is assumed dense over its range: a missing frame means
  no players. With tracker stride 2, only the processed frames are scored.
- ``Frag`` follows TrackEval's CLEAR definition: each time a GT id becomes matched again after
  a timestep in which it was not matched, where "not matched" includes the GT id being absent.
  Gaps inside a GT trajectory therefore count. Corrected GT should keep one continuous id per
  visible player. A seed exported from tracker output inherits the tracker's own gaps, so the
  plumbing self-check shows Frag > 0 even at HOTA = 1.
- GT rows with conf 0 are ignored, and only MOT class 1 (player) is scored.
- Tracks that carry a ``role`` count only when ``role == "player"``, unless ``--all-roles``.
- ``idsw_per_player_minute = IDSW / (GT boxes * frame duration / 60)``. The frame duration
  is the median timestamp step of the scored frames, so it accounts for stride.

Report schema (JSON, ``report_version`` "1")
--------------------------------------------
Common keys:

- ``report_version``, ``kind`` (``detection`` | ``tracking``), ``created_utc``.
- ``diagnostic`` (bool), ``diagnostic_reasons`` (list[str]), ``warning`` (str | null).
- ``summary``: flat ``{metric_key: number | null}``, the keys baselines are compared on.
- ``baseline`` (when ``--baseline``):
  ``{path, comparable, reasons, deltas: {key: {baseline, current, delta}}}``.
  ``comparable`` is false when the eval data or the diagnostic status differ.

Detection keys:

- ``evalset``: ``name, manifest, manifest_sha256, groups, labels_status, images, labels,
  images_sha256, labels_sha256``.
- ``model``: ``weights, weights_sha256, names, imgsz, device, nms, min_conf, max_det,
  class_map, gt_class_merge``.
- ``config``: ``iou_threshold, thresholds, report_thresholds, size_buckets, recall_floors,
  operating_point_rule``.
- ``counts``: ``images, gt_boxes{cls}, predictions{cls}``.
- ``classes.<cls>``:
  - ``gt_boxes, predictions, ap50, ap50_95``.
  - ``sweep``: ``[{threshold, tp, fp, fn, precision, recall, f1}]``.
  - ``operating_point``: a sweep row + ``recall_floor, recall_floor_met, by_size``.
  - ``at_thresholds{"0.25": row + by_size}``.
  - ``by_size{small|medium|large: {range_px, gt, tp, fp, precision, recall}}``.
- Summary keys: ``<cls>.ap50``, ``<cls>.ap50_95``,
  ``<cls>.op.{threshold,precision,recall,f1}``, ``<cls>.op.{recall,precision}.<bucket>``,
  ``<cls>.at0.25.{precision,recall,f1}``, ``<cls>.at0.25.recall.<bucket>``.

Tracking keys:

- ``inputs``: ``gt, gt_sha256, tracks, tracks_sha256, tracks_schema_version, role_filter,
  observed_only, mot_class, gt_step, iou_threshold``.
- ``frames``: ``evaluated, first, last, tracker_stride, gt_range,
  gt_frames_without_tracker_output, gt_boxes_not_evaluated, frame_duration_s``.
- ``counts``: ``gt_ids, tracker_ids, gt_boxes, tracker_boxes,
  tracker_boxes_excluded_by_role, player_minutes``.
- ``metrics``: ``HOTA, DetA, AssA, LocA, DetRe, DetPr, AssRe, AssPr, HOTA(0), IDF1, IDP, IDR,
  IDTP, IDFN, IDFP, MOTA, MOTP, IDSW, Frag, MT, PT, ML, CLR_TP, CLR_FN, CLR_FP,
  idsw_per_player_minute``. ``summary`` repeats them.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import yaml

from tracking.labeling import (
    BACKEND_DIR,
    CLASS_NAMES,
    GROUP_ID_RE,
    IMAGE_SUFFIXES,
    LABEL_CLASSES,
    MOT_PLAYER_CLASS,
    QUARANTINE_FILE,
    Detector,
    check_group_id,
    keep_track,
    load_quarantine,
    load_tracks,
    parse_frame_stem,
    read_mot,
    read_yolo_labels,
    sha256_file,
)

EVALSETS_DIR = Path(__file__).resolve().parent / "evalsets"
REPORT_VERSION = "1"
LABEL_STATUSES = ("prelabel", "corrected")
BUCKET_NAMES = ("small", "medium", "large")
OPERATING_POINT_RULE = (
    "maximize F1 over swept thresholds subject to recall >= recall_floor (if the class has "
    "one); ties -> higher threshold; if no threshold meets the floor, maximize F1 "
    "unconstrained and set recall_floor_met=false"
)


@dataclass(frozen=True)
class DetectionEvalConfig:
    iou_threshold: float = 0.5
    thresholds: tuple[float, ...] = tuple(round(0.05 * i, 2) for i in range(1, 20))
    # 0.10 = pipeline detector floor, 0.25 = pre-label threshold, 0.50 = track threshold.
    report_thresholds: tuple[float, ...] = (0.10, 0.25, 0.50)
    # GT height edges in 1080p-equivalent pixels; see "Size buckets" in the module docstring.
    size_edges_px: tuple[float, float] = (100.0, 150.0)
    reference_height_px: int = 1080
    recall_floors: tuple[tuple[str, float], ...] = (("player", 0.90),)
    min_conf: float = 0.001
    max_det: int = 100

    def __post_init__(self):
        low, high = self.size_edges_px
        if not 0 < low < high:
            raise ValueError("size_edges_px must be two increasing positive numbers")
        if min((*self.thresholds, *self.report_thresholds)) < self.min_conf:
            raise ValueError("Thresholds below min_conf cannot be evaluated")


class EvalPolicyError(RuntimeError):
    """The requested evaluation would not measure held-out accuracy."""


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _clean(value):
    """JSON-safe copy: numpy scalars to Python, NaN/inf to None."""
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return _clean(value.tolist())
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    return value


def write_report(path: Path, report: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_clean(report), indent=2, allow_nan=False) + "\n", "utf-8")


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def _prf(tp: int, fp: int, fn: int) -> dict:
    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "precision": _ratio(tp, tp + fp),
        "recall": _ratio(tp, tp + fn),
        "f1": _ratio(2 * tp, 2 * tp + fp + fn),
    }


# --------------------------------------------------------------------------- eval sets


@dataclass(frozen=True)
class EvalSet:
    name: str
    manifest: Path
    manifest_sha256: str
    images: Path
    labels: Path
    groups: tuple[str, ...]
    labels_status: str
    classes: tuple[str, ...]
    strict_names: bool = True
    description: str = ""


def load_evalset(
    name: str, evalsets_dir: Path = EVALSETS_DIR, base_dir: Path = BACKEND_DIR
) -> EvalSet:
    if not GROUP_ID_RE.fullmatch(name or ""):
        raise ValueError(f"Eval set name {name!r} must match [a-z0-9-]+")
    manifest = Path(evalsets_dir) / f"{name}.yaml"
    if not manifest.is_file():
        raise FileNotFoundError(f"No eval set manifest {manifest}")
    data = yaml.safe_load(manifest.read_text(encoding="utf-8")) or {}
    if data.get("name") != name:
        raise ValueError(f"{manifest}: 'name' must be {name!r}")
    groups = data.get("groups")
    if not isinstance(groups, list) or not groups:
        raise ValueError(f"{manifest}: 'groups' must be a non-empty list")
    status = data.get("labels_status")
    if status not in LABEL_STATUSES:
        raise ValueError(f"{manifest}: labels_status must be one of {LABEL_STATUSES}")
    classes = tuple(data.get("classes") or LABEL_CLASSES)
    unknown = [c for c in classes if c not in CLASS_NAMES]
    if unknown:
        raise ValueError(f"{manifest}: unknown classes {unknown}")
    frame_names = data.get("frame_names", "strict")
    if frame_names not in ("strict", "loose"):
        raise ValueError(f"{manifest}: frame_names must be strict or loose")

    def resolve(key):
        if not data.get(key):
            raise ValueError(f"{manifest}: '{key}' is required")
        path = Path(data[key])
        return path if path.is_absolute() else (Path(base_dir) / path).resolve()

    return EvalSet(
        name=name,
        manifest=manifest.resolve(),
        manifest_sha256=sha256_file(manifest),
        images=resolve("images"),
        labels=resolve("labels"),
        groups=tuple(check_group_id(g) for g in groups),
        labels_status=status,
        classes=classes,
        strict_names=frame_names == "strict",
        description=str(data.get("description", "")).strip(),
    )


def evalset_policy_reasons(evalset: EvalSet, quarantine: frozenset[str]) -> list[str]:
    reasons = []
    outside = sorted(set(evalset.groups) - quarantine)
    if outside:
        reasons.append(
            f"groups not in quarantine.yaml: {outside} (they may be in a training set, or "
            "this is a public dataset: it does not measure held-out accuracy on our footage)"
        )
    if evalset.labels_status != "corrected":
        reasons.append(
            f"labels_status is {evalset.labels_status!r}: the labels are uncorrected model "
            "pre-labels, so scores measure agreement with a model, not accuracy"
        )
    return reasons


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    group: str
    width: int
    height: int
    gt: np.ndarray  # (N, 5) [class_id, x1, y1, x2, y2] pixels


def _digest(items: Sequence[tuple[str, Path]]) -> str:
    digest = hashlib.sha256()
    for name, path in items:
        digest.update(name.encode())
        digest.update(sha256_file(path).encode())
    return digest.hexdigest()


def load_evalset_images(evalset: EvalSet) -> tuple[list[ImageRecord], dict]:
    from PIL import Image

    images = sorted(p for p in evalset.images.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise ValueError(f"No images in {evalset.images}")
    label_files = {p.stem: p for p in evalset.labels.glob("*.txt")}
    missing = [p.name for p in images if p.stem not in label_files]
    orphans = sorted(set(label_files) - {p.stem for p in images})
    if missing or orphans:
        raise ValueError(
            f"Every image needs exactly one label file (empty = no objects). Missing: "
            f"{missing[:5]} ({len(missing)}); labels without images: {orphans[:5]}"
        )
    records = []
    for path in images:
        group, _ = parse_frame_stem(path.stem, strict=evalset.strict_names)
        if group not in evalset.groups:
            raise ValueError(f"{path.name} belongs to group {group!r}, not in the manifest")
        with Image.open(path) as image:
            width, height = image.size
        gt = read_yolo_labels(label_files[path.stem], width, height)
        if len(gt) and gt[:, 0].max() >= len(CLASS_NAMES):
            raise ValueError(f"{label_files[path.stem]}: class id outside CLASS_NAMES")
        records.append(ImageRecord(path, group, width, height, gt))
    hashes = {
        "images_sha256": _digest([(p.name, p) for p in images]),
        "labels_sha256": _digest([(p.stem, label_files[p.stem]) for p in images]),
    }
    return records, hashes


# --------------------------------------------------------------------------- detection metrics


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU matrix between (N, 4) and (M, 4) xyxy boxes."""
    a = np.asarray(a, dtype=float).reshape(-1, 4)
    b = np.asarray(b, dtype=float).reshape(-1, 4)
    width = np.clip(np.minimum(a[:, None, 2], b[None, :, 2]) -
                    np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    height = np.clip(np.minimum(a[:, None, 3], b[None, :, 3]) -
                     np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    inter = width * height
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def match_detections(
    gt_boxes: np.ndarray, pred_boxes: np.ndarray, pred_conf: np.ndarray, iou_threshold: float
) -> np.ndarray:
    """COCO matching: returns the matched GT index per prediction, -1 for false positives."""
    matches = np.full(len(pred_boxes), -1, dtype=np.int64)
    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        return matches
    ious = box_iou(pred_boxes, gt_boxes)
    taken = np.zeros(len(gt_boxes), dtype=bool)
    threshold = min(iou_threshold, 1 - 1e-10)
    for p in np.argsort(-np.asarray(pred_conf), kind="stable"):
        candidates = np.where(taken, -1.0, ious[p])
        best = int(np.argmax(candidates))
        if candidates[best] >= threshold:
            matches[p] = best
            taken[best] = True
    return matches


def size_bucket(heights_1080: np.ndarray, edges: tuple[float, float]) -> np.ndarray:
    return np.searchsorted(np.asarray(edges, dtype=float), heights_1080, side="right")


def coco_average_precision(
    records: Sequence[ImageRecord],
    predictions: Sequence[np.ndarray],
    class_ids: Sequence[int],
    max_det: int,
) -> dict[int, dict]:
    """Per-class AP50 and AP50-95 from pycocotools.COCOeval (101-point interpolation)."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    images, annotations, detections = [], [], []
    for image_id, (record, preds) in enumerate(zip(records, predictions), 1):
        images.append({"id": image_id, "width": record.width, "height": record.height})
        for cls, x1, y1, x2, y2 in record.gt:
            if int(cls) in class_ids:
                annotations.append({
                    "id": len(annotations) + 1, "image_id": image_id,
                    "category_id": int(cls) + 1, "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "area": (x2 - x1) * (y2 - y1), "iscrowd": 0,
                })
        for x1, y1, x2, y2, conf, cls in preds:
            if int(cls) in class_ids:
                detections.append({
                    "image_id": image_id, "category_id": int(cls) + 1,
                    "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    "score": float(conf),
                })
    gt_counts = {c: sum(a["category_id"] == c + 1 for a in annotations) for c in class_ids}
    empty = {c: {"ap50": 0.0 if gt_counts[c] else None,
                 "ap50_95": 0.0 if gt_counts[c] else None} for c in class_ids}
    if not detections or not annotations:
        return empty
    with contextlib.redirect_stdout(io.StringIO()):
        ground_truth = COCO()
        ground_truth.dataset = {
            "images": images,
            "annotations": annotations,
            "categories": [{"id": c + 1, "name": CLASS_NAMES[c]} for c in class_ids],
        }
        ground_truth.createIndex()
        evaluator = COCOeval(ground_truth, ground_truth.loadRes(detections), "bbox")
        evaluator.params.catIds = [c + 1 for c in class_ids]
        evaluator.params.maxDets = [max_det]
        evaluator.params.areaRng = [[0.0, 1e10]]
        evaluator.params.areaRngLbl = ["all"]
        evaluator.evaluate()
        evaluator.accumulate()
    precision = evaluator.eval["precision"]  # [IoU, recall, class, area, maxDets]
    result = {}
    for k, class_id in enumerate(class_ids):
        p = precision[:, :, k, 0, -1]
        p50 = p[0]
        result[class_id] = {
            "ap50": float(p50[p50 > -1].mean()) if (p50 > -1).any() else None,
            "ap50_95": float(p[p > -1].mean()) if (p > -1).any() else None,
        }
    return result


def choose_operating_point(rows: Sequence[dict], recall_floor: float | None) -> dict:
    def key(row):
        return (row["f1"] or 0.0, row["threshold"])

    eligible = [r for r in rows if recall_floor is None or (r["recall"] or 0.0) >= recall_floor]
    best = max(eligible or rows, key=key)
    return {
        **best,
        "recall_floor": recall_floor,
        "recall_floor_met": None if recall_floor is None else bool(eligible),
    }


def compute_detection_metrics(
    records: Sequence[ImageRecord],
    predictions: Sequence[np.ndarray],
    classes: Sequence[str],
    config: DetectionEvalConfig | None = None,
) -> dict:
    """Per-class AP, threshold sweep, operating point and size buckets (no model needed)."""
    config = config or DetectionEvalConfig()
    if len(records) != len(predictions):
        raise ValueError("One prediction array per image is required")
    class_ids = [CLASS_NAMES.index(c) for c in classes]
    trimmed = []
    for preds in predictions:
        preds = np.asarray(preds, dtype=float).reshape(-1, 6)
        preds = preds[preds[:, 4] >= config.min_conf]
        kept = [preds[preds[:, 5] == c] for c in class_ids]
        kept = [p[np.argsort(-p[:, 4], kind="stable")][: config.max_det] for p in kept]
        trimmed.append(np.concatenate(kept) if kept else np.empty((0, 6)))
    ap = coco_average_precision(records, trimmed, class_ids, config.max_det)
    edges = config.size_edges_px
    bucket_ranges = {
        "small": [0.0, edges[0]], "medium": [edges[0], edges[1]], "large": [edges[1], None],
    }
    floors = dict(config.recall_floors)
    results = {}
    for name, class_id in zip(classes, class_ids):
        conf, is_tp, pred_bucket, gt_bucket, gt_conf = [], [], [], [], []
        for record, preds in zip(records, trimmed):
            gt = record.gt[record.gt[:, 0] == class_id, 1:]
            preds = preds[preds[:, 5] == class_id]
            scale = config.reference_height_px / record.height
            matches = match_detections(gt, preds[:, :4], preds[:, 4], config.iou_threshold)
            g_bucket = size_bucket((gt[:, 3] - gt[:, 1]) * scale, edges)
            p_bucket = size_bucket((preds[:, 3] - preds[:, 1]) * scale, edges)
            tp = matches >= 0
            p_bucket[tp] = g_bucket[matches[tp]]
            g_conf = np.full(len(gt), -np.inf)
            g_conf[matches[tp]] = preds[tp, 4]
            conf.append(preds[:, 4])
            is_tp.append(tp)
            pred_bucket.append(p_bucket)
            gt_bucket.append(g_bucket)
            gt_conf.append(g_conf)
        conf, is_tp = np.concatenate(conf), np.concatenate(is_tp)
        pred_bucket, gt_bucket = np.concatenate(pred_bucket), np.concatenate(gt_bucket)
        gt_conf = np.concatenate(gt_conf)
        n_gt = len(gt_conf)

        def row(threshold, conf=conf, is_tp=is_tp, n_gt=n_gt):
            above = conf >= threshold - 1e-12
            tp = int((is_tp & above).sum())
            return {"threshold": float(threshold), **_prf(tp, int((~is_tp & above).sum()),
                                                          n_gt - tp)}

        def by_size(threshold, conf=conf, is_tp=is_tp, pred_bucket=pred_bucket,
                    gt_bucket=gt_bucket, gt_conf=gt_conf):
            above = conf >= threshold - 1e-12
            out = {}
            for index, bucket in enumerate(BUCKET_NAMES):
                gt_count = int((gt_bucket == index).sum())
                tp = int(((gt_conf >= threshold - 1e-12) & (gt_bucket == index)).sum())
                fp = int((~is_tp & above & (pred_bucket == index)).sum())
                out[bucket] = {
                    "range_px": bucket_ranges[bucket], "gt": gt_count, "tp": tp, "fp": fp,
                    "precision": _ratio(tp, tp + fp), "recall": _ratio(tp, gt_count),
                }
            return out

        sweep = [row(t) for t in config.thresholds]
        operating = choose_operating_point(sweep, floors.get(name))
        operating["by_size"] = by_size(operating["threshold"])
        results[name] = {
            "gt_boxes": n_gt,
            "predictions": len(conf),
            **ap[class_id],
            "sweep": sweep,
            "operating_point": operating,
            "at_thresholds": {
                f"{t:.2f}": {**row(t), "by_size": by_size(t)} for t in config.report_thresholds
            },
        }
    return results


def detection_summary(classes: dict) -> dict:
    summary = {}
    for name, result in classes.items():
        summary[f"{name}.ap50"] = result["ap50"]
        summary[f"{name}.ap50_95"] = result["ap50_95"]
        op = result["operating_point"]
        for key in ("threshold", "precision", "recall", "f1"):
            summary[f"{name}.op.{key}"] = op[key]
        for bucket, values in op["by_size"].items():
            summary[f"{name}.op.recall.{bucket}"] = values["recall"]
            summary[f"{name}.op.precision.{bucket}"] = values["precision"]
        for threshold, values in result["at_thresholds"].items():
            for key in ("precision", "recall", "f1"):
                summary[f"{name}.at{threshold}.{key}"] = values[key]
            for bucket, bucket_values in values["by_size"].items():
                summary[f"{name}.at{threshold}.recall.{bucket}"] = bucket_values["recall"]
    return summary


def run_detection_eval(
    evalset_name: str,
    weights: Path,
    out: Path | None = None,
    diagnostic: bool = False,
    baseline: Path | None = None,
    config: DetectionEvalConfig | None = None,
    imgsz: int = 1280,
    device: str = "cpu",
    nms: bool = False,
    person_as_player: bool = False,
    predictor: Callable[[list[Path]], list[np.ndarray]] | None = None,
    evalsets_dir: Path = EVALSETS_DIR,
    base_dir: Path = BACKEND_DIR,
    quarantine_file: Path = QUARANTINE_FILE,
) -> dict:
    """Score weights on a named eval set. Refuses diagnostic-only sets unless ``diagnostic``."""
    config = config or DetectionEvalConfig()
    evalset = load_evalset(evalset_name, evalsets_dir, base_dir)
    reasons = evalset_policy_reasons(evalset, load_quarantine(quarantine_file))
    if reasons and not diagnostic:
        raise EvalPolicyError(
            "Refusing a non-diagnostic evaluation: " + "; ".join(reasons)
            + ". Pass --diagnostic to run anyway; the report will be marked diagnostic."
        )
    records, hashes = load_evalset_images(evalset)
    classes = list(evalset.classes)
    class_map, gt_merge = {}, {}
    if person_as_player:
        # Stock COCO weights: 'person' covers players and officials alike.
        class_map, gt_merge = {"person": "player"}, {"referee": "player"}
        classes = ["player"]
        records = [
            ImageRecord(r.path, r.group, r.width, r.height, _merge_classes(r.gt, gt_merge))
            for r in records
        ]
    weights = Path(weights).resolve()
    model_names = None
    if predictor is None:
        detector = Detector(weights, imgsz, config.min_conf, device, config.max_det, class_map,
                            nms)
        model_names, predictor = detector.names, detector
    predictions = predictor([record.path for record in records])
    class_results = compute_detection_metrics(records, predictions, classes, config)
    report = {
        "report_version": REPORT_VERSION,
        "kind": "detection",
        "created_utc": _utc_now(),
        "diagnostic": bool(reasons),
        "diagnostic_reasons": reasons,
        "warning": ("DIAGNOSTIC ONLY - not a valid held-out accuracy measurement: "
                    + "; ".join(reasons)) if reasons else None,
        "evalset": {
            "name": evalset.name,
            "description": evalset.description,
            "manifest": str(evalset.manifest),
            "manifest_sha256": evalset.manifest_sha256,
            "groups": list(evalset.groups),
            "labels_status": evalset.labels_status,
            "images": str(evalset.images),
            "labels": str(evalset.labels),
            **hashes,
        },
        "model": {
            "weights": str(weights),
            "weights_sha256": sha256_file(weights) if weights.is_file() else None,
            "names": model_names,
            "imgsz": imgsz,
            "device": device,
            "nms": nms,
            "min_conf": config.min_conf,
            "max_det": config.max_det,
            "class_map": class_map,
            "gt_class_merge": gt_merge,
        },
        "config": {
            "iou_threshold": config.iou_threshold,
            "thresholds": list(config.thresholds),
            "report_thresholds": list(config.report_thresholds),
            "size_buckets": {
                "basis": f"GT box height scaled to a {config.reference_height_px}-px-tall frame",
                "edges_px": list(config.size_edges_px),
                "names": list(BUCKET_NAMES),
            },
            "recall_floors": dict(config.recall_floors),
            "operating_point_rule": OPERATING_POINT_RULE,
            "ap_implementation": "pycocotools.COCOeval (101-point), all sizes",
        },
        "counts": {
            "images": len(records),
            "gt_boxes": {name: r["gt_boxes"] for name, r in class_results.items()},
            "predictions": {name: r["predictions"] for name, r in class_results.items()},
        },
        "classes": class_results,
        "summary": detection_summary(class_results),
    }
    return _finish(report, out, baseline)


def _merge_classes(gt: np.ndarray, merge: dict[str, str]) -> np.ndarray:
    gt = gt.copy()
    for source, target in merge.items():
        gt[gt[:, 0] == CLASS_NAMES.index(source), 0] = CLASS_NAMES.index(target)
    return gt


# --------------------------------------------------------------------------- tracking metrics


def load_gt_frames(gt: Path, mot_class: int = MOT_PLAYER_CLASS) -> tuple[dict, dict]:
    rows = read_mot(gt)
    if not len(rows):
        raise ValueError(f"{gt} has no rows")
    keep = (rows[:, 6] != 0) & (rows[:, 7] == mot_class)
    frames: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for frame in np.unique(rows[keep, 0]).astype(int):
        selected = rows[keep & (rows[:, 0] == frame)]
        boxes = np.column_stack((selected[:, 2], selected[:, 3], selected[:, 2] + selected[:, 4],
                                 selected[:, 3] + selected[:, 5]))
        frames[int(frame) - 1] = (selected[:, 1].astype(np.int64), boxes)
    info = {
        "rows": len(rows),
        "ignored_conf0": int((rows[:, 6] == 0).sum()),
        "other_classes": int(((rows[:, 6] != 0) & (rows[:, 7] != mot_class)).sum()),
        "range": [int(rows[:, 0].min()) - 1, int(rows[:, 0].max()) - 1],
    }
    return frames, info


def load_tracker_frames(
    artifact: dict, include_all_roles: bool = False, observed_only: bool = False
) -> tuple[dict, dict, int]:
    frames, timestamps, excluded = {}, {}, 0
    for frame in artifact["frames"]:
        kept = [t for t in frame.get("tracks", [])
                if keep_track(t, include_all_roles, observed_only)]
        excluded += len(frame.get("tracks", [])) - len(kept)
        ids = np.array([t["track_id"] for t in kept], dtype=np.int64)
        boxes = np.array([t["bbox_xyxy"] for t in kept], dtype=float).reshape(-1, 4)
        frames[frame["frame_index"]] = (ids, boxes)
        timestamps[frame["frame_index"]] = frame.get("timestamp_s")
    return frames, timestamps, excluded


def build_trackeval_data(gt_frames: dict, tracker_frames: dict, frames: Sequence[int]) -> dict:
    """Assemble TrackEval's per-sequence ``data`` dict (ids relabeled 0..n-1)."""
    empty = (np.empty(0, dtype=np.int64), np.empty((0, 4)))
    gt_seq = [gt_frames.get(f, empty) for f in frames]
    tr_seq = [tracker_frames.get(f, empty) for f in frames]
    gt_map = {i: k for k, i in enumerate(sorted({int(i) for ids, _ in gt_seq for i in ids}))}
    tr_map = {i: k for k, i in enumerate(sorted({int(i) for ids, _ in tr_seq for i in ids}))}
    for ids, _ in (*gt_seq, *tr_seq):
        if len(ids) != len(set(ids.tolist())):
            raise ValueError("Duplicate id within one frame")
    return {
        "num_timesteps": len(frames),
        "num_gt_ids": len(gt_map),
        "num_tracker_ids": len(tr_map),
        "num_gt_dets": int(sum(len(ids) for ids, _ in gt_seq)),
        "num_tracker_dets": int(sum(len(ids) for ids, _ in tr_seq)),
        "gt_ids": [np.array([gt_map[int(i)] for i in ids], dtype=np.int64) for ids, _ in gt_seq],
        "tracker_ids": [np.array([tr_map[int(i)] for i in ids], dtype=np.int64)
                        for ids, _ in tr_seq],
        "similarity_scores": [box_iou(g, t) for (_, g), (_, t) in zip(gt_seq, tr_seq)],
    }


def tracking_metrics(data: dict, iou_threshold: float = 0.5) -> dict:
    """HOTA / CLEAR / Identity via TrackEval's reference metric classes."""
    from trackeval.metrics import CLEAR, HOTA, Identity

    config = {"THRESHOLD": iou_threshold, "PRINT_CONFIG": False}
    with contextlib.redirect_stdout(io.StringIO()):
        hota = HOTA().eval_sequence(data)
        clear = CLEAR(dict(config)).eval_sequence(data)
        identity = Identity(dict(config)).eval_sequence(data)
    metrics = {key: float(np.mean(hota[key])) for key in
               ("HOTA", "DetA", "AssA", "LocA", "DetRe", "DetPr", "AssRe", "AssPr")}
    metrics["HOTA(0)"] = float(hota["HOTA(0)"])
    metrics.update({key: float(identity[key]) for key in ("IDF1", "IDP", "IDR")})
    metrics.update({key: int(identity[key]) for key in ("IDTP", "IDFN", "IDFP")})
    metrics.update({key: float(clear[key]) for key in ("MOTA", "MOTP")})
    metrics.update({key: int(clear[key]) for key in
                    ("IDSW", "Frag", "MT", "PT", "ML", "CLR_TP", "CLR_FN", "CLR_FP")})
    return metrics


def _seed_reason(gt: Path) -> str | None:
    digest = sha256_file(gt)
    for directory in (gt.parent, gt.parent.parent):
        seed = directory / "seed.json"
        try:
            if json.loads(seed.read_text(encoding="utf-8")).get("gt_sha256") == digest:
                return f"ground truth is an uncorrected tracker seed ({seed})"
        except (OSError, ValueError):
            continue
    return None


def run_tracking_eval(
    gt: Path,
    tracks: Path,
    out: Path | None = None,
    baseline: Path | None = None,
    include_all_roles: bool = False,
    observed_only: bool = False,
    gt_step: int = 1,
    frame_range: tuple[int, int] | None = None,
    iou_threshold: float = 0.5,
) -> dict:
    gt, tracks = Path(gt).resolve(), Path(tracks).resolve()
    gt_frames, gt_info = load_gt_frames(gt)
    artifact = load_tracks(tracks)
    tracker_frames, timestamps, excluded = load_tracker_frames(
        artifact, include_all_roles, observed_only
    )
    first, last = frame_range or gt_info["range"]
    processed = sorted(tracker_frames)
    annotated = set(range(first, last + 1, gt_step))
    frames = [f for f in processed if f in annotated]
    if not frames:
        raise ValueError("tracks.json and the ground truth share no frames")
    frame_set = set(frames)
    data = build_trackeval_data(gt_frames, tracker_frames, frames)
    metrics = tracking_metrics(data, iou_threshold)
    steps = np.diff([timestamps[f] for f in frames if timestamps[f] is not None])
    fps = (artifact.get("video") or {}).get("fps")
    frame_duration = float(np.median(steps)) if len(steps) else (1 / fps if fps else None)
    player_minutes = data["num_gt_dets"] * frame_duration / 60 if frame_duration else None
    metrics["idsw_per_player_minute"] = _ratio(metrics["IDSW"], player_minutes)
    strides = np.diff(processed)
    reasons = [reason for reason in (_seed_reason(gt),) if reason]
    report = {
        "report_version": REPORT_VERSION,
        "kind": "tracking",
        "created_utc": _utc_now(),
        "diagnostic": bool(reasons),
        "diagnostic_reasons": reasons,
        "warning": ("DIAGNOSTIC ONLY - not a valid tracking accuracy measurement: "
                    + "; ".join(reasons)) if reasons else None,
        "inputs": {
            "gt": str(gt),
            "gt_sha256": sha256_file(gt),
            "gt_rows": gt_info,
            "tracks": str(tracks),
            "tracks_sha256": sha256_file(tracks),
            "tracks_schema_version": artifact.get("schema_version"),
            "role_filter": "all" if include_all_roles else "player-or-unset",
            "observed_only": observed_only,
            "mot_class": MOT_PLAYER_CLASS,
            "gt_step": gt_step,
            "iou_threshold": iou_threshold,
        },
        "frames": {
            "evaluated": len(frames),
            "first": frames[0],
            "last": frames[-1],
            "tracker_stride": int(np.bincount(strides).argmax()) if len(strides) else None,
            "gt_range": [first, last],
            "gt_frames_without_tracker_output": len(annotated - set(processed)),
            "gt_boxes_not_evaluated": int(sum(len(ids) for f, (ids, _) in gt_frames.items()
                                              if f not in frame_set)),
            "frame_duration_s": frame_duration,
        },
        "counts": {
            "gt_ids": data["num_gt_ids"],
            "tracker_ids": data["num_tracker_ids"],
            "gt_boxes": data["num_gt_dets"],
            "tracker_boxes": data["num_tracker_dets"],
            "tracker_boxes_excluded_by_role": excluded,
            "player_minutes": player_minutes,
        },
        "metrics": metrics,
        "summary": dict(metrics),
    }
    return _finish(report, out, baseline)


# --------------------------------------------------------------------------- baselines


def compare_reports(current: dict, baseline: dict, baseline_path: str | None = None) -> dict:
    reasons = []
    if current.get("kind") != baseline.get("kind"):
        reasons.append(f"report kinds differ: {baseline.get('kind')} vs {current.get('kind')}")
    if bool(current.get("diagnostic")) != bool(baseline.get("diagnostic")):
        reasons.append("one report is diagnostic and the other is not")
    if current.get("kind") == "detection":
        for key in ("name", "images_sha256", "labels_sha256"):
            if (current.get("evalset") or {}).get(key) != (baseline.get("evalset") or {}).get(key):
                reasons.append(f"eval set {key} differs")
    elif current.get("kind") == "tracking":
        if (current.get("inputs") or {}).get("gt_sha256") != (
            baseline.get("inputs") or {}
        ).get("gt_sha256"):
            reasons.append("ground truth differs")
        if (current.get("frames") or {}).get("evaluated") != (
            baseline.get("frames") or {}
        ).get("evaluated"):
            reasons.append("number of evaluated frames differs (stride or range)")
    deltas = {}
    base_summary, summary = baseline.get("summary") or {}, current.get("summary") or {}
    for key, value in summary.items():
        old = base_summary.get(key)
        if isinstance(value, bool) or isinstance(old, bool):
            continue
        if isinstance(value, (int, float)) and isinstance(old, (int, float)):
            deltas[key] = {"baseline": old, "current": value, "delta": value - old}
        elif key in base_summary:
            deltas[key] = {"baseline": old, "current": value, "delta": None}
    return {"path": baseline_path, "comparable": not reasons, "reasons": reasons,
            "deltas": deltas}


def _finish(report: dict, out: Path | None, baseline: Path | None) -> dict:
    if baseline is not None:
        baseline = Path(baseline)
        previous = json.loads(baseline.read_text(encoding="utf-8"))
        report["baseline"] = compare_reports(report, previous, str(baseline.resolve()))
    report = _clean(report)
    if out is not None:
        write_report(out, report)
    return report


# --------------------------------------------------------------------------- CLI


def _format_headline(report: dict) -> str:
    lines = []
    if report.get("warning"):
        lines.append(report["warning"])
    summary = report["summary"]
    if report["kind"] == "detection":
        for name, result in report["classes"].items():
            op = result["operating_point"]
            sizes = ", ".join(
                f"{b} {v['recall']:.3f}" if v["recall"] is not None else f"{b} n/a"
                for b, v in op["by_size"].items()
            )
            lines.append(
                f"{name}: AP50 {_fmt(result['ap50'])} AP50-95 {_fmt(result['ap50_95'])} | "
                f"op@{op['threshold']:.2f} P {_fmt(op['precision'])} R {_fmt(op['recall'])} "
                f"F1 {_fmt(op['f1'])} (floor met: {op['recall_floor_met']}) | recall {sizes}"
            )
    else:
        keys = ("HOTA", "DetA", "AssA", "IDF1", "MOTA", "IDSW", "Frag",
                "idsw_per_player_minute")
        lines.append(" ".join(f"{key} {_fmt(summary.get(key))}" for key in keys))
    if report.get("baseline"):
        comparable = report["baseline"]["comparable"]
        lines.append(f"baseline comparable: {comparable} {report['baseline']['reasons'] or ''}")
    return "\n".join(lines)


def _fmt(value) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}" if isinstance(value, float) else str(value)


def _parse_pair(text: str) -> tuple[str, float]:
    name, _, value = text.partition("=")
    if name not in CLASS_NAMES or not value:
        raise argparse.ArgumentTypeError("expected CLASS=VALUE, e.g. player=0.9")
    return name, float(value)


def main(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(description="Detection and tracking evaluation")
    commands = parser.add_subparsers(dest="command", required=True)

    detect = commands.add_parser("detect", help="Detector eval on a named eval set")
    detect.add_argument("--evalset", required=True, help="Manifest name in tracking/evalsets/")
    detect.add_argument("--weights", type=Path, required=True)
    detect.add_argument("--out", type=Path, required=True)
    detect.add_argument("--baseline", type=Path)
    detect.add_argument("--diagnostic", action="store_true",
                        help="Allow non-quarantined or uncorrected sets; marks the report")
    detect.add_argument("--imgsz", type=int, default=1280)
    detect.add_argument("--device", default="cpu")
    detect.add_argument("--nms", action="store_true", help="Use the one-to-many head + NMS")
    detect.add_argument("--person-as-player", action="store_true",
                        help="Stock COCO weights: model 'person' -> player; GT referee -> player")
    detect.add_argument("--size-edges", default="100,150", help="1080p-equivalent px edges")
    detect.add_argument("--min-conf", type=float, default=0.001)
    detect.add_argument("--max-det", type=int, default=100)
    detect.add_argument("--recall-floor", type=_parse_pair, action="append",
                        help="CLASS=VALUE (default player=0.9)")

    track = commands.add_parser("track", help="Tracker eval against MOT gt.txt")
    track.add_argument("--gt", type=Path, required=True)
    track.add_argument("--tracks", type=Path, required=True)
    track.add_argument("--out", type=Path, required=True)
    track.add_argument("--baseline", type=Path)
    track.add_argument("--all-roles", action="store_true", help="Score non-player roles too")
    track.add_argument("--observed-only", action="store_true")
    track.add_argument("--gt-step", type=int, default=1, help="GT annotated every N frames")
    track.add_argument("--frames", help="FIRST:LAST 0-based frame indices (default: GT range)")
    track.add_argument("--iou", type=float, default=0.5)

    args = parser.parse_args(argv)
    if args.command == "detect":
        low, high = (float(value) for value in args.size_edges.split(","))
        config = DetectionEvalConfig(
            size_edges_px=(low, high),
            min_conf=args.min_conf,
            max_det=args.max_det,
            recall_floors=tuple(args.recall_floor) if args.recall_floor
            else DetectionEvalConfig().recall_floors,
        )
        report = run_detection_eval(
            args.evalset, args.weights, args.out, args.diagnostic, args.baseline, config,
            args.imgsz, args.device, args.nms, args.person_as_player,
        )
    else:
        frame_range = None
        if args.frames:
            first, last = (int(value) for value in args.frames.split(":"))
            frame_range = (first, last)
        report = run_tracking_eval(
            args.gt, args.tracks, args.out, args.baseline, args.all_roles, args.observed_only,
            args.gt_step, frame_range, args.iou,
        )
    print(_format_headline(report))
    print(f"Report written to {args.out}")


if __name__ == "__main__":
    main()
