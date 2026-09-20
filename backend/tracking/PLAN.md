# Implementation plan — YOLO26 detector training

Companion to [`README.md`](./README.md) (what each module *is*) and to `ROADMAP.md` (why the
phases are ordered as they are). This doc covers **one thing**: getting from the current
docstring-only scaffold to a trained, evaluated, promoted player detector — roadmap **2.8**,
with the ball detector (**2.3b**) as the follow-on.

Out of scope here: the inference DAG (2.1), decode (2.2), OC-SORT integration (2.3), and the
worker→Modal seam. Those consume the weights this plan produces; none of them block it.

---

## 0. The decision that reorders the scaffold

The `datasets/` subsystem — pull → remap → merge → weight → split, five modules — is the
largest single build in this package. The README justifies it as *"public data is pretraining,
our footage is fine-tuning."* That reasoning is solid for the **ball**. It is **unproven for
players**: YOLO26's COCO weights already contain one of the most over-trained classes in
computer vision (`person`), and the things public basketball sets would add on top of that are
narrow — player-vs-referee separation, and our gym's domain quirks. Broadcast-angle basketball
footage may contribute close to nothing for the person class specifically.

That is not a reason to delete `datasets/`. It is a reason to **not build it first**. Sequence
so that a cheap experiment decides how much of it gets built:

| Run | Training data | Answers |
|---|---|---|
| **A** | none — stock YOLO26 COCO weights | 2.8's gate: is fine-tuning needed at all? |
| **B** | our own labeled frames only | is the thin path sufficient? |
| **C** | merged public → our frames | does public pretraining beat B enough to justify `datasets/`? |

Runs A and B need **zero** of `datasets/` beyond a trivial single-source loader. If B ≈ C, then
`datasets/` stays thin until the ball detector, where it genuinely earns its keep (public ball
annotations are the only realistic alternative to labeling thousands of boxes by hand).

Building the full merge/remap/weighting machinery before running A and B is exactly the failure
the roadmap's own guardrail warns about: *measure before you fine-tune; don't optimize a stage
that's already good enough.*

---

## 1. Milestones and gates

### M0 — Eval set before training set

Gate A cannot run without held-out labeled footage, so this comes first, and it is not code.

- **~250–350 frames**, sampled from **≥3 distinct games or halves**.
- **≥3–5 s apart.** Adjacent frames are near-duplicates; sampling densely inflates every metric
  that follows and teaches you nothing.
- **Stratified deliberately, not uniformly.** Halfcourt set, transition, inbound / dead ball,
  dense paint traffic, far-side players, and at least one panning segment. The tail cases are
  the thing being measured — a uniform sample is mostly easy frames.
- **Label every person**, including partially occluded and far-side. Classes: `player`,
  `referee`. Decide explicitly what happens to coaches and crowd — their own class, or excluded
  — and write it into a one-page labeling guide before labeling anything. Same discipline as
  roadmap 5.0, same reason: fuzzy class boundaries put a ceiling on quality that no model
  recovers, and re-labeling costs weeks.
- **Quarantine these games permanently.** They never enter any training set. Record the game ids
  somewhere durable (`splits.py` asserts against this list).

### M1 — Gate A: measure the stock model

Needs `config.py` and `evaluate.py` only. No training, no Modal GPU.

Score stock `yolo26{n,s,m}` across several `imgsz` values against M0. The output is a
**resolution + confidence-threshold recommendation**, not a single mAP number.

**Gate:** if stock detection at the chosen resolution clears the Phase 0.2 bar (~90% of on-court
players found), 2.8 says stop — skip to M5 and spend the time on the ball instead.

### M2 — Own-footage training set + gate B

- **800–1,500 frames** from games *other than* M0's, same sampling and labeling rules.
- Implement `training.py` and `remote/train.py` for the single-source case.
- Run B. Compare against A on M0.

### M3 — Gate C, only if B underperforms

Now build `sources.yaml` → `taxonomy.py` → `roboflow.py` → `merge.py` → `splits.py`, and run C.
Keep or shelve `datasets/` based on the C − B delta, and write the number down.

### M4 — Promote

Winner's weights → S3, referenced in `config.py` by version + sha256 + the dataset manifest hash
that produced them. Bump `DETECT_VERSION`. Store the eval comparison report beside the weights.
**Promotion is a human step**, gated on the M4 report — the training job never promotes itself.

### M5 — Ball detector (2.3b)

Same machinery, different config: single class, higher or tiled inference resolution, and here
`datasets/` is load-bearing. Expect this to be the hard one — the roadmap flags ball detection as
the largest schedule risk in the project.

---

## 2. File-by-file

### [`config.py`](./config.py) — build first, it is the contract

Frozen module-level constants, no I/O, importable from anywhere including the Modal container.

- **Stage versions** — `DETECT_VERSION`, `TRACK_VERSION`, `BALL_VERSION` as independent strings.
  The rule to enforce in review: *changing weights or a threshold ⇒ bump that stage's version.*
  Nothing else makes 2.1's resumable-DAG skip logic safe.
- **Weights references** — a record per model: logical name, S3 key, sha256, the `imgsz` it was
  trained at, and the dataset manifest hash behind it. Not a bare filename. The digest is what
  lets you prove which weights produced a given artifact six weeks later.
- **Thresholds** — per-class score threshold, inference `imgsz`, max detections.
  **YOLO26 is end-to-end / NMS-free**, so there is no NMS IoU to tune — but that also means its
  confidence distribution is *not* comparable to the YOLOv8/11 line. Every threshold here must
  come from the M1 sweep on our own footage, never from a tutorial. Say so in a comment; this is
  the highest-traffic footgun in the file.
- **Training defaults** — so `training.py` and `remote/train.py` cannot drift apart.

### [`evaluate.py`](./evaluate.py) — build second

Two distinct surfaces; do not conflate them.

1. **Detector eval** — per-class precision / recall / mAP50 / mAP50-95 on M0, **swept across
   confidence thresholds**, emitting the recommended operating point. Include a **size-stratified
   breakdown** (small / medium / large boxes): far-side players are precisely where a low `imgsz`
   fails, and aggregate mAP hides it.
2. **Tracker eval** — HOTA / IDF1 / MOTA / ID-switch counts on hand-verified clips. Needed later
   (Track B), but stub the interface now so `track.py` has a target to hit.

The deliverable is a **comparison against the incumbent weights**, written as JSON next to the
new weights version. *"+3.1 recall on far-side players, −0.4 on refs"* is what a promotion
decision is actually made from; a bare mAP is not.

**Enforcement detail:** make the eval entry point take a **game-id list**, not a dataset path.
Then there is no natural way to accidentally pass it a public val split — which the README
already forbids, and which would silently score a different camera domain.

### [`training.py`](./training.py) — the framework layer

One primary entry point, roughly `train_detector(cfg: TrainConfig) -> RunResult`:

- `TrainConfig` — a pydantic model: data.yaml path, model size, `imgsz`, epochs, LR, `freeze`,
  output dir, W&B run name, seed.
- `RunResult` — weights path, metrics, run id.

Properties that matter:

- **No Modal, no S3, no distributed setup.** Paths in, paths out. This is what makes a
  `pytest` smoke test on 8 images / 2 epochs / CPU able to catch config errors before you pay
  for a GPU. Make that test exist at M2 — it is the cheapest thing in this plan.
- **Two callable phases, not one flag** — `pretrain` (merged public) and `finetune` (our frames,
  lower LR, optional backbone freeze). `finetune` must accept a starting weights path, so run C
  is literally `finetune(weights=pretrain_output)`.
- **Resume support** — accept a checkpoint path and pass it through, so a preempted Modal GPU
  restarts instead of starting over.
- **Fixed seed**, and log the effective resolved config into the run directory.

### [`datasets/`](./datasets/) — deferred to M3, built bottom-up

Order matters; each file constrains the one before it.

1. **`sources.yaml` first, as research, not code.** Survey Roboflow Universe; record each
   candidate with its pinned version and license. License is a **filter applied before
   download**, not a note recorded after — unspecified is not permissive, and this is a
   commercial product. Record `domain: broadcast | tripod | mixed` too; that field is what
   justifies the sample weighting later.
2. **[`taxonomy.py`](./datasets/taxonomy.py)** — writable only once you know what the sources
   actually contain. Holds a canonical ordered class tuple (index = class id, **frozen forever**;
   reordering silently corrupts every label file already written), per-source remap tables
   mapping source class → canonical or explicit `None` drop, and a validation pass that
   **raises** on any source class missing from the remap. Silent pass-through is how you end up
   training on scrambled ids.
   - **Decision to make here:** recommend `player` + `referee` as separate classes in one
     detector. Refs-as-a-class hands Phase 4.3's non-player filtering a far more reliable signal
     than the planned court-polygon + appearance heuristics, for free. Confirm enough sources
     carry referee labels before committing to it.
3. **[`roboflow.py`](./datasets/roboflow.py)** — download by pinned `(workspace, project,
   version)` into a content-addressed directory named by a hash of those coordinates; exists ⇒
   skip. Pull in YOLO txt format, but **re-derive class ids through `taxonomy.py`** — never trust
   the downloaded `data.yaml` ordering.
4. **[`merge.py`](./datasets/merge.py)** — remap, assemble one YOLO dir + `data.yaml`, and emit
   `dataset_manifest.json`: source list with versions, per-class counts, total frames, and an
   overall dataset hash. That hash goes into the `config.py` weights record. Sample weighting is
   simplest and most honest as explicit oversampling of own-footage entries in the train list —
   record the factor in the manifest rather than burying it in code.
5. **[`splits.py`](./datasets/splits.py)** — group-wise split keyed by game, with a hard
   assertion that no group appears in two splits, and a check against M0's quarantine list. The
   test split is our footage only.

### [`remote/app.py`](./remote/app.py) — the Modal shell

- **One `App`, one `Image`**, shared by train and infer — the entire point of this file. Pin
  `ultralytics`, `torch`, and CUDA explicitly; a floating ultralytics version is how weights
  start scoring differently in production than they did in training. Verify which ultralytics
  release ships YOLO26 weights and pin at or above it.
- Mount `tracking/` as **local Python source** rather than baking it into the image, so a code
  edit does not trigger an image rebuild.
- Three Volumes (`datasets`, `checkpoints`, `weights`) and the Secret names
  (`roboflow-api-key`, `wandb-api-key`, S3 creds).
- GPU class as a constant here, not scattered across decorators.

### [`remote/train.py`](./remote/train.py) — the GPU wrapper

- **Split dataset prep off the GPU.** A CPU-only function pulls/merges into the `datasets`
  Volume; the GPU function assumes the dataset is already there. Downloading gigabytes at GPU
  rates is pure waste, and this is the one structural choice in the Modal layer that
  meaningfully moves cost.
- GPU function flow: mount Volumes + Secrets → verify the dataset hash matches the requested one
  → call `training.train_detector` with output on the `checkpoints` Volume → `volume.commit()`
  periodically so preemption is survivable → on success upload weights to S3 and write a weights
  manifest (digest, dataset hash, metrics, W&B run id) → return metrics.
- A `local_entrypoint` for `modal run --detach`, taking config as a file or flags.
- **It must not promote weights into `config.py`.** Promotion is M4's human step.

### Packaging — currently missing

None of `ultralytics`, `modal`, `wandb`, `roboflow`, `boxmot`, `opencv`, or `decord`/`PyAV` are
in `pyproject.toml`. Add them as an **optional dependency group** (e.g. `cv`), not to base
`dependencies` — the API and Celery containers must not pull torch. The Modal image installs the
`cv` group; Railway/Render installs base only.

---

## 3. Starting hyperparameters (to beat, not to trust)

- **Resolution over capacity.** On 1080p tripod film a far-side player is roughly 60–100 px tall;
  at `imgsz=640` that is 20–33 px. Start at **`imgsz=1280` with `yolo26s`** rather than `yolo26m`
  at 640. Sweep it in M1 — it is the single most impactful knob, and it feeds straight into
  2.7's $/game-hour, so measure accuracy and cost in the same pass.
- **Pretrain** (if M3 happens): 50–100 epochs, early stopping, mosaic on with `close_mosaic` for
  roughly the last 10.
- **Fine-tune:** 20–40 epochs at ~1/10 the LR, milder augmentation. At ~1,000 frames, run both
  frozen-backbone and full fine-tune — freezing often wins at that data volume, and it is one
  cheap extra run to find out.
- **Augmentation:** no vertical flip (basketball has a fixed up direction); horizontal flip is
  fine. HSV augmentation is safe here — team assignment runs a separate encoder on crops in
  Phase 4, so the detector never needs to preserve jersey color.
- **Hardware:** L4 or A10G is right for a model this size; A100 is a waste. Run detached from the
  first real run.

---

## 4. Risks specific to this stage

1. **Labeling is the real cost, not GPU time.** M0 + M2 is ~1,500 box-dense frames, it gates
   everything, and there is no shortcut. Bootstrap it: run stock YOLO26 to pre-annotate, then
   correct — roughly 3–5× faster than labeling cold, and it doubles as a qualitative read on
   gate A before you formally measure it.
2. **Eval-set leakage.** The M0 quarantine holds only if the game ids are written down and
   `splits.py` checks them. This is the mistake that stays invisible until real uploads
   disappoint.
3. **YOLO26 threshold semantics.** NMS-free means the confidence distribution differs from every
   v8/11 tutorial you will find. The first threshold must come from M1's own-footage sweep.
4. **`datasets/` scope creep.** Five modules of pull/remap/merge/weight machinery is a week-plus
   of work. Gates A and B decide whether it is needed now or at M5.

---

## Build order at a glance

```
M0  label eval set (3+ quarantined games, ~300 frames)
      ↓
M1  config.py + evaluate.py  →  gate A: stock YOLO26 vs. 0.2 bar  →  [clears? → M5]
      ↓
M2  label train set (~1k frames) + training.py + remote/{app,train}.py  →  gate B
      ↓
M3  datasets/* (sources → taxonomy → roboflow → merge → splits)  →  gate C  [only if B is weak]
      ↓
M4  promote winner: S3 + config.py record + DETECT_VERSION bump + eval report
      ↓
M5  ball detector (2.3b) — same machinery, single class, higher res
```
