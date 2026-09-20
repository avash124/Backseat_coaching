# Player detection and tracking

This package now runs a standalone video pipeline with your trained YOLO26 weights
and BoxMOT OC-SORT. It produces cached detections, `tracks.json`, and a silent
H.264 preview. The upload API and Celery worker still use their existing stub;
they are not connected to this pipeline yet.

## 1. Understand the flow

```text
video -> decode.py -> detect.py (YOLO26) -> detections.jsonl
video + detections.jsonl -> track.py (OC-SORT) -> tracks.json + annotated.mp4
```

YOLO finds boxes independently in each frame. OC-SORT associates those boxes over
time and assigns IDs. A track ID is a temporary trajectory identifier, not a
jersey number or a player's identity. IDs can change during occlusion, crossings,
or camera cuts.

| File | Responsibility |
|---|---|
| `config.py` | Weights path, thresholds, and stage versions |
| `decode.py` | Stream frames and read their source-video timestamps |
| `detect.py` | Load YOLO once; select the class named `player` |
| `track.py` | Create one BoxMOT OC-SORT instance per video and update every frame |
| `pipeline.py` | Run detection and tracking; reuse completed stages |
| `schemas.py` | Validate the artifact format |
| `artifacts.py` | Stream JSON; write files atomically; verify cache hashes |
| `viz.py` | Draw boxes and IDs; encode a silent H.264 preview |
| `remote/app.py` | Modal inference image and volumes |
| `remote/infer.py` | Upload inputs and invoke the pipeline on an L4 GPU |
| `train_modal.py` | Existing standalone Modal training entry point |

Pure CV modules do not import FastAPI, Celery, or Modal. Modal wraps the same
`run_pipeline()` function that the local command uses.

## 2. Prepare the local environment

Run commands from `backend` in PowerShell:

```powershell
uv sync --group cv --group remote --system-certs
```

Use `backend/.venv/Scripts/python.exe` as the VS Code interpreter. The explicit
executables below avoid accidentally using the global Python installation.

The default checkpoint is:

```text
backend/tracking/weights/players-v1.pt
```

The supplied checkpoint contains `0: player` and `1: referee`. Inference discovers
the player class by name and filters out referees. Weights and generated runs are
gitignored.

## 3. Run 30 seconds on Modal

Replace the video path with a real local file:

```powershell
.\.venv\Scripts\modal.exe run --env main --detach -m tracking.remote.infer --video "C:\path\to\game.mp4" --run-name players-demo --max-seconds 30
```

This uploads the video and checkpoint, creates these volumes if missing, and runs
inference on an L4:

- `nextup-videos`: input videos, named by their content hash.
- `nextup-weights`: model checkpoints, named by their content hash.
- `nextup-inference`: generated artifacts under `/players-demo`.

The first run also builds the image. This uses paid Modal GPU time. Training does
not run again, and this does not modify the training script's H100 selection.
The full input file is uploaded even when processing only the first 30 seconds;
use a pre-trimmed clip to reduce upload time.

After the job finishes, download its outputs:

```powershell
.\.venv\Scripts\modal.exe volume get nextup-inference /players-demo tracking/runs/players-demo --env main
```

Open `tracking/runs/players-demo/annotated.mp4`. If downloading a later rerun to
the same local directory, add `--force` to replace that directory's output files.

Use a unique `--run-name` for simultaneous jobs. To process the whole video, set
`--max-seconds 0`; for a new video, choose a new run name. GPU jobs have a
four-hour timeout.

The image uses Torch 2.8.0, Torchvision 0.23.0, and Ultralytics 8.4.155 to match
training, plus BoxMOT 25.0.0. The CUDA wheel source is explicit.

## 4. Optional local run

The current local Torch installation is CPU-only. Test a few frames locally:

```powershell
.\.venv\Scripts\python.exe -m tracking.pipeline --video "C:\path\to\game.mp4" --output tracking/runs/local-demo --device cpu --max-frames 10
```

Both entry points accept `--weights`, `--imgsz`, `--confidence`, `--stride`,
`--max-age`, `--min-hits`, and `--iou-threshold`. The local command additionally
accepts `--max-frames`. For the local command, omit `--max-seconds` to process
the whole video.

## 5. Inspect the output

| Output | Meaning |
|---|---|
| `annotated.mp4` | Visual review of player boxes and temporary IDs; no audio |
| `tracks.json` | Frame indices, timestamps, boxes, IDs, confidence, and provenance |
| `detections.jsonl` | Cached raw player detections, one record per processed frame |
| `video.json` | Video dimensions, nominal FPS, and counts |
| `run.json` | Configuration, hashes, timings, and cache reuse flags |
| `*.complete.json` | Internal cache markers and output hashes |

A frame in `tracks.json` looks like:

```json
{
  "frame_index": 12,
  "timestamp_s": 0.4,
  "tracks": [
    {
      "track_id": 0,
      "bbox_xyxy": [100.0, 80.0, 160.0, 240.0],
      "confidence": 0.91,
      "class_id": 0,
      "observed": true
    }
  ]
}
```

Boxes are `[left, top, right, bottom]` in original video pixels. Timestamps start
at the first decoded frame and use the video's presentation timestamps, not
inference wall-clock time. Empty frames remain in the output and still advance
OC-SORT. No boxes are invented to fill missing observations. Preview dimensions
are padded by at most one pixel when H.264 needs even dimensions.

Check an uninterrupted 30-60 second passage of your own footage:

1. Does YOLO consistently find the visible players and exclude officials/spectators?
2. Does each moving player keep the same ID through ordinary motion?
3. Where do crossings and occlusions cause missed boxes or ID switches?

The automated smoke test used four copies of one validation image with the real
checkpoint. It confirmed loading, inference, tracking, artifact validation, and
H.264 encoding. It is not an accuracy or real-motion tracking evaluation.

## 6. Tune one stage at a time

Start with `--imgsz 1280 --confidence 0.25 --stride 1`.

- Missing or false boxes: evaluate the detector first. Lower confidence can recover
  missed players but can also admit spectators; higher confidence has the reverse
  tradeoff. Representative labeled footage is needed to choose a threshold.
- Good boxes but unstable IDs: inspect association near crossings, then test tracker
  settings. `max-age` counts processed frames, so its duration depends on FPS and
  stride. Increasing it can bridge gaps but can also reconnect the wrong player.
- `min-hits` controls track confirmation after the initial startup period.
  `iou-threshold` affects box matching. There is no universally best setting.
- Use separate runs for separate continuous shots. Automatic cut detection,
  player re-identification, team/jersey identification, and ball tracking are not
  implemented in this slice.

Reuse the same output directory/run name to tune the tracker on cached detections:

```powershell
.\.venv\Scripts\modal.exe run --env main --detach -m tracking.remote.infer --video "C:\path\to\game.mp4" --run-name players-demo --max-seconds 30 --max-age 45
```

Detector inputs/configuration changes invalidate detection. Tracker changes
invalidate tracking only. Completed stages are reused only when their hashes
match. Bump the relevant stage version in `config.py` when changing its algorithm.
Do not run concurrent jobs against the same output directory.

## 7. Connect the application after reviewing the clip

The next integration replaces the fake track artifact in `app/worker/tasks.py`:

1. Retrieve the uploaded source video using its storage key.
2. Invoke the Modal inference job and record its call ID on the existing job.
3. Publish `tracks.json` and the preview to object storage using versioned
   `app.storage.artifact_key(...)` paths.
4. Mark the job successful only after the artifacts are uploaded; record failures.
5. Return an authorized artifact URL from the results API.
6. In the frontend, render a canvas over the video using `video.currentTime`.
   Select the corresponding timestamp and scale pixel boxes to the displayed
   video rectangle, including any letterboxing.

Do not save every frame's boxes as database rows. Keep the dense data in the JSON
artifact and store only job/artifact references and track-level summaries in the
database. Player tracking alone does not generate play labels or possession.

## Validation

```powershell
.\.venv\Scripts\pytest.exe tracking/tests -q
```

Tests cover geometry and sequence validation, real OC-SORT gap/expiry behavior,
independent tracker instances, variable source timestamps, H.264 decoding,
sampling, cache reuse, corruption recovery, and input overwrite protection.

## Training and future stages

`PLAN.md` and `datasets/prepare_ebard.py` describe the current public-data training
workflow. Keep game-level splits and record dataset versions and licenses. Public
validation data is useful during development; evaluate on held-out footage from
the actual deployment setting before judging readiness.

The separate ball detection/tracking, possession, evaluation, and general training
modules remain scaffolds. The existing `train_modal.py` remains the working trainer.
