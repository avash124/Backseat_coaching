# Build Roadmap — AI Sports Film Analysis Platform (Basketball v1)

A detailed, ordered, step-by-step process for a solo developer (strong ML, limited CV). Phases are in strict development order; steps within a phase are mostly sequential. Two tracks — **Data Collection** and **Evaluation** — run in parallel the whole time and are called out at the end.

**The governing principle:** prove the scary CV works in throwaway code → build the empty product end-to-end with stubs → drop in real pipeline stages one at a time in dependency order. Never build a later stage before the stage that feeds it exists.

The dependency chain you cannot reorder:
`decode → detect → track (players + ball) → register court → assign teams / filter non-players → (trajectories) → match up defenders → segment plays → classify plays → attribute possession → outcomes → possession diagnostics`

The product has **two pillars**, not one. Pillar 1 is *what happened*: identify the play being run (Phase 6, top-3 suggestions + coach confirm). Pillar 2 is *why it worked or didn't*: possession diagnostics — shot made / defensive stop, but also "the cutter never gained separation," "that was a contested shot," "the weak-side corner was wide open." Pillar 2 sits at the end of the longest dependency chain in the system and is the reason ball tracking is not optional.

*(Revision notes: court registration now precedes team assignment; ball tracking is a hard dependency rather than a deferrable extra; defender matchup (4.5) and the diagnostics engine (Phase 7.5) are new — see the revision log at the bottom.)*

---   

## PHASE 0 — Feasibility spike + project setup (≈ 1–2 weeks)

Goal: prove the three scariest CV stages work *on your actual footage* before you invest in any product scaffolding, and stand up your tooling.

**0.1 — Get and inspect your footage — including a real Hudl export.** Pull your 2.5 hrs of HS film, **and ask a coach for one actual Hudl game export now**, because Hudl exports (not your own recordings) are the input format most customers will bring — their resolution, fps, bitrate, and camera angle drive every later decision. Note resolution, fps, camera position, whether there's a usable scoreboard, jersey color contrast, how often players occlude each other, and how much the camera pans/zooms.

**0.2 — Throwaway feasibility notebook (this code gets deleted).** In a single Jupyter/Colab notebook, on ~30 seconds of your real footage:
- Run **YOLO26** out of the box on sampled frames. Does it find your players reliably? Note false positives (refs, crowd, coaches). It's end-to-end (NMS-free), so there's no NMS threshold to tune — but that also means the confidence distribution behaves differently from the YOLOv8/11 line, so re-check your score threshold rather than reusing a number from a tutorial.
- Run **OC-SORT** on top **at (near-)full framerate**. Do IDs stay stable for ~5+ seconds? Where do they break (screens, fast breaks)?
- Attempt **homography**: manually click 4–6 court landmarks on a clean frame, compute the transform, project player feet onto a top-down court. Then test it on a **panning segment**, not just a static frame — panning is the normal case for HS film and is where homography actually gets hard.
- **Spike ball detection too** (new — because Phase 7.5 makes it load-bearing). Run the base detector's `sports ball` class on ~200 frames spanning a dribble, a pass, and a shot. Record the recall honestly. Expect it to be poor: the ball is often <20 px, motion-blurred, and occluded by bodies for most of a possession. You are not trying to make it work here — you are finding out *how far from working* it is, because that number sets the schedule for the entire diagnostics pillar.

**Define explicit go/no-go thresholds before you run, not after** — e.g. detector finds ≥ ~90% of on-court players by eyeball, a track survives ≥ 5 s through at least one screen, projected foot positions land within ~2–3 ft of where they should on the court diagram. If all three pass with rough edges, proceed. If homography is a disaster on your camera angle, you've learned that *now*, cheaply — fix the angle requirement or the approach before building anything.

**One threshold is not enough for Pillar 2 — measure your homography error budget explicitly.** ±2–3 ft is fine for play *classification*, which reads the coarse shape of a possession. It is **not** fine for diagnostics, which make quantitative claims about distances: the difference between a contested and an open shot is roughly 4 ft, so ±3 ft of projection error swamps the effect you are trying to measure. On your reference frame, click ~10 points whose true court coordinates you know (line intersections, arc apex, block, elbow), project them, and record the **actual error distribution in feet** — not a vibe check. If you cannot get median error under ~1.5 ft on a panning segment, the descriptive diagnostics in 7.5.1 will be measuring their own noise, and you need to fix the camera position or the registration approach before Phase 7.5 is worth building. Write the number down; 7.5 refers back to it.

**0.3 — Repo + tooling.** Monorepo (e.g. `/web`, `/api`, `/cv`, `/infra`). Set up: `uv` or Poetry (Python), `pnpm` (Node), Ruff, ESLint + Prettier, pre-commit hooks, a basic `pytest` and `vitest` skeleton. Initialize Git, push to GitHub.

**0.4 — Accounts.** Create accounts you'll need soon: object storage (R2/S3), a managed Postgres (Neon/Supabase), Modal or RunPod (GPU), Vercel, Railway/Render, Sentry, Weights & Biases. Don't configure them deeply yet — just exist.

**0.5 — Legal/privacy note (write it down now, act on it before Phase 8).** HS film is footage of minors, and coaches will upload film of *other* schools' teams. Before real users touch the product: consent/ToS language, a data-retention policy, no public indexing of film, and a check of Hudl's export terms. None of this blocks building; all of it must exist before Phase 8.

**Checkpoint:** you can articulate, from evidence on your own film, that detection/tracking/homography are viable, and you can run lint + tests locally.

---

## PHASE 1 — Product skeleton with stubs, no ML (≈ 3–4 weeks)

Goal: a deployed, end-to-end vertical slice where a user uploads a video, a job "runs," and a results page appears — all with **fake** results. This forces every plumbing problem out into the open while it's cheap to fix.

**1.1 — Database schema v1.** Design core tables: `users`, `videos` (with status, storage key, metadata), `jobs` (status, pipeline_version, timestamps), `artifacts` (type, storage key, the `video_id + stage + stage_version` it belongs to). Add `plays`, `tracks`, `stats` as empty-ish tables you'll flesh out later. Also add **`observations`** now, even though nothing writes to it until Phase 7.5: `play_id`, `type` (contested_shot, lost_separation, open_teammate, …), `subject_track_id`, `object_track_id` (the defender/teammate it's about), `timestamp`, `severity`, `confidence`, `evidence` (JSON — the measured numbers behind the claim), `coach_verdict` (unreviewed/confirmed/rejected). Creating it early costs nothing and keeps diagnostics first-class data rather than something bolted onto `stats` later. Set up SQLAlchemy 2.0 + Alembic; run your first migration.

**1.2 — Storage layer with multipart uploads.** Wire S3/R2. Game film is 1–6 GB; a single presigned PUT will fail routinely on home upload bandwidth. Use **S3 multipart upload with presigned part URLs** (or tus/Uppy for resumability) so an interrupted upload resumes instead of restarting. Enforce size/duration caps and validate container/codec at confirm time. Define your artifact-path convention now: `videos/{id}/raw.mp4`, `videos/{id}/{stage}/{stage_version}/tracks.json`, etc.

**1.3 — API.** FastAPI with: create-upload (returns presigned part URLs), confirm-upload (creates a `video` + enqueues a `job`), get-job-status, get-results. Pydantic v2 schemas throughout.

**1.4 — Job queue with a STUB worker — and a planned decision point.** Redis + Celery works, but note: once the heavy work moves to Modal in Phase 2, the Celery worker becomes a thin shim that just calls a Modal function. At that point, **decide deliberately** whether to keep Celery as orchestration or cut Redis+Celery entirely and have FastAPI spawn Modal functions directly (Modal `.spawn()` + polling/webhook). For a solo dev, fewer moving parts usually wins — don't sleepwalk into maintaining both. For now, the stub worker sleeps, writes a hardcoded fake `tracks.json` and fake `plays` to storage/DB, and marks the job complete. This proves the async path end-to-end.

**1.5 — Frontend.** Next.js: an upload page (drag-drop → resumable presigned upload → confirm), a job-status page that polls, and a results page that renders the fake data. Tailwind + shadcn/ui. TanStack Query for all server calls.

**1.6 — Auth.** Drop in Clerk or Supabase Auth. Gate uploads and results behind login. Don't hand-roll.

**1.7 — Deploy now, while it's simple.** Frontend → Vercel. API + worker → Railway/Render. GitHub Actions to build/test/deploy on push. Wire Sentry. Getting CI/CD working on a trivial app saves you from debugging deploys later when everything is complex.

**Checkpoint:** logged-in user uploads a video on the deployed site, watches a job run, and sees fake results. No ML anywhere, but the entire skeleton is real and deployed.

---

## PHASE 2 — Detection + tracking (first real CV) (≈ 3–5 weeks)

Goal: replace the stub with a real GPU job that produces player tracks and an annotated playback view. First real "wow."

**2.1 — GPU job on Modal/RunPod.** Move the heavy work to a serverless GPU function (Python decorator on Modal). Your API/worker now enqueues/awaits this GPU job instead of doing the work itself. You pay per-second, only when processing. **Make the pipeline a resumable DAG from day one:** each stage is idempotent, keyed by `(video_id, stage, stage_version)`, and skips itself if its output artifact already exists. Version stages independently (not one global pipeline_version) so bumping the classifier later never forces re-running decode/detect/track — reprocessing GPU stages is your dominant cost.

**2.2 — Decode + sample.** decord/PyAV to pull frames. FFmpeg for any transcoding. Mind disk: 1 hr of 1080p ≈ tens of GB of frames; stream/process in chunks, don't materialize everything.

**2.3 — Detection + tracking at dense framerate, trajectories at sparse rate.** Run detection + **OC-SORT** at **15–30 fps** — tracker association degrades badly with large inter-frame displacement, and 5 fps will fragment IDs on every fast break. If compute-bound, shrink inference resolution or skip every other frame, but do not drop to 5 fps for *tracking*. You can downsample the resulting tracks to 5–10 Hz afterward for downstream stages — classification is happy with sparse trajectories; **association is not**. Don't chase perfect long-term re-ID: plays are short, and play-level analysis tolerates track fragments — jersey-number identity comes later (see 7.0).

**Why OC-SORT and not ByteTrack.** ByteTrack's association leans on a constant-velocity Kalman filter, and basketball violates that assumption constantly — jab steps, screen contact, backdoor cuts, closeouts. OC-SORT is built for exactly that failure: observation-centric re-update repairs the filter's accumulated error after an occlusion instead of letting a bad velocity estimate compound, and observation-centric momentum adds a direction term to the association cost so a player who reverses direction isn't handed to the nearest track still moving the old way. Two practical notes:
- **It's an add-on, not a config flag.** Ultralytics ships `bytetrack.yaml`/`botsort.yaml` in-box; OC-SORT is not one of them. Pull it via `boxmot` (or the reference OC-SORT repo) and feed it detections yourself — a small amount of glue in the detect→track stage, worth budgeting a day for.
- **It's motion-only — no appearance model.** That's the point (fast, no ReID network to run per frame), but it means identity after a long full occlusion is not recoverable from OC-SORT alone. If ID switches through screens turn out to be the dominant error in the 0.2 spike, the upgrade path is an appearance-augmented variant (Deep OC-SORT / BoT-SORT with ReID), not more Kalman tuning. Decide that from spike evidence, not in advance.

**2.3b — Ball tracking is its own stage, and it is a hard dependency (revised).** The original plan let you defer the ball "if it's fighting you." That is no longer available: contested-shot, open-teammate, and every possession-attributed diagnostic in Phase 7.5 requires knowing *who has the ball at time t*. Budget for it as real work, not a flag on the player detector:
- **Separate small-object pass.** A dedicated higher-resolution or tiled inference pass for the ball, not the same forward pass as players. Almost certainly needs fine-tuning — base COCO `sports ball` weights are trained on clean, large balls and degrade badly on a blurred 15 px orange smear. Plan on labeling ~2,000–5,000 ball frames (tedious but fast per frame: one tiny box).
- **Physics-aware tracking.** A Kalman filter with a ballistic motion model, plus gap interpolation — the ball is *invisible* for large stretches (in a player's hands, behind bodies). Frame-wise detection alone will not give you a usable trajectory; you are reconstructing a path, not reading one.
- **Possession attribution is inference, not detection.** "Who has the ball" is a separate model/heuristic on top: nearest-player-to-ball with temporal continuity, plus an explicit **no-possession state** for passes and shots in flight. Write it as its own artifact (`possession.json`: per timestamp, either a track-ID or `in_air`).
- **Cheap partial fallback if ball detection plateaus.** Ball-handler identity is partially recoverable from *player trajectories alone* — defender convergence, spacing geometry, and dribble-motion signature. It's weaker than real ball tracking, but it degrades gracefully and unblocks the openness diagnostics (7.5.3) while ball detection is still bad. The review tool (5.3) can also let a coach tag the handler on a sample of plays, which doubles as training data.

**Track this stage's status honestly against Phase 7.5's schedule** — this is the single most likely thing in the whole roadmap to blow its estimate.

**2.4 — Store as versioned artifacts.** Output `tracks.json`: per frame, per track-ID, the box + foot position. Write it keyed by `video_id + stage_version`. This is the contract every later stage reads from. Lock the schema now.

**2.5 — Annotated playback: overlay client-side, don't re-encode.** Instead of burning boxes into a re-encoded MP4 (slow, costs GPU time, and must be redone on every pipeline bump), render overlays **in the browser**: a canvas layered over the video element, drawing boxes/IDs from `tracks.json` synced to `video.currentTime`. Iterating on the visualization becomes free and instant. Keep short burned-in clips only as debug artifacts for yourself.

**2.6 — Frontend.** Results page plays the original video (Video.js/HLS.js) with the overlay canvas and scrubbing. Now a coach can *see* the system working.

**2.7 — Measure cost per game-hour and set a budget.** Decode+detect dominate compute. Instrument the Modal job, compute $/game-hour on your first real runs, and set a target (e.g. under $1–2/game). This number determines pricing viability and drives choices like inference resolution and model size for the rest of the project.

**2.8 — Fine-tune YOLO only if needed.** If out-of-box detection from 0.2 was weak: label ~500–1,500 frames in Roboflow/CVAT (refs and crowd as negatives matter), fine-tune, redeploy. Track the run in W&B. Don't fine-tune if the base model is already good enough — measure first.

**Checkpoint:** upload → real detections + stable tracks → scrubbable annotated playback on the deployed site, with a known $/game cost.

---

## PHASE 3 — Court registration + 2D top-down view (≈ 2–4 weeks)

Goal: map pixel coordinates to a canonical court. This is what converts "boxes on video" into **trajectories**, which everything downstream depends on. Second "wow."

*(Moved ahead of team assignment: it's the riskiest remaining stage after tracking — surface that risk early — and non-player filtering needs the court polygon anyway, which was a circular dependency in the old ordering.)*

**3.1 — Semi-automatic registration is the v1 plan of record, not the fallback.** Fully-automatic court keypoint detection on HS gym footage (non-standard floors, logos, glare, low contrast, panning cameras) is the hardest CV in this project — treat it as a later optimization. For v1, exploit the "one camera type" guardrail: HS film is almost always a fixed tripod that pans/zooms. So: **calibrate once per video** — click 4–6 court landmarks on a reference frame in a simple UI — then **propagate the homography frame-to-frame** via feature tracking/optical flow on court points, re-anchoring periodically and flagging drift for a manual re-click. This turns a 3–5 week research risk into a 1–2 week engineering task. Explore TVCalib/PnLCalib-style automatic detection later, as an enhancement that pre-fills the manual clicks (note: those models are soccer-oriented; expect adaptation work).

**3.2 — Homography per frame.** Compute the pixel→court transform per frame from the propagated landmarks. Smooth across frames; handle zooms by re-estimating scale.

**3.3 — Court-type-aware normalization.** Encode the real dimensions per level (HS 84 ft, NCAA/NBA 94 ft; differing 3pt line, paint, restricted area) so every level projects into one shared **normalized court space**. This is the step that makes NBA/college and HS trajectories comparable later.

**3.4 — Produce trajectories.** Project each player's foot position → `trajectories.json` (per track-ID, per timestamp, normalized x/y). Foot-point projection is noisy: apply Kalman or Savitzky–Golay smoothing to the projected positions before writing the artifact. This artifact is the input to segmentation and classification.

**3.5 — Frontend top-down view.** Render a 2D court (D3/Konva) with moving dots per player, synced to the video scrubber. Huge perceived value jump.

**Checkpoint:** a synced top-down court animation driven by real normalized trajectories.

---

## PHASE 4 — Team assignment, non-player filtering + defender matchup (≈ 2–3 weeks)

Goal: split tracked players into two teams, filter non-players, and pair each offensive player with the defender guarding them. Short but required before plays mean anything.

**4.1 — Torso-crop embeddings, not raw HSV.** For each track, crop the torso region across several frames. Instead of HSV histograms (brittle under gym lighting and white/light jerseys), embed the crops with a small pretrained vision encoder (e.g. SigLIP/CLIP image features — the approach used in Roboflow's sports pipeline) and average per track. Keep HSV as a cheap fallback/feature, not the primary signal.

**4.2 — Cluster into two teams.** k-means (k=2) over track-level embeddings. Assign each track a team label.

**4.3 — Filter non-players.** Remove referees, coaches, and crowd using the court polygon from Phase 3 (are they inside/outside the court, are they stationary on the sideline?) plus appearance (striped/grey refs). This is why registration now comes first.

**4.4 — Ambiguity handling.** For similar-colored matchups, train a tiny classifier on a few hundred labeled crops. Store team labels into `tracks.json`/`trajectories.json` (bump the stage version).

**4.5 — Defender matchup assignment (new — the cheapest unlock in the diagnostics pillar).** Nothing upstream pairs an offensive player with the defender guarding them, and half of what a coach wants to hear ("he never got separation," "the switch was late," "nobody picked up the roller") is a statement about that pair. This stage runs on Phase 3 trajectories + Phase 4 team labels and **needs no ball**, so build it here rather than waiting for 2.3b:
- **Per-frame assignment:** Hungarian matching (`scipy.optimize.linear_sum_assignment`) between offensive and defensive tracks, minimizing a cost that is mostly distance but can include "is between the man and the basket."
- **Temporal smoothing is mandatory.** Frame-independent matching flips assignments constantly. Smooth with a Viterbi pass or hysteresis + windowed majority vote, so a genuine switch reads as one event rather than forty.
- **Handle fewer than 10 tracks.** Track fragments from Phase 2 mean you often have 8 or 9 players, not 10. Allow unmatched slots instead of forcing a bad pairing.
- **Emit both a matchup signal and a matchup-free one.** Man-to-man assignment is *meaningless against a zone*, and HS teams play a lot of zone (2-3, 1-3-1). So write two things: `matchups.json` (assigned defender per offensive track, with a confidence) **and** a simple **nearest-defender distance** series per offensive player. Nearest-defender is robust, needs no assignment, and never breaks against zone — it's what most of 7.5 should actually consume. Assigned-defender is richer where it's valid. A cheap man-vs-zone classifier on defensive spacing tells you which to trust; when in doubt, fall back to nearest-defender.

**Checkpoint:** annotated playback and top-down view color players by team; non-players are filtered; hovering an offensive player highlights their assigned defender and shows live separation distance.

---

## PHASE 5 — Play segmentation + in-app labeling tool (≈ 3–4 weeks)

Goal: cut continuous film into discrete possessions/plays, and build the tool you'll use to label them.

**5.0 — Define the play taxonomy and write a labeling guide BEFORE labeling anything.** Pick the 5–8 play types, write one page defining each with edge-case rules ("a PnR that flows into iso counts as…"), and include an explicit **"other/unknown"** class. Fuzzy taxonomy is the #1 silent killer here: inconsistent labels put a hard ceiling on Phase 6 accuracy that no model fixes, and re-labeling later costs weeks. Sanity-check the taxonomy with a coach if you can.

**5.1 — Pick the simplest segmentation that works on your footage**, in this order:
- **Trajectory-based first (new default):** you already have normalized trajectories from Phase 3 — possession changes show up as all ten players flowing to the other half, halfcourt crossings, and bursts/lulls in aggregate speed. This needs no scoreboard and no ball detection. Start here.
- Scoreboard present → OCR the game/shot clock to refine boundaries and catch dead balls.
- Ball-based motion cues (tip-off/inbound detection) → an enhancement, and where ball detection from 2.3 earns its keep.
- Fallback → a lightweight binary "in-play vs dead-ball" clip classifier.

**5.2 — Emit play segments.** Write `plays` rows: start/end timestamps, which team has possession, links to the trajectory slice. Even rough cuts unlock everything downstream.

**5.3 — Build the in-app labeling tool (do not skip).** A review UI where you scrub a segmented play, see its top-down trajectory, and assign a play-type label (from your 5.0 taxonomy) + outcome. This tool is **quadruple**-purpose: it produces your training labels, becomes the human-in-the-loop review UI for coaches, is where per-player roster tagging happens (7.0), and is where diagnostics get confirmed or rejected (7.5.5). Store labels as first-class data.

While you're in here, add two cheap capture affordances that pay off much later: a **ball-handler tag** (click the player with the ball at a few moments per play — training/eval data for 2.3b's possession attribution) and a free-text **"what went wrong"** note. The free-text notes are not training data; they're how you discover the diagnostic vocabulary coaches actually use before you commit to the taxonomy in 7.5.0.

**5.4 — Start labeling.** Label every play in your 2.5 hrs (~250–350 possessions). You need a few hundred before training anything in Phase 6.

**Checkpoint:** film auto-cuts into plays; you can label each play's type + outcome in-app; your first labeled set exists.

---

## PHASE 6 — Play classification (the core ML) (≈ 4–6 weeks)

Goal: automatically classify each segmented play. This is where your ML strength pays off — but ship it as *assistive* first, autonomous later.

**6.0 — Baseline before transformer.** Before any deep model, train a baseline on handcrafted trajectory features (player-pair distances, screen proximity events, ball-side, motion energy, paint touches) with XGBoost/logistic regression, or a small GRU/TCN. With only a few hundred labels, the baseline may well beat a transformer — and it validates the eval harness, sets the floor, and tells you whether errors come from the model or from label noise/segmentation.

**6.1 — Assemble pretraining data — knowing it is UNLABELED.** Gather NBA/college **trajectory** data (the released 2015–16 SportVU tracking, academic datasets, public sets) and map it into your normalized court space. Critical correction to plan around: **these datasets do not come with play-type labels**, so "pretrain the classifier on NBA data" cannot mean supervised pretraining. Respect the legal notes — prefer licensed/released datasets over broadcast rips; keep league data to pretraining, not your shipped product's core training set.

**6.2 — Trajectory transformer with self-supervised pretraining.** Build the encoder: player positions over a play window as tokens, with team/role + time embeddings. **Pretrain self-supervised** — masked-trajectory modeling / next-position prediction / contrastive between augmented views — on *all* unlabeled trajectories you have: the NBA/college data **plus every HS game your pipeline processes** (that pile grows for free with every upload). Then fine-tune a classification head on your labeled HS plays. Optionally add weak supervision: rule-based heuristic detectors (e.g. screener-proximity PnR detection) can noisy-label the NBA data as an auxiliary task. This framing also mostly dissolves the class-frequency-mismatch worry — SSL pretraining doesn't care that the NBA skews to PnR/iso; handle residual imbalance at fine-tune time with stratified sampling/reweighting.

**6.3 — Augment.** Court mirror (horizontal flip), small plane rotations, time warping, per-player track dropout — effectively multiplies your small HS set 5–10x. Consider synthetic-trajectory pretraining if real data is thin.

**6.4 — Ship as ASSISTIVE classification first.** With a few-hundred-play training set, do not promise autonomous 85% on day one. Integrate the model as the final CV stage writing **top-3 suggested play types + confidences** into `plays`; the review tool (5.3) presents suggestions and a human confirms with one click. Every confirmation is a fresh training label (your active-learning loop). **Full-auto is a graduation criterion, not the launch state:** flip a play type to auto-accept only when it clears your eval bar on held-out teams. This de-risks the whole product timeline — the product is useful to a coach even while the model is mediocre.

**6.5 — Iterate against the eval harness** (see Evaluation track). Don't chase a number blindly — look at the confusion matrix and fix the most-confused class pairs; check whether errors trace back to segmentation or tracking before blaming the classifier.

**Checkpoint:** a new uploaded game runs end-to-end and emits suggested play types with confidences; confident classes auto-accept, uncertain ones surface for one-click review.

---

## PHASE 7 — Stats, playbook, dashboards (≈ 3–4 weeks)

Goal: turn classified plays into the coach-facing product — charts, success rates, playbook.

**7.0 — Player identity (required for "by player" stats — was missing from the plan).** Nothing upstream produces *who* a track is; track IDs fragment and carry no name. V1 path: **manual roster tagging in the review tool** — the coach (or you) assigns jersey numbers/names to team-colored tracks per game, with the tool propagating identity across a play. Jersey-number OCR is the later automation, not the v1 dependency. Without this step, 7.2's per-player stats are impossible.

**7.1 — Outcome / success detection.** Infer play outcome: scoreboard OCR for points where available (much easier than ball-rim interaction detection — prefer it when the scoreboard exists), or ball-rim interaction for makes/misses; possession result (score, turnover, stop). Manual confirm in the review tool is acceptable for v1.

**7.1b — Shot events with a shooter attached (new — the bridge to 7.5).** A possession-level make/miss is enough for success rates, but not for diagnostics: "that was a contested shot" needs a **shot event** — timestamp of release, shot location on the normalized court, and *which player took it*. Build it here:
- **Rim location is free.** Court registration (Phase 3) already puts the rim at a known point in normalized space, so you don't need to detect the hoop — you know where it is. This makes "ball trajectory heading rimward, apex, then rim interaction" a much easier test than it sounds.
- **Shooter = last possessor before the ball goes in-air toward the rim**, straight off `possession.json` from 2.3b.
- **Write shot location, not just make/miss** — shot charts fall out for free, and 7.5.2 needs the location to judge shot quality.
- **Known limitation, write it down:** a ground-plane homography gives you **no height information**. You cannot measure jump height, arm extension, or a hand in the shooter's face. Every "contested" judgment in this system is a proxy built on ground distance and closeout speed. Say that plainly in the UI rather than implying a precision you don't have.

**7.2 — Aggregate stats.** Success rate by play type, by player (via 7.0), by quarter, frequency distributions, and the same for the opponent — **opponent tendencies**. Store in `stats`.

**7.3 — Playbook / X's-and-O's view.** Generate top-down diagrams per play type for both teams — but use the **medoid/representative play** (or cluster-then-medoid) rather than averaged trajectories: averaging across plays is ill-posed (players aren't aligned or consistently ordered across instances) and produces mush. D3/Konva.

**7.4 — Dashboards.** Recharts dashboards: your team's playbook + success rates, opponent scouting report, searchable play library (scrub to any tagged play). pgvector on trajectory embeddings (you have an encoder from 6.2) enables "find plays similar to this one."

**Checkpoint:** a coach gets a real scouting report: their playbook, the opponent's tendencies, success rates, and a searchable tagged film library.

---

## PHASE 7.5 — Possession diagnostics: *why* a play worked (≈ 8–14 weeks, ball-gated)

Goal: the second product pillar. Phase 6 says *what* was run; Phase 7.1 says *whether it scored*. This phase says **why** — the cutter never gained separation, that shot was contested, the weak-side corner was open. This is the part a coach can't get from a stat sheet, and it's the reason to buy.

*(Numbered 7.5 rather than renumbered into 8 to keep every existing "before Phase 8" cross-reference valid. It is a first-class phase, not an appendix.)*

**Read this before starting: diagnostics are held to a different standard than classification.** A play label is a fuzzy summary — calling a possession "PnR" when it flowed into iso is forgivable. A diagnostic is a **specific, checkable, falsifiable claim about a named 16-year-old**. If you tell a coach "#12 was four feet behind his man" and he rewinds the film and sees otherwise, you don't lose one label — you lose his trust in the whole system, permanently, in a small word-of-mouth community. Three consequences run through everything below:
- **Precision over recall, always.** Surfacing three high-confidence observations per game beats surfacing thirty with five wrong ones. Saying nothing is a valid, safe output.
- **Show the evidence, not the verdict.** Every observation ships with the numbers behind it and a scrub-to-moment link. "Separation 1.4 ft at the catch → 1.1 ft at the shot" invites the coach to check; "bad cut" invites him to argue.
- **Error compounds down the longest chain in the system.** decode → detect → track → homography → teams → matchup → ball → possession → shot → contest → judgment. Every stage's error multiplies into the final claim. Before shipping any observation type, trace what upstream error does to it.

**7.5.0 — Diagnostic taxonomy + measurable definitions, before any code.** Same discipline as 5.0, and for the same reason: fuzzy definitions cap quality no model recovers. For each observation type write the **numeric rule**, the **threshold**, and the **error bars** — "open teammate = nearest defender > 6 ft AND no defender within 3 ft of the passing lane segment." Pull the vocabulary from the free-text notes you collected in 5.3, and pressure-test the thresholds with a coach — coaches disagree about "wide open," and you need *one* definition you can defend. Include an explicit **"not confident enough to say"** state.

**7.5.1 — Tier 1: descriptive, matchup-based, no ball required.** Start here, because it's unblocked by 4.5 and doesn't wait on 2.3b:
- **Separation** — nearest-defender (and assigned-defender) distance over time per offensive player; flag "never created separation" on cuts and off-ball actions.
- **Closeout / recovery speed** — how fast the distance closes, from the derivative of the same series.
- **Screen effectiveness** — did the ball-handler's defender's distance spike at the screen, and for how long? This is one of the highest-value diagnostics available and it's pure geometry.
- **Spacing** — average inter-teammate distance, players in the paint, whether the floor was compressed.
- **Defensive shape** — man vs. zone from the 4.5 classifier, gaps in the zone.

These are the safest claims in the phase: they're measurements, they degrade gracefully, and they need nothing you don't already have. **Their accuracy is bounded directly by your homography error budget from 0.2** — if median projection error is 3 ft, a 4 ft separation reading is noise wearing a number's clothes. Check the budget before trusting these, and consider suppressing absolute distances in favor of *relative changes over time* (which partly cancel systematic registration error).

**7.5.2 — Tier 2: shot quality (gated on ball, 2.3b + 7.1b).** Nearest-defender distance at release, closeout velocity, shot location, and time-to-release. Compare against a **shot-quality baseline** — start with a simple empirical model of make-rate by (distance to rim, nearest-defender distance) fit on whatever labeled shots you have, and note that NBA tracking data actually *does* help here in a way it doesn't for play types, because shot outcomes exist in it. That gives you "this was a 32% shot, your average look is 44%" instead of a bare adjective. Respect the height limitation from 7.1b.

**7.5.3 — Tier 3: openness + passing lanes (gated on possession attribution).** Per teammate, per moment: nearest-defender distance (already have it) plus **passing-lane clearance** — the minimum perpendicular distance from any defender to the segment between handler and teammate. Cheap geometry once you know who has the ball. Two honest caveats to encode: openness in a coach's sense includes *shooting range and whether that player should be shooting*, which needs roster context (7.0) and per-player tendencies — v1 is geometric only; and a teammate being open is not the same as the pass being available, since the handler may be mid-move with his back turned.

**7.5.4 — Tier 4: prescriptive judgments — rules first, and deliberately conservative.** "Should have cut," "should have passed," and "should have taken that shot" are **counterfactual value claims**, categorically different from the measurements above: they require knowing what the alternative was worth. With a few hundred HS possessions you cannot learn that, so v1 is **hand-authored rules with coach-set thresholds**, and only for the small set of cases where the geometry is unambiguous (a teammate open by >8 ft in the corner for >1.5 s while the handler holds the ball). Ship these behind a higher confidence bar than everything else, phrased as observations rather than verdicts — *"corner was open 8+ ft for 1.8 s"*, not *"should have passed."* Let the coach draw the conclusion; he's better at it than the model and he knows what was called in the huddle.

**7.5.5 — Surface in the review tool, and harvest the corrections.** Reuse the 5.3 UI: observations appear on a possession timeline, each scrub-linked to its moment with its evidence numbers visible, each with confirm / reject / "not what happened" controls writing to `observations.coach_verdict`. This is the same assistive-first loop as 6.4 — every rejection is a threshold-tuning signal, and per-type confirm rates tell you which observation types are ready to show by default and which need to stay behind a "show low-confidence" toggle.

**7.5.6 — Deferred: a learned value model (EPV-style).** The principled version of 7.5.4 — estimate the expected points of the current possession state, compare against counterfactual states, and let "should have passed" fall out of the value difference. This is genuinely research-grade (6–10+ weeks), needs large tracking data with outcomes, and carries a real transfer question: NBA shot-quality distributions are not HS shot-quality distributions. Do not attempt it before the rule-based version is in coaches' hands and the descriptive tiers are trusted.

**Difficulty and dependency at a glance:**

| Step | Effort | Difficulty | Gated on | Risk |
|---|---|---|---|---|
| 7.5.1 descriptive / separation | 2–3 wk | Low–moderate | 4.5 only | Homography error budget |
| 7.5.2 shot quality | ~2 wk | Moderate | 2.3b, 7.1b | No height data; needs shot sample |
| 7.5.3 openness / lanes | ~1–2 wk | Low | possession attribution | Geometry ≠ basketball openness |
| 7.5.4 prescriptive rules | 2–3 wk + ongoing | Moderate (judgment, not CV) | 7.5.3 | False positives burn trust fast |
| 7.5.6 value model | 6–10+ wk | High / research | large outcome data | Defer past v1 |
| **2.3b ball + possession** (upstream) | **3–6 wk, may plateau** | **High** | — | **The schedule risk for this whole phase** |

**Checkpoint:** for a single possession, a coach sees the play type (Phase 6), the outcome (7.1), and a short ranked list of evidence-backed observations about *why* — each scrub-linked to the moment, each confirmable in one click.

---

## PHASE 8 — Coach feedback loop + iteration (≈ ongoing, months 7–9+)

Goal: real users, real corrections, real accuracy gains.

**8.1 — Recruit 1–3 friendly coaches** (HS / D3). Trade free access + custom opponent reports for their season film and weekly feedback. One reliable coach beats ten survey responses. (Privacy/consent items from 0.5 must be in place before this.)

**8.2 — Harden the review UI** so coaches correct mislabels easily — every correction is premium training data feeding back into Phase 6, and every confirmed play moves a class closer to auto-accept graduation. The same loop runs for Phase 7.5 observations: track **per-observation-type confirm rate** as your headline trust metric, and pull any type that sits below ~90% confirmed rather than leaving it up and hoping it improves. A diagnostic type coaches keep rejecting is worse than no diagnostic at all.

**8.3 — Iterate on the real failure modes** they surface: accuracy on confused play pairs, edge cases (occlusions, unusual formations, lighting), UI friction. Expect this to dominate your time — and to be where the product actually becomes good.

**8.4 — Only then expand:** more play types, a second camera angle, or football — one new axis at a time, never several at once.

---

## PARALLEL TRACK A — Data collection (start in Phase 0, never stop)

Variety beats volume: 5 games from 5 teams > 25 games from 1. Ask coaches early for season **Hudl exports** — and inspect one in Phase 0 since it's the real input format. Rough targets for a *generalizing* classifier: 200–500 plays = demo; 2,000–5,000 = decent v1; 10,000+ = production. Your 2.5 hrs is runway, not the destination — keep the funnel filling from Phase 0 onward so you're not data-starved when you hit Phase 6. Note the dependency explicitly: **Phase 6's autonomous mode gates on this track reaching ~2,000+ labeled plays**; the assistive mode (6.4) is what makes the product shippable before then.

## PARALLEL TRACK B — Evaluation harness (build before Phase 6, ideally Phase 5)

Before training anything, set up: a held-out test set (games/teams the model never trains on — test generalization, not memorization), metrics (per-class precision/recall, confusion matrix, overall accuracy), and W&B/MLflow experiment tracking. Define your "good enough to auto-accept" bar up front (~85% per-class on held-out teams is roughly the coach-tolerance floor).

**Also build stage-level golden sets, not just classifier metrics:** a small fixed set of clips with hand-verified tracks, homographies, and play boundaries, run as regression tests whenever you bump an upstream stage. Otherwise a tracking regression silently shows up weeks later as "the classifier got worse" and you debug the wrong layer. Extend the golden set to the new stages as they land: hand-verified **ball possession** spans (2.3b), **defender matchups** (4.5), and a **measured homography error distribution in feet** (0.2) — that last one is a prerequisite for trusting anything in 7.5.1, not a nice-to-have.

**Diagnostics need a different eval than classification.** Accuracy and confusion matrices don't fit a system whose correct answer is often "say nothing." Evaluate Phase 7.5 on **precision at a fixed volume** — of the top 5 observations surfaced per game, how many does a coach confirm? — and hold each observation type to a precision bar (~90%+) before it ships on by default. Recall is a secondary concern and deliberately so: missing a real observation costs you nothing visible; asserting a false one costs trust you can't buy back.

---

## Ordering rationale (why this sequence)

- **Skeleton before ML (Phase 1 before 2):** the classic failure is building a great model attached to a product that never ships. Plumbing-first guarantees a shippable spine.
- **Feasibility spike before skeleton (Phase 0):** with limited CV experience, your biggest unknown is "can the CV even work on my footage." Answer that in throwaway code before investing in scaffolding.
- **Registration before team assignment (3 before 4 — revised):** non-player filtering needs the court polygon (the old order had a circular dependency), and registration is the riskier stage — surface risk early. Team assignment is quick and slots in afterward.
- **Strict CV stage order (2→3→4→5→6):** each stage literally consumes the previous stage's artifact. Trajectories (3) feed segmentation and classification; team labels (4) define what a possession is.
- **Labeling tool in Phase 5, not later:** you need it to produce Phase 6 labels, and it doubles as the coach review UI and the roster-tagging surface, so building it once serves all three.
- **Stats/playbook last among build phases (7):** they're pure functions of classified plays; nothing depends on them, so they can't come earlier and shouldn't be polished before classification works.
- **Matchup in Phase 4, diagnostics in 7.5 (new):** defender pairing needs only trajectories and team labels, so it slots in cheaply at 4 and unblocks the entire no-ball tier of diagnostics early. The rest of 7.5 consumes possession, shot events, and outcomes, so it genuinely cannot come before 7.
- **Ball promoted from optional to required (new):** the old plan deferred ball tracking because nothing depended on it. Diagnostics do. Ball is now the schedule risk for the second pillar, which is why it gets spiked in Phase 0 — you want that bad news in week two, not month six.
- **Coaches in Phase 8, not Phase 1:** show them something that works end-to-end, or early roughness burns goodwill in a small, word-of-mouth community.

## Solo-dev guardrails (taped to your monitor)

- One sport, one camera type, one play taxonomy until it works. Scope kills these projects, not skill.
- Lock each artifact's schema when you create it; bump the **stage** version instead of breaking it, so re-running stage 7 never forces re-running 1–6.
- Measure before you fine-tune; don't optimize a stage that's already good enough.
- **Know your $/game-hour at all times** (from 2.7); a pipeline you can't afford to run per-upload isn't a product.
- Deploy early and keep it deployed; integrate continuously rather than big-bang.
- Assistive-with-review beats autonomous-but-wrong at every stage — homography re-clicks, play suggestions, outcome confirms, diagnostic observations. Human-in-the-loop is the v1 strategy, not a compromise.
- **A diagnostic is a falsifiable claim about a real kid; a play label is a summary.** Hold Phase 7.5 to precision, show the evidence behind every observation, and let "not confident enough to say" be a normal output. Three right observations beat thirty with five wrong ones.
- **Know your homography error in feet, not vibes.** Everything in 7.5.1 is a distance measurement; if you don't know your error bars you don't know whether you're measuring basketball or noise.
- Keep this roadmap and the hand-off doc as living files — update them as decisions change.

---

## Revision log — 2026-07-12

1. **Phase 6 pretraining corrected:** public NBA/college tracking data has no play-type labels, so supervised pretraining as originally written was impossible. Replaced with self-supervised trajectory pretraining (+ optional weak labels), fine-tuned on HS labels.
2. **Player identity gap closed:** original plan promised per-player stats (7.2) but nothing produced player identity. Added 7.0 (manual roster tagging in the review tool; jersey OCR deferred).
3. **Phases 3 and 4 swapped:** court registration before team assignment — removes the circular dependency in ref filtering and fronts the riskier stage.
4. **Registration strategy demoted from research to engineering:** one-time manual calibration + frame-to-frame propagation is the v1 plan of record; fully-automatic keypoint detection is a later enhancement.
5. **Tracking framerate corrected:** track at 15–30 fps (association breaks at 5 fps on fast breaks); downsample trajectories afterward for downstream stages.
6. **Annotated video replaced with client-side overlays** rendered from `tracks.json` — no GPU re-encode, free iteration on visuals.
7. **Phase 6 reframed as assistive-first:** top-3 suggestions + one-click confirm at launch; auto-accept per class only after clearing the eval bar on held-out teams.
8. **Added:** baseline model before transformer (6.0); taxonomy + labeling guide before labeling (5.0); trajectory-based possession segmentation as the default (5.1); multipart/resumable uploads (1.2); per-stage versioning + resumable DAG (2.1); $/game cost budget (2.7); embedding-based team clustering (4.1); trajectory smoothing (3.4); medoid playbook diagrams (7.3); stage-level golden-set regression tests (Track B); Hudl export inspection in Phase 0; minors/privacy checklist (0.5); explicit go/no-go thresholds in the spike (0.2).
9. **Queue decision flagged (1.4):** once Modal lands in Phase 2, either keep Celery as a thin shim or cut Redis+Celery and spawn Modal directly — decide, don't drift into both.

---

## Revision log — 2026-07-26 (diagnostics pillar)

The product is now explicitly **two pillars**: *what was run* (Phase 6) and *why it worked* (Phase 7.5). The second pillar was absent from the original plan — 7.1 produced only a possession-level make/miss, and nothing produced defender relationships, shot contest, or openness. Changes:

1. **New PHASE 7.5 — possession diagnostics**, in four tiers: descriptive/matchup-based (no ball), shot quality, openness + passing lanes, and prescriptive rule-based judgments; plus a deferred learned value model (7.5.6). Numbered 7.5 rather than renumbering Phase 8, to keep existing cross-references valid.
2. **Ball tracking promoted from optional to a hard dependency** (2.3b). The old 2.3 allowed deferring it "if it's fighting you"; possession attribution now gates half of 7.5. Written up as its own stage with a fine-tuning budget, ballistic tracking, an explicit `possession.json` artifact, and a trajectory-only fallback for when detection plateaus. Flagged as the largest schedule risk in the roadmap.
3. **Ball detection added to the Phase 0 spike** (0.2) — if it's hopeless on your footage, that must surface in week two.
4. **Homography error budget added to the Phase 0 go/no-go** (0.2). ±2–3 ft is adequate for classification and inadequate for diagnostics, since contested-vs-open is a ~4 ft distinction. Measure median error in feet against known court points on a panning segment; target under ~1.5 ft.
5. **New 4.5 — defender matchup assignment** (Hungarian matching + temporal smoothing), and Phase 4 retitled/re-estimated to 2–3 weeks. Emits both assigned-defender and **nearest-defender** signals, because man-matchup is meaningless against zone and HS teams play a lot of zone.
6. **New 7.1b — shot events with an attributed shooter** (release time, court location, shooter track). Rim position comes free from Phase 3 registration. Documents the hard limitation that ground-plane homography carries **no height information**, so "contested" is always a ground-distance proxy.
7. **`observations` table added to schema v1** (1.1) so diagnostics are first-class data with evidence and coach verdicts from the start.
8. **5.3 labeling tool extended** with ball-handler tagging (training data for 2.3b) and free-text "what went wrong" notes (to discover coach vocabulary before fixing the 7.5.0 taxonomy).
9. **Evaluation track extended** (Track B): diagnostics are evaluated on **precision at fixed volume**, not accuracy — the correct output is frequently "say nothing." Per-type precision bar (~90%) before shipping on by default; golden sets extended to possession spans, matchups, and measured homography error.
10. **Governing principle for the new pillar:** a diagnostic is a specific, falsifiable claim about a named player, unlike a fuzzy play label — so show evidence rather than verdicts, prefer precision over recall, and phrase prescriptive items as observations ("corner open 8+ ft for 1.8 s") rather than judgments ("should have passed").

---

## Revision log — 2026-08-27 (detector + tracker swap)

1. **Detector: YOLO11 → YOLO26** in the Phase 0 spike (0.2) and as the baseline detector throughout. End-to-end/NMS-free, so there is no NMS threshold to tune — but confidence scores are not comparable to the YOLOv8/11 line, so the score threshold must be set from your own footage rather than copied.
2. **Tracker: ByteTrack/BoT-SORT → OC-SORT** in 0.2 and 2.3. Rationale added inline at 2.3: ByteTrack's constant-velocity Kalman assumption is a poor fit for basketball's non-linear motion, and OC-SORT's observation-centric re-update + momentum are designed for exactly that. Two consequences recorded: OC-SORT is not an Ultralytics built-in (pull via `boxmot`, budget glue code in the detect→track stage), and it is motion-only — if the 0.2 spike shows ID switches through screens dominating, the upgrade path is an appearance-augmented variant (Deep OC-SORT / BoT-SORT+ReID), decided on spike evidence rather than up front.
3. **Unchanged by this swap:** the 15–30 fps tracking requirement (a motion-model improvement does not buy back large inter-frame displacement), the `tracks.json` contract (2.4), and the fine-tune-only-if-needed rule (2.8).
