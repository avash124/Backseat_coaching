"""RETIRED. Detector training now goes through `tracking/remote/train.py`.

This standalone script trained players-v1 (H100, batch 4, imgsz 1280, 60 epochs,
yolo26s, `/datasets/players_v1`). It recorded no dataset hash, no resolved config and
no weights manifest, and its GPU class was wrong for the job. Its replacement keeps the
same volumes (`nextup-datasets`, `nextup-checkpoints`) and defaults, on an L4:

    .\\.venv\\Scripts\\modal.exe run --env main --detach -m tracking.remote.train --config <request.yaml>

The body is removed so an old `modal run tracking/train_modal.py` command fails loudly
here instead of silently starting an unrecorded H100 run.
"""

raise RuntimeError(
    "tracking/train_modal.py is retired; use `modal run -m tracking.remote.train --config ...` "
    "(see tracking/remote/train.py)"
)
