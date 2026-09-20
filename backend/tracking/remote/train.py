"""GPU training job on Modal — the wrapper, not the training logic.

Responsibilities: request the GPU, mount the dataset + checkpoint Volumes, inject
Secrets, call `tracking.training`, then promote the finished weights out of the Volume
and record the run id in W&B. Detached (`modal run --detach`) so a closed laptop does
not kill a multi-hour run.
"""
