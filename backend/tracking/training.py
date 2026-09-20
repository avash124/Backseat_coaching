"""Detector fine-tuning logic (roadmap 2.8 players, 2.3b ball).

Framework-level training only — no Modal imports, so this stays runnable locally on a
few frames to catch config errors before paying for a GPU. `remote/train.py` provisions
the hardware and calls in here.

Sequence that matters: pretrain on the merged public sets, then fine-tune on our own
labeled frames. Public basketball data is mostly broadcast footage; HS tripod film is a
different domain, and a model trained only on the former will look good on its own val
split and disappoint on real uploads. Measure against our footage (evaluate.py), and per
2.8 do not fine-tune at all if the base model already clears the 0.2 bar.

Runs are tracked in W&B; produced weights are versioned and referenced from config.py.
"""
