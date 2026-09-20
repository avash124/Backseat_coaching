"""Train/val/test split policy.

Split by game, never by frame — adjacent frames are near-duplicates, and a frame-level
split leaks them across the boundary and inflates every metric. The held-out test set
is our own footage only (see evaluate.py); public val splits measure the wrong domain.
"""
