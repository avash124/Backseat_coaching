"""Physics-aware ball tracking: ballistic Kalman filter + gap interpolation (2.3b).

The ball is invisible for large stretches (in hands, behind bodies) — this
reconstructs a path rather than reading one.
"""
