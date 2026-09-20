"""Pull Roboflow datasets by pinned (workspace, project, version).

Always pin the version — Roboflow projects change under you, and an unpinned pull
makes a training run unreproducible. API key comes from a Modal Secret, never a
committed env var.
"""
