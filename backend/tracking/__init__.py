"""Phase 2 CV package: decode -> detect -> track -> versioned artifacts.

Deliberately free of FastAPI/Celery imports so this package can be lifted onto a
Modal GPU function (roadmap 2.1) without dragging the API along. The worker calls
into `tracking.pipeline`; nothing here calls back into the worker.
"""
