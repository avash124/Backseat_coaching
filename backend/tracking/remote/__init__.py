"""Modal entry points — the only place in the repo that imports `modal`.

Named `remote/` rather than `modal/` on purpose: a local package named `modal` will
shadow the pip package the moment anything runs with this directory on sys.path.

Everything here is a thin wrapper. The actual work lives in `tracking.training` and
`tracking.pipeline`, which stay runnable on a laptop against a handful of frames.
"""
