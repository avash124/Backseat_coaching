"""Dataset assembly: pull public sets, remap their classes, merge, split.

A package rather than a module because training data now comes from several sources
with different label taxonomies, and the merge/remap step is where that gets
reconciled before anything reaches the trainer.

    taxonomy.py       frozen canonical classes + per-source remap tables (raise on unmapped)
    common.py         naming convention, label remap, data.yaml, dataset_sha256
    splits.py         group-wise splits, disjointness + quarantine enforcement
    prepare_ebard.py  eBAR export -> players_v1 / full_v1
    own_footage.py    corrected CVAT "YOLO 1.1" exports / images+labels -> dataset
    merge.py          public + own -> one dataset, explicit own-footage oversampling
    roboflow.py       pinned Roboflow pulls (license-gated, content-addressed)
    sources.py        sources.yaml loader
    synthetic.py      tiny synthetic frames for smoke tests
"""
