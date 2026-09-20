"""Dataset assembly: pull public sets, remap their classes, merge, split.

A package rather than a module because training data now comes from several sources
with different label taxonomies, and the merge/remap step is where that gets
reconciled before anything reaches the trainer.
"""
