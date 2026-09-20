"""The canonical class list, and per-source remapping into it.

Public sets disagree: some label `player`, some split `player`/`referee`, some add
`hoop`/`net`. Merging without an explicit remap silently trains on scrambled class
ids. Classes we do not use get dropped here, not ignored downstream.
"""
