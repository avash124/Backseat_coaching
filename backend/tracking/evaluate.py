"""Golden-set regression + tracker metrics (Parallel Track B).

Detection mAP/recall, and HOTA/IDF1/MOTA + ID-switch counts on hand-verified clips —
this is what decides OC-SORT vs. an appearance-augmented variant on evidence.

The eval set is **our own footage**, never a public dataset's val split. Public sets
are broadcast-angle; scoring against them measures how well we do on someone else's
camera. A model can gain 6 points of mAP on a Roboflow val split and lose accuracy on
real HS uploads, and only this file would catch it.
"""
