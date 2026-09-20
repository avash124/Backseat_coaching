"""Ball detection: a separate high-resolution / tiled small-object pass (roadmap 2.3b).

Not the same forward pass as players — base COCO `sports ball` weights collapse on a
blurred 15 px smear, so this stage owns its own fine-tuned weights.
"""
