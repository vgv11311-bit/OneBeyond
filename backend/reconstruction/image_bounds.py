"""Normalize provider image bounds before segmentation or inventory matching."""
import math


def normalized_box(values):
    box = list(map(float, values))
    if len(box) != 4 or not all(math.isfinite(v) for v in box):
        raise ValueError('Invalid image bounds')
    # Providers sometimes mix normalized X with their 0..1000 Y convention.
    # Normalize each axis independently; applying one scale to both silently
    # creates subpixel strips and corrupts every downstream image operation.
    for lo, hi in ((0, 2), (1, 3)):
        if box[hi] > 1:
            box[lo] /= 1000
            box[hi] /= 1000
        if not 0 <= box[lo] < box[hi] <= 1:
            raise ValueError('Invalid image bounds')
    return box
