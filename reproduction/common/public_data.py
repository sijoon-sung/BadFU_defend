"""Disjoint, class-stratified public reference splits."""
import hashlib
import numpy as np


def split_public(images, labels, ids, counts, seed, classes=10):
    """Three stratified, disjoint splits: reference, search, audit.

    Counts are PER CLASS. Pixel duplicates are rejected even under different IDs.
    """
    if images.dtype != np.uint8 or images.ndim != 4:
        raise ValueError("uint8 NHWC public images required")
    if (labels.shape != (len(images),) or labels.dtype.kind not in 'iu'
            or np.any(labels < 0) or np.any(labels >= classes)):
        raise ValueError("valid public integer labels required")
    if (ids.shape != labels.shape or ids.dtype.kind not in 'US'
            or len(set(ids.tolist())) != len(ids)):
        raise ValueError("unique public image IDs required")
    if len(counts) != 3 or min(counts) < 1:
        raise ValueError("three positive per-class counts required")
    rng, splits = np.random.default_rng(seed), [[], [], []]
    for label in range(classes):
        available = rng.permutation(np.flatnonzero(labels == label))
        if len(available) < sum(counts):
            raise ValueError(f"not enough public images for class {label}")
        start = 0
        for dest, count in zip(splits, counts):
            dest.extend(available[start:start + count].tolist())
            start += count
    result = [np.asarray(s, dtype=int) for s in splits]
    hashes = []
    for s in result:
        h = {hashlib.sha256(images[i].tobytes()).digest() for i in s}
        if len(h) != len(s) or any(h & old for old in hashes):
            raise ValueError("duplicate public pixels within or across splits")
        hashes.append(h)
    return result

