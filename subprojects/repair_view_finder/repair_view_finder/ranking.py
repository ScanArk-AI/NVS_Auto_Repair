"""Greedy deduplicated ranking of candidate views."""

from __future__ import annotations

import numpy as np


def rank_views(records, n_vox, top_k, min_gain=1e-6):
    """Greedy set-cover over high-UQ voxels.

    records: list of dicts with keys cand_idx, bad_vox, bad_val.
    Returns a list of (cand_idx, gain, new_voxel_count), best first.
    """
    seen = np.zeros(n_vox, dtype=bool)
    remaining = list(range(len(records)))
    ranked = []
    while remaining and len(ranked) < top_k:
        best_i, best_gain, best_new = None, -1.0, 0
        for i in remaining:
            r = records[i]
            bad = r["bad_vox"]
            if len(bad) == 0:
                gain = 0.0
                new = 0
            else:
                mask = ~seen[bad]
                gain = float(np.sum(np.maximum(r["bad_val"][mask], 0.0)))
                new = int(mask.sum())
            if gain > best_gain:
                best_i, best_gain, best_new = i, gain, new
        if best_i is None or best_gain < min_gain:
            break
        ranked.append((records[best_i]["cand_idx"], best_gain, best_new))
        seen[records[best_i]["bad_vox"]] = True
        remaining.remove(best_i)
    return ranked
