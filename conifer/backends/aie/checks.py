import logging

import numpy as np

logger = logging.getLogger(__name__)

# BV_WORDS generalises, and depth 6 is already the two-word case,
# but the study was conducted up to depth 6. Other kernel variants
# may be more optimized for deeper trees.
# TODO: remove the cap
MAX_DEPTH = 6

_TOL = 1e-9


def _rows(tree):
    return np.asarray(tree.weight, dtype=np.float64)


def _is_leaf(feature_i):
    return feature_i == -2


def classify_row(row):
    '''One of "leaf", "axis", "oblique", "unsupported"'''
    nz = row[np.abs(row) > _TOL]
    if nz.size == 0:
        return "leaf"
    if nz.size == 1 and abs(nz[0] - 1.0) <= _TOL:
        return "axis"
    if np.all(np.abs(np.abs(nz) - 1.0) <= _TOL):
        return "oblique"
    return "unsupported"


def survey(trees):
    """Walk every node once. Returns (is_oblique, [(tree, node, value), ...] unsupported)"""
    oblique = False
    unsupported = []
    for ic, trees_c in enumerate(trees):
        for it, tree in enumerate(trees_c):
            rows = _rows(tree)
            for n, row in enumerate(rows):
                if _is_leaf(tree.feature[n]):
                    continue
                kind = classify_row(row)
                if kind == "oblique":
                    oblique = True
                elif kind == "unsupported":
                    oblique = True
                    bad = row[np.abs(row) > _TOL]
                    bad = bad[np.abs(np.abs(bad) - 1.0) > _TOL]
                    unsupported.append((ic, it, n, float(bad[0])))
    return oblique, unsupported


def check_weights(trees):
    """Reject any oblique split whose projection weights are not binary +-1

    TODO: ModelBase.is_oblique() tests sum(row) not in (0, 1), which a signed +-1 row passes
    about half the time, so it is not usable here.
    """
    oblique, unsupported = survey(trees)
    if unsupported:
        ic, it, n, val = unsupported[0]
        raise NotImplementedError(
            f"The aie backend supports oblique splits with binary +-1 projection weights "
            f"only. Class {ic} tree {it} node {n} has weight {val}. "
            f"{len(unsupported)} node(s) affected. Train with ydf "
            f'sparse_oblique_weights="BINARY" (its default), which stays +-1 under every '
            f"normalization ydf offers"
        )
    return oblique


def check_max_depth(max_depth):
    if max_depth > MAX_DEPTH:
        raise ValueError(
            f"max_depth {max_depth} is not supported yet: the aie backend goes up to "
            f"{MAX_DEPTH}"
        )
    if max_depth < 1:
        raise ValueError(f"max_depth must be at least 1, got {max_depth}")


def check_n_classes(n_classes):
    if n_classes > 2:
        raise NotImplementedError(
            f"The aie backend scores one value per sample, so it supports binary and "
            f"regression models only, got n_classes={n_classes}"
        )


def check_n_tiles(n_tiles, device_tiles, plio_channels_out):
    if n_tiles < 1:
        raise ValueError(f"n_tiles must be at least 1, got {n_tiles}")
    if n_tiles > device_tiles:
        raise ValueError(
            f"n_tiles {n_tiles} exceeds the {device_tiles} AI Engine tiles on this device"
        )
    if plio_channels_out and n_tiles > plio_channels_out:
        raise ValueError(
            f"n_tiles {n_tiles} exceeds the {plio_channels_out} outgoing PLIO channels this "
            f"platform routes: every tile emits its own partial score on its own channel, "
            f"on both split axes"
        )


def check_vector_width(W, feat_bytes):
    if W < 1 or W & (W - 1):
        raise ValueError(f"W must be a power of two, got {W}")
    per_word = 4 // feat_bytes
    if per_word and W % per_word:
        raise ValueError(
            f"W {W} must be a multiple of the {per_word} feature values in a 32-bit PLIO "
            f"word at this compare width"
        )


def padded_n_features(n_features, oblique):
    """The oblique kernel loads a node weight row as one vector, so it needs a power of two

    Padding with all-zero weight columns is value-preserving: no axis-aligned split
    references them and they add zero to a projection.
    """
    if not oblique or n_features < 1:
        return n_features
    return 1 << (n_features - 1).bit_length()


def warn_tile_memory(breakdown, tile_memory_bytes):
    """Report when a tile's memory is likely to be exceeded using our estimate. The toolchain has the final say"""
    total = breakdown["total"]
    if total <= tile_memory_bytes:
        return
    parts = ", ".join(
        f"{k} {v / 1024:.1f} kB"
        for k, v in sorted(breakdown.items(), key=lambda kv: -kv[1])
        if k != "total"
    )
    logger.warning(
        f"Estimated tile data memory {total / 1024:.1f} kB exceeds the "
        f"{tile_memory_bytes / 1024:.1f} kB a tile has ({parts}). The build may fail in"
        f"the AIE mapper: raise n_tiles, reduce the ensemble, or narrow the precision"
    )


def warn_score_range(n_trees, max_leaf_value, score_precision):
    """the accumulator is the user's to size, but we report saturation"""
    worst = n_trees * abs(max_leaf_value)
    if worst > score_precision.max_representable:
        logger.warning(
            f"score_precision {score_precision.type_string} holds "
            f"{score_precision.max_representable:.4g} but the ensemble can reach "
            f"{worst:.4g} ({n_trees} trees x {abs(max_leaf_value):.4g}). Scores may "
            f"saturate. Widen the integer bits of score_precision"
        )
