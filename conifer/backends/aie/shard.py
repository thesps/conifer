import logging
import random

import numpy as np

logger = logging.getLogger(__name__)


def tree_feature_sets(tables, n_trees, nodes_per_tree):
    """Features each tree tests. Padding slots hold feature 0 and must be skipped"""
    n = n_trees * nodes_per_tree
    used = list(tables.qt_used) + [False] * (n - len(tables.qt_used))
    group = list(tables.qt_group) + [0] * (n - len(tables.qt_group))
    return [
        frozenset(
            group[h * nodes_per_tree + j]
            for j in range(nodes_per_tree)
            if used[h * nodes_per_tree + j]
        )
        for h in range(n_trees)
    ]


def assign_contiguous(n_trees, n_shards):
    trees_per_tile = n_trees // n_shards
    return [
        list(range(s * trees_per_tile, (s + 1) * trees_per_tile))
        for s in range(n_shards)
    ]


def feature_permutation(groups, tree_feats, n_features, sweeps=20):
    """Order feature rows so each shard's form a near-contiguous window"""
    shard_feats = [
        frozenset().union(*[tree_feats[h] for h in g]) if g else frozenset()
        for g in groups
    ]
    owners = [
        [s for s, fs in enumerate(shard_feats) if f in fs] for f in range(n_features)
    ]
    fperm = list(range(n_features))
    for _ in range(sweeps):
        pos = {f: p for p, f in enumerate(fperm)}
        centre = [
            sum(pos[f] for f in fs) / len(fs) if fs else 0.0 for fs in shard_feats
        ]
        key = [
            (
                sum(centre[s] for s in owners[f]) / len(owners[f])
                if owners[f]
                else float("inf"),
                f,
            )
            for f in range(n_features)
        ]
        nxt = sorted(range(n_features), key=lambda f: key[f])
        if nxt == fperm:
            break
        fperm = nxt
    return fperm


def _windows_for(groups, tree_feats, fperm):
    inv = {f: p for p, f in enumerate(fperm)}
    out = []
    for g in groups:
        pos = [inv[f] for h in g for f in tree_feats[h]]
        out.append(max(pos) - min(pos) + 1 if pos else 1)
    return out


def _cost(groups, tree_feats, fperm):
    w = _windows_for(groups, tree_feats, fperm)
    return (max(w), sum(w))


def search_span(
    tree_feats, n_trees, n_features, n_shards, seed=0, restarts=6, iters=None
):
    """Assign trees and permute features together, minimising the window

    A memtile hands a tile a contiguous range of rows, so the only way to make a shard's
    rows contiguous is to choose the order they are written in
    """
    if iters is None:
        iters = min(120_000, max(6_000, 400 * (n_trees + n_features * n_shards)))
    rng = random.Random(seed)
    trees_per_tile = n_trees // n_shards
    best = None
    for _ in range(restarts):
        order = list(range(n_trees))
        rng.shuffle(order)
        g = [
            order[s * trees_per_tile : (s + 1) * trees_per_tile]
            for s in range(n_shards)
        ]
        fperm = list(range(n_features))
        rng.shuffle(fperm)
        cur = _cost(g, tree_feats, fperm)
        for _ in range(iters // restarts):
            if n_shards >= 2 and rng.random() < 0.5:
                a, b = rng.sample(range(n_shards), 2)
                i, j = rng.randrange(len(g[a])), rng.randrange(len(g[b]))
                g[a][i], g[b][j] = g[b][j], g[a][i]
                new = _cost(g, tree_feats, fperm)
                if new <= cur:
                    cur = new
                else:
                    g[a][i], g[b][j] = g[b][j], g[a][i]
            else:
                u, v = rng.sample(range(n_features), 2)
                fperm[u], fperm[v] = fperm[v], fperm[u]
                new = _cost(g, tree_feats, fperm)
                if new <= cur:
                    cur = new
                else:
                    fperm[u], fperm[v] = fperm[v], fperm[u]
        if best is None or cur < best[0]:
            best = (cur, [sorted(x) for x in g], list(fperm))
    return best[1], best[2]


class Sharding:
    """Trees assigned to tiles, with each tile's feature rows a contiguous window

    QT_FEAT[h] is read only by the tile owning tree h, so one table can be interpreted
    in a different local feature frame per shard
    """

    def __init__(
        self,
        tables,
        n_trees,
        n_features,
        n_shards,
        groups=None,
        fperm=None,
        optimize="search",
        seed=0,
    ):
        self.tables = tables
        self.n_trees = n_trees
        self.n_features = n_features
        self.n_shards = n_shards
        self.nodes_per_tree = tables.nodes_per_tree

        self.tree_feats = tree_feature_sets(tables, n_trees, self.nodes_per_tree)
        chosen = groups is None and fperm is None and n_shards > 1
        if chosen and optimize == "search":
            self.groups, self.fperm = search_span(
                self.tree_feats, n_trees, n_features, n_shards, seed=seed
            )
        else:
            self.groups = groups or assign_contiguous(n_trees, n_shards)
            self.fperm = (
                list(fperm)
                if fperm
                else feature_permutation(self.groups, self.tree_feats, n_features)
            )
        if sorted(h for g in self.groups for h in g) != list(range(n_trees)):
            raise ValueError("the tree assignment is not a partition of the ensemble")
        if sorted(self.fperm) != list(range(n_features)):
            raise ValueError("the feature permutation is not a permutation")

        self.perm = [h for g in self.groups for h in g]
        self.t_count = [len(g) for g in self.groups]
        self.t_begin = np.cumsum([0] + self.t_count[:-1]).tolist()
        self.owner = [s for s, g in enumerate(self.groups) for _ in g]

        self._windows()
        self._build_tables()

    def _windows(self):
        inv = {f: p for p, f in enumerate(self.fperm)}
        self.feats, self.offset = [], []
        for g in self.groups:
            pos = sorted({inv[f] for h in g for f in self.tree_feats[h]})
            # an all-null shard still needs one row
            if not pos:
                pos = [0]
            lo, hi = pos[0], pos[-1]
            self.offset.append(lo)
            self.feats.append([self.fperm[p] for p in range(lo, hi + 1)])
        self.n_feat = [len(f) for f in self.feats]

    def _build_tables(self):
        t, P = self.tables, self.nodes_per_tree
        n = self.n_trees * P
        base_group = list(t.qt_group) + [0] * (n - len(t.qt_group))
        base_used = list(t.qt_used) + [False] * (n - len(t.qt_used))
        all_ones = (1 << t.bv_bits) - 1
        base_bv = list(t.qt_bv) + [all_ones] * (n - len(t.qt_bv))
        base_thr = list(t.qt_thr_f) + [0.0] * (n - len(t.qt_thr_f))

        local = [{f: i for i, f in enumerate(fs)} for fs in self.feats]
        self.qt_group, self.qt_thr_f, self.qt_bv, self.qt_used = [], [], [], []
        for new_h, old_h in enumerate(self.perm):
            loc = local[self.owner[new_h]]
            for j in range(P):
                i = old_h * P + j
                self.qt_group.append(loc[base_group[i]] if base_used[i] else 0)
                self.qt_thr_f.append(base_thr[i])
                self.qt_bv.append(base_bv[i])
                self.qt_used.append(base_used[i])

        init_v = list(t.init_v) + [all_ones] * (self.n_trees - len(t.init_v))
        self.init_v = [init_v[h] for h in self.perm]
        leaves = np.zeros((self.n_trees, t.max_leaves), dtype=np.float64)
        leaves[: len(t.leaves)] = t.leaves
        self.leaves = leaves[self.perm]

    @property
    def max_rows_per_tile(self):
        """Rows the busiest tile reads"""
        return max(self.n_feat)

    @property
    def total_rows(self):
        return sum(self.n_feat)

    def verify(
        self,
        X,
        threshold_precision,
        score_precision,
        init_predict,
        norm=1.0,
        split_le=True,
    ):
        """Replay the sharded model and require it to equal the unsharded one exactly

        Sums partial scores the way the AIE array does, each shard from its own cut of the
        rows, so the permutation, the local frames and the row windows are all checked.
        """
        base = self.tables.replay(
            X,
            threshold_precision,
            score_precision,
            init_predict,
            norm=norm,
            split_le=split_le,
        )
        thr_q = threshold_precision.quantize(np.asarray(self.qt_thr_f))
        leaves_q = score_precision.quantize(self.leaves * norm)
        init_q = int(score_precision.quantize([float(init_predict) * norm])[0])
        xq = threshold_precision.quantize(X)
        P = self.nodes_per_tree

        total = np.zeros(len(X), dtype=np.int64)
        for s in range(self.n_shards):
            cut = xq[:, self.feats[s]]
            for si, x in enumerate(cut):
                acc = init_q if s == 0 else 0
                for new_h in range(self.t_begin[s], self.t_begin[s] + self.t_count[s]):
                    v = self.init_v[new_h]
                    for j in range(P):
                        i = new_h * P + j
                        if (
                            thr_q[i] < int(x[self.qt_group[i]])
                            if split_le
                            else thr_q[i] <= int(x[self.qt_group[i]])
                        ):
                            v &= self.qt_bv[i]
                    acc += int(leaves_q[new_h][(v & -v).bit_length() - 1])
                total[si] += acc
        return np.flatnonzero(total != base)
