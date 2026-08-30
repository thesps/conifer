import logging

import numpy as np

logger = logging.getLogger(__name__)

LEAF = -2

# One vector of tail slack on the term tables, so the kernel's unrolled reads never run
# off the end
_TERM_PAD = 64


def tree_depth(tree):
    """Deepest leaf, root at depth 0"""
    best, stack = 0, [(0, 0)]
    while stack:
        n, d = stack.pop()
        if tree.feature[n] == LEAF:
            best = max(best, d)
            continue
        stack.append((tree.children_left[n], d + 1))
        stack.append((tree.children_right[n], d + 1))
    return best


def _leaf_layout(tree):
    """Left-to-right leaf numbering, and each internal node's left-subtree leaf range"""
    leaf_nodes, left_ranges = [], {}

    def visit(n):
        if tree.feature[n] == LEAF:
            j = len(leaf_nodes)
            leaf_nodes.append(n)
            return j, j
        la, lb = visit(tree.children_left[n])
        _, rb = visit(tree.children_right[n])
        left_ranges[n] = (la, lb)
        return la, rb

    visit(0)
    return leaf_nodes, left_ranges


def build_pair_basis(qt_w, max_terms_floor=1):
    """Every distinct signed feature pair

    Canonical up to global sign: (i,j,+1,-1) is the negation of (i,j,-1,+1), so one is
    stored and the other references it with sign -1.
    """
    seen, b_i, b_j, b_wi, b_wj, terms = {}, [], [], [], [], []

    def entry(i, j, wi, wj):
        sign = 1
        if wi < 0 or (wi == 0 and wj < 0):
            wi, wj, sign = -wi, -wj, -1
        key = (int(i), int(j), int(wi), int(wj))
        b = seen.get(key)
        if b is None:
            b = len(b_i)
            seen[key] = b
            b_i.append(int(i))
            b_j.append(int(j))
            b_wi.append(int(wi))
            b_wj.append(int(wj))
        return b, sign

    for row in qt_w:
        k = [int(v) for v in np.nonzero(row)[0]]
        t = []
        for a in range(0, len(k), 2):
            if a + 1 < len(k):
                t.append(entry(k[a], k[a + 1], row[k[a]], row[k[a + 1]]))
            else:
                t.append(entry(k[a], k[a], row[k[a]], 0))
        terms.append(t)

    true_max_terms = max((len(t) for t in terms), default=1)
    max_terms = max(1, max_terms_floor)
    while max_terms < true_max_terms:
        max_terms <<= 1

    n = len(qt_w) * max_terms + _TERM_PAD
    bterm = np.zeros(n, dtype=np.int64)
    bsign = np.zeros(n, dtype=np.int64)
    for slot, t in enumerate(terms):
        for a, (b, sg) in enumerate(t):
            bterm[slot * max_terms + a] = b
            bsign[slot * max_terms + a] = sg

    return {
        "basis_i": np.asarray(b_i, dtype=np.int64),
        "basis_j": np.asarray(b_j, dtype=np.int64),
        "basis_wi": np.asarray(b_wi, dtype=np.int64),
        "basis_wj": np.asarray(b_wj, dtype=np.int64),
        "qt_bterm": bterm,
        "qt_bsign": bsign,
        "max_terms": max_terms,
        "true_max_terms": true_max_terms,
        "basis_n": len(b_i),
    }


class QuickScorerTables:
    """The QuickScorer tables the AIE kernels read, built from a conifer ensemble

    Grouping is by the compared quantity: a feature for an axis-aligned split, the
    projection vector for an oblique one, which keeps each group's false nodes a prefix
    and reduces to the paper's construction when every weight row is one-hot
    """

    def __init__(self, trees, n_features, oblique=False, weight_precision=None):
        self.n_features = n_features
        self.oblique = oblique
        self.weight_precision = weight_precision
        self.trees = trees
        self.n_trees = len(trees)
        if oblique and weight_precision is None:
            raise ValueError("an oblique model needs a weight precision to group by")
        self.max_depth = max(tree_depth(t) for t in trees)
        self._build()

    def _quantized_row(self, row):
        return tuple(int(v) for v in self.weight_precision.quantize(np.asarray(row)))

    def _seed_groups(self):
        if not self.oblique:
            return list(range(self.n_features)), {k: k for k in range(self.n_features)}
        directions, index = [], {}
        for k in range(self.n_features):
            onehot = np.zeros(self.n_features)
            onehot[k] = 1.0
            row = self._quantized_row(onehot)
            index.setdefault(row, len(directions))
            directions.append(row)
        return directions, index

    def _build(self):
        directions, index = self._seed_groups()
        entries = {}
        leaf_values = []
        nnz_max = 0

        for h, tree in enumerate(self.trees):
            leaf_nodes, left_ranges = _leaf_layout(tree)
            n_leaves = len(leaf_nodes)
            leaf_values.append([float(tree.value[n]) for n in leaf_nodes])
            ones = (1 << n_leaves) - 1
            for n, (a, b) in left_ranges.items():
                bv = ones ^ (((1 << (b - a + 1)) - 1) << a)
                if self.oblique:
                    row = self._quantized_row(tree.weight[n])
                    nnz_max = max(nnz_max, int(np.count_nonzero(row)))
                    g = index.get(row)
                    if g is None:
                        g = len(directions)
                        index[row] = g
                        directions.append(row)
                else:
                    g = int(tree.feature[n])
                    nnz_max = max(nnz_max, 1)
                entries.setdefault(g, []).append((float(tree.threshold[n]), h, bv))

        self.group_rows = directions
        self.nnz_max = nnz_max

        thresholds, tree_ids, bitvectors, group_of = [], [], [], []
        offsets = [0]
        for g in range(len(directions)):
            for thr, h, bv in sorted(entries.get(g, []), key=lambda t: t[0]):
                thresholds.append(thr)
                tree_ids.append(h)
                bitvectors.append(bv)
                group_of.append(g)
            offsets.append(len(thresholds))

        self.offsets = offsets
        self.thresholds = np.asarray(thresholds, dtype=np.float64)
        self.tree_ids = np.asarray(tree_ids, dtype=np.int64)
        self.bitvectors = bitvectors
        self.group_of = group_of
        self.n_nodes = len(thresholds)

        self.max_leaves = max(len(lv) for lv in leaf_values)
        self.init_v = [(1 << len(lv)) - 1 for lv in leaf_values]
        self.leaves = np.zeros((self.n_trees, self.max_leaves), dtype=np.float64)
        for h, lv in enumerate(leaf_values):
            self.leaves[h, : len(lv)] = lv

        self._regroup_tree_major()

    def _regroup_tree_major(self):
        """The same nodes tree by tree. The ANDs commute, so the score is unchanged"""
        self.nodes_per_tree = (1 << self.max_depth) - 1
        n_slots = self.n_trees * self.nodes_per_tree
        all_ones = (1 << self.bv_bits) - 1

        self.qt_group = [0] * n_slots
        self.qt_thr_f = [0.0] * n_slots
        self.qt_bv = [all_ones] * n_slots
        self.qt_used = [False] * n_slots
        fill = [0] * self.n_trees
        for i, h in enumerate(self.tree_ids):
            j = fill[h]
            if j >= self.nodes_per_tree:
                raise ValueError(
                    f"Tree {h} has more than {self.nodes_per_tree} internal nodes, which is "
                    f"the most a depth-{self.max_depth} tree can have"
                )
            slot = h * self.nodes_per_tree + j
            self.qt_group[slot] = self.group_of[i]
            self.qt_thr_f[slot] = float(self.thresholds[i])
            self.qt_bv[slot] = self.bitvectors[i]
            self.qt_used[slot] = True
            fill[h] = j + 1
        self.fill = fill
        self.n_slots = n_slots

    def qt_weight_rows(self):
        """Per-slot quantized weight rows; a padding slot is all zero, so it projects to 0"""
        rows = np.zeros((self.n_slots, self.n_features), dtype=np.int64)
        for slot in range(self.n_slots):
            if self.qt_used[slot]:
                rows[slot] = self.group_rows[self.qt_group[slot]]
        return rows

    def basis(self):
        if not self.oblique:
            return None
        return build_pair_basis(self.qt_weight_rows())

    @property
    def bv_bits(self):
        if self.max_leaves <= 16:
            return 16
        if self.max_leaves <= 32:
            return 32
        return 64

    @property
    def n_groups(self):
        return len(self.group_rows)

    def quantize(self, threshold_precision, score_precision, norm=1.0):
        """Quantized copies of every table that carries a real number"""
        return {
            "thresholds": threshold_precision.quantize(self.thresholds),
            "qt_thr": threshold_precision.quantize(np.asarray(self.qt_thr_f)),
            "leaves": score_precision.quantize(self.leaves * norm),
        }

    def replay(
        self,
        X,
        threshold_precision,
        score_precision,
        init_predict,
        norm=1.0,
        split_le=True,
    ):
        """Score X exactly as the kernel does, on the quantized tables"""
        q = self.quantize(threshold_precision, score_precision, norm)
        xq = threshold_precision.quantize(X)
        init_q = int(score_precision.quantize([float(init_predict) * norm])[0])
        qt_thr, leaves_q = q["qt_thr"], q["leaves"]
        wgt_shift = self.weight_precision.shift if self.oblique else 0
        rows = self.qt_weight_rows() if self.oblique else None

        out = np.empty(len(xq), dtype=np.int64)
        for s, x in enumerate(xq):
            acc = init_q
            for h in range(self.n_trees):
                v = self.init_v[h]
                base = h * self.nodes_per_tree
                for j in range(self.nodes_per_tree):
                    i = base + j
                    if self.oblique:
                        p = int(np.dot(rows[i], x)) >> wgt_shift
                    else:
                        p = int(x[self.qt_group[i]])
                    failed = qt_thr[i] < p if split_le else qt_thr[i] <= p
                    if failed:
                        v &= self.qt_bv[i]
                acc += int(leaves_q[h][(v & -v).bit_length() - 1])
            out[s] = acc
        return out
