"""
QuickScorer tree ensemble traversal. Implementation of Algorithm 2 from
https://doi.org/10.1145/2766462.2767733 in Python and Numpy.

Rather than traversing each tree root-to-leaf, QuickScorer represents every
internal node with a bitvector that masks off the leaves of its left subtree,
and performs an interleaved, feature-by-feature traversal of the whole ensemble.

  Step 1: for every feature k, the thresholds of all the nodes testing it
    (across all trees of the ensemble) are stored in a sorted array. Given
    an input x, the nodes whose test fails form a prefix of that (sorted) array,
    delimited by the position of x[k] among the sorted thresholds. The bitvector
    of each false node is ANDed into the result bitvector v[h] of the tree h the
    node belongs to.

  Step 2: after all features are processed, the exit leaf of tree h is the
    leftmost set bit in v[h] (Theorem 1 of the paper). The output values
    of the exit leaves are summed to produce the score.

BWQS (blockwise QS) improves the cache behaviour of QS on large ensembles:
the ensemble is split into disjoint blocks of 'tau' trees, each with its own
copy of the relevant data structures, and blocks of 'delta' samples are scored
together over one tree block before moving to the next.

Details on how this implementation differs from the original paper:
 - Leaves are numbered left-to-right; leaf j of a tree is stored at bit j with
   bit 0 the *least* significant bit. The paper's "leftmost bit set to 1" is
   therefore the least significant set bit here (the bit order is mirrored,
   which changes nothing in the algorithm).
 - Bitvectors are always full machine words (numpy uint64), so trees may have
   at most 64 leaves. The limit is inherent to the bitvector representation.
   The paper instead pads bitvectors to B in {1, 2, 4, 8} bytes fitting Lambda
   (the max leaves per tree), so for shallow ensembles (e.g. Lambda <= 32,
   B = 4) its bitvectors array would be half the size of the one here.
 - The thresholds and tree ids arrays are float64/int64, as opposed to the
   paper's float/uint (4 bytes). We chose float64 thresholds to make the traversal
   identical to the plain tree walk for correctness checks. Together with the
   above, the read-only structures should be around ~1.7x the size reported in
   Table 1 of the paper for the same ensemble.
 - leaf_nodes (needed for return_leaf=True) has no counterpart in the paper,
   and doubles the size of the leaf tables.
 - The paper finds the false-node prefix with a linear skip scan. Here, numpy
   searchsorted (binary search) computes the same prefix boundary, vectorized
   over a whole sample block at once.
 - The paper scores one sample at a time and keeps one result bitvector v[h]
   per tree. samples are vectorized with small delta (up to 16) to fit the
   delta replicas of v in cache. Here the whole sample block is processed by
   each numpy operation, so v is a (delta, tau) matrix and useful delta values
   need to be much larger (small blocks cannot amortize the numpy overhead).
   In this batch setting the v matrix occupies the most cache space.
 - Unlike the paper's BWQS, no running partial score per sample is
   accumulated across tree blocks. Each tree's leaf values are gathered and
   summed once at the end, in the same order as the plain python tree walk, so
   that treewalk, QS and BWQS return identical floating point scores.
   Accumulating block partial sums would change the floating point ordering!
"""

import numpy as np


class _TreeBlock:
    """
    The QS data structures and interleaved traversal for a block of trees.
    """

    def __init__(self, flat_trees, n_features, side):
        """
        Parameters
        ----------
        flat_trees: list of DecisionTreeBase
            the trees of this block
        n_features: int
            number of input features of the ensemble
        side: 'left' or 'right'
            numpy searchsorted side locating the false-node prefix boundary
            ('left' for the "<=" splitting convention, 'right' for "<")
        """
        self.n_features = n_features
        self.side = side
        n_flat = len(flat_trees)

        # per-tree structures
        all_leaf_values = []  # leaf output values, left-to-right
        all_leaf_nodes = []  # original node index of each leaf
        # per-node triples, grouped per feature
        triples = [
            [] for _ in range(n_features)
        ]  # feature -> [(threshold, tree_id, bitvector)]

        for h, tree in enumerate(flat_trees):
            leaf_nodes, left_ranges = self._tree_leaf_layout(tree)
            n_leaves = len(leaf_nodes)
            if n_leaves > 64:
                raise NotImplementedError(
                    f"QuickScorer bitvectors are 64-bit words, but a tree has {n_leaves} leaves"
                )
            all_leaf_values.append([tree.value[n] for n in leaf_nodes])
            all_leaf_nodes.append(leaf_nodes)
            ones = (1 << n_leaves) - 1
            for n, (a, b) in left_ranges.items():
                self._check_axis_aligned(tree, n)
                # node bitvector: 0s at the leaves of the left subtree (bits a..b), 1s elsewhere
                bitvector = ones ^ (((1 << (b - a + 1)) - 1) << a)
                triples[tree.feature[n]].append((tree.threshold[n], h, bitvector))

        # global arrays
        thresholds, tree_ids, bitvectors = [], [], []
        self.offsets = np.zeros(n_features + 1, dtype=np.int64)
        for k in range(n_features):
            # ascending threshold within each block
            triples[k].sort(key=lambda t: t[0])
            for t, h, bv in triples[k]:
                thresholds.append(t)
                tree_ids.append(h)
                bitvectors.append(bv)
            self.offsets[k + 1] = len(thresholds)
        self.thresholds = np.array(thresholds, dtype=np.float64)
        self.tree_ids = np.array(tree_ids, dtype=np.int64)
        self.bitvectors = np.array(bitvectors, dtype=np.uint64)

        # initialize result bitvectors: all leaves of each tree are candidate exit leaves
        self.init_v = np.array(
            [(1 << len(lv)) - 1 for lv in all_leaf_values], dtype=np.uint64
        )
        # leaf output values (and original node ids), one row per tree, padded to the widest tree
        max_leaves = max(len(lv) for lv in all_leaf_values)
        self.leaves = np.zeros((n_flat, max_leaves), dtype=np.float64)
        self.leaf_nodes = np.full((n_flat, max_leaves), -1, dtype=np.int64)
        for h in range(n_flat):
            self.leaves[h, : len(all_leaf_values[h])] = all_leaf_values[h]
            self.leaf_nodes[h, : len(all_leaf_nodes[h])] = all_leaf_nodes[h]

    def nbytes(self):
        """Total size in bytes of the traversal data structures (cf. Table 1 of the paper)"""
        return (
            self.thresholds.nbytes
            + self.tree_ids.nbytes
            + self.bitvectors.nbytes
            + self.offsets.nbytes
            + self.init_v.nbytes
            + self.leaves.nbytes
            + self.leaf_nodes.nbytes
        )  # leaf_nodes is specific to this implementation

    @staticmethod
    def _tree_leaf_layout(tree):
        """
        Number the leaves of a tree left-to-right and, for every internal node,
        record the (contiguous) range of leaf numbers in its left subtree.

        Returns
        ----------
        leaf_nodes: list of int
            original node index of each leaf, in left-to-right order
        left_ranges: dict of int -> (int, int)
            internal node index -> inclusive range (first, last) of the leaf
            numbers in its left subtree
        """
        leaf_nodes = []
        left_ranges = {}

        def visit(n):
            if tree.feature[n] == -2:  # leaf
                j = len(leaf_nodes)
                leaf_nodes.append(n)
                return j, j
            la, lb = visit(tree.children_left[n])
            rb = visit(tree.children_right[n])[1]
            left_ranges[n] = (la, lb)
            return la, rb

        visit(0)
        return leaf_nodes, left_ranges

    @staticmethod
    def _check_axis_aligned(tree, n):
        """Verify internal node n tests exactly one feature with unit weight"""
        w = tree.weight[n]
        nonzero = [i for i, wi in enumerate(w) if wi != 0]
        if nonzero != [tree.feature[n]] or w[tree.feature[n]] != 1:
            raise NotImplementedError(
                "QuickScorer supports only axis-aligned trees (one feature per split), "
                f"but node {n} has weight vector {w}"
            )

    def exit_leaves(self, X):
        """
        Run the interleaved traversal (Steps 1 and 2 of Algorithm 2) of this
        block's trees for a block of samples.

        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
            Input sample block

        Returns
        ----------
        j: ndarray of shape (n_samples, n_trees_in_block)
            left-to-right index of the exit leaf of every tree, for every sample
        """
        n_samples = X.shape[0]

        # QS step 1: interleaved traversal, feature by feature. All samples of the
        # block are processed together: node i of a feature block (ascending
        # thresholds) is a false node for sample d iff i < cuts[d], and its
        # bitvector is ANDed into that sample's result bitvector for the node's
        # tree.
        v = np.repeat(
            self.init_v[np.newaxis, :], n_samples, axis=0
        )  # result bitvectors, (n_samples, n_trees)
        for k in range(self.n_features):
            start, end = self.offsets[k], self.offsets[k + 1]
            if start == end:
                continue
            # boundary of the false nodes prefix for every sample
            # TODO: compare with the paper's stepped linear scan (paper claimed binary search doesn't provide gains)
            cuts = np.searchsorted(self.thresholds[start:end], X[:, k], side=self.side)
            for i in range(end - start):
                rows = cuts > i
                if not rows.any():
                    break  # thresholds are sorted: this node is a true node for all samples, so are the rest
                v[rows, self.tree_ids[start + i]] &= self.bitvectors[start + i]

        # QS step 2: the exit leaf of each tree is the leftmost candidate leaf left
        # in v. As explained in the top, that is the LSB.
        # Note that v is never 0 since the exit leaf bit is never cleared.
        lsb = v & ~(v - np.uint64(1))
        return np.log2(lsb.astype(np.float64)).astype(
            np.int64
        )  # exact integer as lsb is a pow of 2


class QuickScorer:
    """
    QuickScorer scorer for a conifer ModelBase.

    Precomputes the QS data structures from the model's trees at construction,
    then scores inputs with `decision_function`.

    With the default `tau=None, delta=None` this is the non-blocked QS
    algorithm: one set of data structures for the whole ensemble, all
    samples traversed together. Passing `tau` and/or `delta` gives BWQS: the
    ensemble is split into blocks of `tau` trees, each with its own small data
    structures, and samples are scored `delta` at a time over one tree block
    before moving to the next.
    """

    def __init__(self, model, tau=None, delta=None):
        """
        Parameters
        ----------
        model: ModelBase
            conifer model to build the QuickScorer data structures from.
            Must be axis-aligned with at most 64 leaves per tree.

        tau: int, optional
            number of trees per block. Defaults to the whole ensemble.

        delta: int, optional
            number of samples scored together over one tree block. Defaults to
            all samples. Note: unlike the paper's per-sample C implementation
            (best delta up to 16), useful values here are in the thousands, since
            each numpy operation processes the whole sample block.
        """
        if model.is_oblique():
            raise NotImplementedError(
                "QuickScorer supports only axis-aligned trees (one feature per split), "
                "but this model contains oblique splits"
            )
        self.n_features = model.n_features
        self.n_trees = model.n_trees
        self.n_classes = 1 if model.n_classes == 2 else model.n_classes
        self.init_predict = np.array(model.init_predict, dtype=np.float64)
        self.norm = model.norm
        # false node iff threshold < x ("<=" convention) or threshold <= x ("<")
        assert model.splitting_convention in ("<", "<="), (
            f"Unknown splitting convention {model.splitting_convention}"
        )
        side = "left" if model.splitting_convention == "<=" else "right"

        # one result bitvector lane per (tree, class) pair, h = i_tree * n_classes + i_class
        flat_trees = [tree for trees_class in model.trees for tree in trees_class]
        self.n_flat = len(flat_trees)
        self.delta = delta
        tau = self.n_flat if tau is None else tau
        assert tau > 0, f"tau must be positive, got {tau}"
        assert delta is None or delta > 0, f"delta must be positive, got {delta}"
        # blocks of tau trees, each with its own copy of the QS data structures
        self.blocks = [
            (h0, _TreeBlock(flat_trees[h0 : h0 + tau], self.n_features, side))
            for h0 in range(0, self.n_flat, tau)
        ]

    def nbytes(self):
        """Total size in bytes of the traversal data structures"""
        return sum(block.nbytes() for _, block in self.blocks)

    def decision_function(self, X, return_leaf=False):
        """
        Score input samples with the QuickScorer traversal, block by block
        if tau/delta blocking is configured.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
            Input sample

        return_leaf: bool, optional
            If True, returns the exit leaf node indices of each tree in the
            ensemble instead of the score. Defaults to False.

        Returns
        ----------
        score: ndarray of shape (n_samples, n_classes) or (n_samples,)
        """
        X = np.asarray(X, dtype=np.float64)
        assert len(X.shape) == 2, "Expected 2D input"
        assert X.shape[1] == self.n_features, (
            f"Wrong number of features, expected {self.n_features}, got {X.shape[1]}"
        )
        n_samples = X.shape[0]
        delta = n_samples if self.delta is None else self.delta

        # per-tree exit leaf values (or node ids), gathered across all blocks
        out = np.empty((n_samples, self.n_flat), dtype=np.float64)
        for h0, block in self.blocks:  # loop over blocks of tau trees
            h1 = h0 + len(block.init_v)
            table = block.leaf_nodes if return_leaf else block.leaves
            for d0 in range(0, n_samples, delta):  # loop over blocks of delta samples
                j = block.exit_leaves(X[d0 : d0 + delta])
                out[d0 : d0 + delta, h0:h1] = table[np.arange(h1 - h0), j]

        out = out.reshape(n_samples, self.n_trees, self.n_classes)
        if return_leaf:
            # match the layout of ModelBase.decision_function(return_leaf=True)
            return np.squeeze(out.transpose(0, 2, 1))  # (n_samples, n_classes, n_trees)

        # aggregate with the same array layout and operations as
        # ModelBase.decision_function, for identical floating point results
        y = np.ascontiguousarray(
            out.transpose(1, 2, 0)
        )  # (n_trees, n_classes, n_samples)
        y = (np.transpose(np.sum(y, axis=0)) + self.init_predict) * self.norm
        return np.squeeze(y)
