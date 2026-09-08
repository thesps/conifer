'''
Random conifer model generation for performance scans.
'''
import numpy as np

# conifer tree serialisation fields expected by the backends
TREE_FIELDS = ['feature', 'weight', 'threshold', 'value', 'children_left', 'children_right']
_LIBRARY = 'conifer'
_SPLITTING_CONVENTION = '<'
LEAF = -2


class _Node:
  '''Internal mutable node used while building a random tree.'''
  def __init__(self, i):
    self.i = i
    self.feature = LEAF          # >=0 marks a split node for the backends
    self.features = []           # active feature indices (1 for axis-aligned, k for oblique)
    self.weights = []            # weights aligned with self.features
    self.threshold = 0.0
    self.value = 0.0
    self.child_left = None
    self.child_right = None

  def is_leaf(self):
    return self.child_left is None and self.child_right is None


class RandomTree:
  '''
  A single randomised decision tree with axis-aligned or oblique splits.
  Thresholds of axis-aligned splits are drawn to be logically reachable along every root-to-node path.
  '''
  def __init__(self, rng, max_depth, n_features, sparsity=0.0, split='axis',
               oblique_k=3, oblique_weight_scale=1.0, threshold_range=32.0, value_range=32.0):
    if split not in ('axis', 'oblique'):
      raise ValueError(f"split must be 'axis' or 'oblique', got {split!r}")
    self.rng = rng
    self.max_depth = int(max_depth)
    self.n_features = int(n_features)
    self.split = split
    self.oblique_k = max(1, min(int(oblique_k), self.n_features))
    self.oblique_weight_scale = float(oblique_weight_scale)
    self.threshold_range = float(threshold_range)
    self.value_range = float(value_range)

    self.nodes = [_Node(0)]
    n = 1
    while len(self.nodes) < 2 ** self.max_depth - 1:
      n = self._add_children_one_node(n)
    self._assign_splits()
    self._prune_to_sparsity(float(np.clip(sparsity, 0.0, 0.999)))
    self._rationalise_indices()
    self._sort()
    self._assign_thresholds()
    self._add_leaves()

  def _add_children_one_node(self, n):
    '''Attach two children to the first childless node.'''
    for node in self.nodes:
      if node.is_leaf():
        cl, cr = _Node(n), _Node(n + 1)
        node.child_left, node.child_right = cl, cr
        self.nodes += [cl, cr]
        return n + 2
    return n

  def _assign_splits(self):
    '''Choose the active feature(s) and (for oblique) hyperplane weights of every split node.'''
    for node in self.nodes:
      if self.split == 'axis':
        f = int(self.rng.integers(0, self.n_features))
        node.feature, node.features, node.weights = f, [f], [1.0]
      else:
        k = int(self.rng.integers(1, self.oblique_k + 1))
        feats = np.sort(self.rng.choice(self.n_features, size=k, replace=False))
        w = self.rng.normal(0.0, self.oblique_weight_scale, size=k)
        if not np.any(w):
          w[0] = self.oblique_weight_scale or 1.0
        node.feature = int(feats[0])
        node.features = [int(x) for x in feats]
        node.weights = [float(x) for x in w]

  def _prune_to_sparsity(self, sparsity):
    '''Remove random leaves until the requested fraction of the full tree is gone.'''
    def _current():
      return 1 - len(self.nodes) / (2 ** self.max_depth - 1)
    while _current() < sparsity and len(self.nodes) >= 2:
      leaves = [nd for nd in self.nodes if nd.is_leaf()]
      victim = leaves[int(self.rng.integers(0, len(leaves)))]
      parent, side = self._parent_of(victim)
      self.nodes.remove(victim)
      setattr(parent, f'child_{side}', None)

  def _parent_of(self, node):
    for nd in self.nodes:
      if nd.child_left is node:
        return nd, 'left'
      if nd.child_right is node:
        return nd, 'right'
    return None, None

  def _rationalise_indices(self):
    '''Renumber nodes so indices are contiguous from 0.'''
    for new_i, node in enumerate(self.nodes):
      node.i = new_i

  def _sort(self):
    self.nodes.sort(key=lambda nd: nd.i)

  def _assign_thresholds(self):
    '''Axis-aligned: reachable thresholds along each path; oblique: scaled to the projection spread.'''
    parents = {id(nd): self._parent_of(nd) for nd in self.nodes}
    for node in self.nodes:
      if node.is_leaf():
        continue
      if self.split == 'oblique':
        spread = np.sqrt(np.sum(np.square(node.weights))) * self.threshold_range / np.sqrt(3.0)
        node.threshold = float(self.rng.uniform(-spread, spread)) if spread > 0 else self.threshold_range
        continue
      lo, hi = -self.threshold_range, self.threshold_range
      child, (parent, side) = node, parents[id(node)]
      while parent is not None:
        if parent.feature == node.feature:
          if side == 'left':
            hi = min(hi, parent.threshold)
          else:
            lo = max(lo, parent.threshold)
        child, (parent, side) = parent, parents[id(parent)]
      if lo >= hi:
        lo, hi = -self.threshold_range, self.threshold_range
      node.threshold = float(self.rng.uniform(lo, hi))

  def _add_leaves(self):
    '''Give every missing child a value leaf.'''
    n = len(self.nodes)
    for node in list(self.nodes):
      for side in ('left', 'right'):
        if getattr(node, f'child_{side}') is None:
          leaf = _Node(n)
          leaf.value = float(self.rng.uniform(-self.value_range, self.value_range))
          setattr(node, f'child_{side}', leaf)
          self.nodes.append(leaf)
          n += 1

  def to_dict(self):
    '''Return the tree as a conifer tree dict (dense one-hot weights for axis-aligned splits).'''
    idx = {id(nd): nd.i for nd in self.nodes}
    out = {f: [] for f in TREE_FIELDS}
    for node in self.nodes:
      leaf = node.is_leaf()
      w = [0.0] * self.n_features
      if not leaf:
        for f, wt in zip(node.features, node.weights):
          w[f] = wt
      out['feature'].append(LEAF if leaf else int(node.feature))
      out['weight'].append(w)
      out['threshold'].append(0.0 if leaf else float(node.threshold))
      out['value'].append(float(node.value) if leaf else 0.0)
      out['children_left'].append(-1 if leaf else idx[id(node.child_left)])
      out['children_right'].append(-1 if leaf else idx[id(node.child_right)])
    return out


def random_model(rng, *, n_trees, max_depth, n_features, n_classes=2, sparsity=0.0,
                 split='axis', oblique_k=3, oblique_weight_scale=1.0,
                 threshold_range=32.0, value_range=32.0, norm=1):
  '''
  Build a random conifer ensembleDict ready for conifer.model.make_model.
  sparsity may be a scalar or a per-tree sequence of length n_trees.
  '''
  n_trees, max_depth, n_features = int(n_trees), int(max_depth), int(n_features)
  n_classes = int(n_classes)
  spars = np.broadcast_to(np.asarray(sparsity, dtype=float), (n_trees,))
  n_class_slots = 1 if n_classes == 2 else n_classes
  trees = []
  for i in range(n_trees):
    per_class = [RandomTree(rng, max_depth, n_features, spars[i], split, oblique_k,
                            oblique_weight_scale, threshold_range, value_range).to_dict()
                 for _ in range(n_class_slots)]
    trees.append(per_class)
  init_predict = [float(rng.uniform(-value_range, value_range)) for _ in range(n_class_slots)]
  return {
    'n_classes': n_classes,
    'n_features': n_features,
    'n_trees': n_trees,
    'max_depth': max_depth,
    'init_predict': init_predict,
    'norm': norm,
    'library': _LIBRARY,
    'splitting_convention': _SPLITTING_CONVENTION,
    'trees': trees,
  }
