import numpy as np
from conifer.model import ModelBase, DecisionTreeBase



class DecisionNode:
  def __init__(self, feature, threshold, score, left=None, right=None):
    self.f = feature
    self.t = threshold
    self.s = score
    self.l = left
    self.r = right

  def addChild(self, left=None, right=None):
    if left is not None:
      self.l = left
    elif right is not None:
      self.r = right

  def isLeaf(self):
    return (self.l is None) and (self.r is None)
  
  def children(self):
    return [n for n in [self.l, self.r] if n is not None]
  
  def isDescendantOf(self, node):
    """
    Return True if self a descendant of node
    """
    # direct child check
    if self in node.children():
        return True
    
    # recursive check
    for child in node.children():
        if self.isDescendantOf(child):
            return True
    
    return False

  def isRightOf(self, node):
    """
    Return True if self is the right child of node,
    or is a descendant of node's right child.
    """
    if node is None or node.r is None:
        return False
    
    # direct right child
    if self is node.r:
        return True

    # recursive: check subtree of node.right
    return self.isDescendantOf(node.r)

  def isLeftOf(self, node):
    """
    Return True if self is the left child of node,
    or is a descendant of node's left child.
    """
    if node is None or node.l is None:
        return False
    
    # direct left child
    if self is node.l:
        return True

    # recursive: check subtree of node.l
    return self.isDescendantOf(node.l)

class DecisionTree(DecisionTreeBase):
  def __init__(self, treeDict):
    super(DecisionTree, self).__init__(treeDict, "<=")
    nodes = []
    cl = self.children_left
    cr = self.children_right
    t = self.threshold
    f = self.feature
    v = self.value

    def addChildren(node, i, cl, cr, t, f, v):
      if cl[i] != -1:
        child = DecisionNode(f[cl[i]], t[cl[i]], v[cl[i]])
        node.addChild(left=child)
        addChildren(child, cl[i], cl, cr, t, f, v)
      if cr[i] != -1:
        child = DecisionNode(f[cr[i]], t[cr[i]], v[cr[i]])
        node.addChild(right=child)
        addChildren(child, cr[i], cl, cr, t, f, v)

    self.root = DecisionNode(f[0], t[0], v[0])
    addChildren(self.root, 0, cl, cr, t, f, v)
    self.masks = self.computeMasks()

  def leaves(self):
    l = []
    def recurse(node, l):
      if node.isLeaf():
        l += [node]
      for node in node.children():
        recurse(node, l)
    node = self.root
    recurse(node, l)
    return l
  
  def nonLeaves(self):
    l = []
    def recurse(node, l):
      if not node.isLeaf():
        l += [node]
      for node in node.children():
        recurse(node, l)
    node = self.root
    recurse(node, l)
    return l
  
  def computeMasks(self):
    m = ~np.array([[l.isLeftOf(nl) for l in self.leaves()] for nl in self.nonLeaves()])
    return m
  
  def apply(self, x):
    y = np.array([x[n.f] <= n.t for n in self.nonLeaves()])
    v = np.logical_or(y[:, np.newaxis], self.masks)
    v = np.logical_and.reduce(v)
    il = np.argwhere(v)[0][0]
    return self.leaves()[il].s
    
class Model(ModelBase):

  def __init__(self, ensembleDict, config, metadata=None):
    super(Model, self).__init__(ensembleDict, config, metadata)
    trees = ensembleDict.get('trees', None)
    self.trees = [[DecisionTree(tree) for tree in trees_class] for trees_class in trees]

    feature = np.array([f for trees_class in self.trees for tree in trees_class for f in tree.feature])
    non_leaf = feature != -2
    feature = feature[non_leaf]
    threshold = np.array([f for trees_class in self.trees for tree in trees_class for f in tree.threshold])
    threshold = threshold[non_leaf]
    itree = np.array([i for trees_class in self.trees for tree in trees_class for i in range(len(tree.feature))])
    itree = itree[non_leaf]

    idx = np.argsort(feature)
    self.feature = feature[idx]
    self.threshold = threshold[idx]
    self.itree = itree[idx]

  def decision_function(self, X):
        '''
        Compute the decision function of `X`.
        The backend performs the actual computation
        
        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
            Input sample
        
        Returns
        ----------    
        score: ndarray of shape (n_samples, n_classes) or (n_samples,)   

        '''
        assert len(X.shape) == 1, 'Expected 1D input'
        assert X.shape[0] == self.n_features, f'Wrong number of features, expected {self.n_features}, got {X.shape[1]}'

        n_classes = 1 if self.n_classes == 2 else self.n_classes
        #n_samples = X.shape[0]
        y = np.zeros((self.n_trees, n_classes))
        for it, trees in enumerate(self.trees):
            for ic, tree_c in enumerate(trees):
                y[it, ic] = tree_c.apply(X)
        y = (np.transpose(np.sum(y, axis=0)) + self.init_predict) * self.norm
        return np.squeeze(y)

def make_model(ensembleDict, config):
  return Model(ensembleDict, config)