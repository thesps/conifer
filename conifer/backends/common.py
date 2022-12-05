from conifer.model import DecisionTree

class BottomUpDecisionTree(DecisionTree):
  _tree_fields = DecisionTree._tree_fields + ['parent', 'depth', 'iLeaf']
  def __init__(self, treeDict):
    for key in DecisionTree._tree_fields:
        val = treeDict.get(key, None)
        assert val is not None, f"Missing expected key {key} in treeDict"
        setattr(self, key, val)
    self.addParentAndDepth()

  def padTree(self, max_depth):
    '''Pad a tree with dummy nodes if not perfectly balanced or depth < max_depth'''
    n_nodes = len(self.children_left)
    # while th tree is unbalanced
    while n_nodes != 2 ** (max_depth + 1) - 1:
      for i in range(n_nodes):
        if self.children_left[i] == -1 and self.depth[i] != max_depth:
          self.children_left.extend([-1, -1])
          self.children_right.extend([-1, -1])
          self.parent.extend([i, i])
          self.feature.extend([-2, -2])
          self.threshold.extend([-2.0, -2.0])
          val = self.value[i]
          self.value.extend([val, val])
          newDepth = self.depth[i] + 1
          self.depth.extend([newDepth, newDepth])
          iRChild = len(self.children_left) - 1
          iLChild = iRChild - 1
          self.children_left[i] = iLChild
          self.children_right[i] = iRChild
      n_nodes = len(self.children_left)
    self.iLeaf = []
    for i in range(n_nodes):
      if self.depth[i] == max_depth:
        self.iLeaf.append(i)

  def addParentAndDepth(tree):
    n = len(tree.children_left) # number of nodes
    parents = [0] * n
    for i in range(n):
      j = tree.children_left[i]
      if j != -1:
        parents[j] = i
      k = tree.children_right[i]
      if k != -1:
        parents[k] = i
    parents[0] = -1
    tree.parent = parents
    # Add the depth info
    tree.depth = [0] * n
    for i in range(n):
      depth = 0
      parent = tree.parent[i]
      while parent != -1:
        depth += 1
        parent = tree.parent[parent]
      tree.depth[i] = depth
