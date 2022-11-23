
def padTree(ensembleDict, treeDict):
  '''Pad a tree with dummy nodes if not perfectly balanced or depth < max_depth'''
  n_nodes = len(treeDict['children_left'])
  # while th tree is unbalanced
  while n_nodes != 2 ** (ensembleDict['max_depth'] + 1) - 1:
    for i in range(n_nodes):
      if treeDict['children_left'][i] == -1 and treeDict['depth'][i] != ensembleDict['max_depth']:
        treeDict['children_left'].extend([-1, -1])
        treeDict['children_right'].extend([-1, -1])
        treeDict['parent'].extend([i, i])
        treeDict['feature'].extend([-2, -2])
        treeDict['threshold'].extend([-2.0, -2.0])
        val = treeDict['value'][i]
        treeDict['value'].extend([val, val])
        newDepth = treeDict['depth'][i] + 1
        treeDict['depth'].extend([newDepth, newDepth])
        iRChild = len(treeDict['children_left']) - 1
        iLChild = iRChild - 1
        treeDict['children_left'][i] = iLChild
        treeDict['children_right'][i] = iRChild
    n_nodes = len(treeDict['children_left'])
  treeDict['iLeaf'] = []
  for i in range(n_nodes):
    if treeDict['depth'][i] == ensembleDict['max_depth']:
      treeDict['iLeaf'].append(i)
  return treeDict

def addParentAndDepth(treeDict):
  n = len(treeDict['children_left']) # number of nodes
  parents = [0] * n
  for i in range(n):
    j = treeDict['children_left'][i]
    if j != -1:
      parents[j] = i
    k = treeDict['children_right'][i]
    if k != -1:
      parents[k] = i
  parents[0] = -1
  treeDict['parent'] = parents
  # Add the depth info
  treeDict['depth'] = [0] * n
  for i in range(n):
    depth = 0
    parent = treeDict['parent'][i]
    while parent != -1:
      depth += 1
      parent = treeDict['parent'][parent]
    treeDict['depth'][i] = depth
  return treeDict

