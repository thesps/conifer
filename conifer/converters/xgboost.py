import numpy as np
import json
from conifer.converters.common import addParentAndDepth, padTree

def convert(bdt):
    meta = json.loads(bdt.save_config())
    max_depth = int(meta['learner']['gradient_booster']['updater']['grow_colmaker']['train_param']['max_depth'])
    n_classes = int(meta['learner']['learner_model_param']['num_class'])
    fn_classes = 1 if n_classes == 0 else n_classes # the number of learners
    n_classes = 2 if n_classes == 0 else n_classes # the actual number of classes
    n_features = int(meta['learner']['learner_model_param']['num_feature'])
    ensembleDict = {'max_depth' : max_depth,
                    'n_trees' : int(len(bdt.get_dump()) / fn_classes),
                    'n_classes' : n_classes,
                    'n_features' : n_features,
                    'trees' : [],
                    'init_predict' : [0] * n_classes,
                    'norm' : 1}
    
    feature_names = {}
    if bdt.feature_names is None:
      for i in range(n_features):
        feature_names[f'f{i}'] = i
    else:
      for i, feature_name in enumerate(bdt.feature_names):
        feature_names[feature_name] = i

    trees = bdt.get_dump()
    for i in range(ensembleDict['n_trees']):
        treesl = []
        for j in range(fn_classes):
            tree = trees[fn_classes * i + j]
            tree = treeToDict(tree, feature_names)
            tree = addParentAndDepth(tree)
            #tree = padTree(ensembleDict, tree)
            treesl.append(tree)
        ensembleDict['trees'].append(treesl)
    return ensembleDict

def treeToDict(tree, feature_names):
  # First of all make the tree sklearn-like
  # split by newline, ignore the last line
  nodes = tree.split('\n')[:-1]
  # remove tab characters
  nodes = list(map(lambda x: x.replace('\t',''), nodes))
  real_nNodes = len(nodes)
  # Number of nodes that are in the tree
  # Pruning removes nodes but does not reset index 
  old_node_indices = []
  for i in range(real_nNodes):
    iNode = int(nodes[i].split(':')[0])
    old_node_indices.append(iNode)
    # Node indices that are left in the tree after pruning
  nNodes = max(old_node_indices)+1
  # Maximum Node index 
  nPrunedNodes = nNodes - len(old_node_indices) 
  if nPrunedNodes > 0:
    node_to_node_dict = dict(list(enumerate(sorted(old_node_indices))))
    node_to_node_dict = {value:key for key, value in node_to_node_dict.items()}
    # Create a dictionary remapping old Node indicies to new node indicies and invert
  features = [0] * nNodes
  thresholds = [0] * nNodes
  children_left = [0] * nNodes
  children_right = [0] * nNodes
  values = [0] * nNodes 
  for node in nodes:
    if node == '':
        pass
    elif 'leaf' in node: # is a leaf
      # Looks like: 'i:leaf=value[i]'
      data = node.split('leaf')
      iNode = int(data[0].replace(':',''))
      if nPrunedNodes > 0:
        iNode = node_to_node_dict[iNode]
        # Remap node index
      feature = -2
      threshold = 0
      child_left = -1
      child_right = -1
      value = float(data[1].replace('=',''))
    else:
      # Looks like:
      # 'i:[f{feature[i]}<{threshold[i]} yes={children_left[i]},no={children_right[i]}...'
      iNode = int(node.split(':')[0]) # index comes before ':'
      if nPrunedNodes > 0:
        iNode = node_to_node_dict[iNode]
        # Remap node index
      # split around 'feature<threshold'
      data = node.split('<')
      feature = feature_names[data[0].split('[')[-1]]
      threshold = float(data[1].split(']')[0])
      child_left = int(node.split('yes=')[1].split(',')[0])
      child_right = int(node.split('no=')[1].split(',')[0])
      if nPrunedNodes > 0:
        child_left = node_to_node_dict[child_left]
        child_right = node_to_node_dict[child_right]
        # Remap node index for children to preserve tree structure
      value = 0
    features[iNode] = feature
    thresholds[iNode] = threshold
    children_left[iNode] = child_left
    children_right[iNode] = child_right
    values[iNode] = value
  if nPrunedNodes > 0:
    del features[-nPrunedNodes:]
    del thresholds[-nPrunedNodes:] 
    del children_left[-nPrunedNodes:]
    del children_right[-nPrunedNodes:] 
    del values[-nPrunedNodes:]
    # Remove the last N unused nodes in the tree 
  treeDict = {'feature' : features, 'threshold' : thresholds, 'children_left' : children_left,
              'children_right' : children_right, 'value' : values}
  return treeDict
