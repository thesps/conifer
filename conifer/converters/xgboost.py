import numpy as np
import json
from .converter import addParentAndDepth, padTree
from ..model import model

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
    trees = bdt.get_dump()
    for i in range(ensembleDict['n_trees']):
        treesl = []
        for j in range(fn_classes):
            tree = trees[fn_classes * i + j]
            tree = treeToDict(bdt, tree)
            tree = addParentAndDepth(tree)
            tree = padTree(ensembleDict, tree)
            treesl.append(tree)
        ensembleDict['trees'].append(treesl)
    return ensembleDict

def treeToDict(bdt, tree):
  # First of all make the tree sklearn-like
  # split by newline, ignore the last line
  nodes = tree.split('\n')[:-1]
  # remove tab characters
  nodes = list(map(lambda x: x.replace('\t',''), nodes))

  tmp_nNodes = len(nodes)
  actual_nodes = []
  for i in range(tmp_nNodes):
    iNode = int(nodes[i].split(':')[0])
    actual_nodes.append(iNode)

  nNodes = max(actual_nodes)+1
  expected_nodes = [i for i in range(nNodes)]
  pruned_nodes = [i for i in expected_nodes + actual_nodes if i not in expected_nodes or i not in actual_nodes] 

  if len(pruned_nodes) > 0:
    #print("Length of nodes: ",tmp_nNodes)
    #print("Actual Nodes: ",actual_nodes)
    #print("Expected Nodes: ",expected_nodes)
    #print("Pruned Nodes: ",pruned_nodes)
    iNode_shift = max(expected_nodes)-max(pruned_nodes)
    min_pruned_nodes = min(pruned_nodes)
  else:
    min_pruned_nodes = max(expected_nodes)


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
      if iNode > min_pruned_nodes:
        iNode -= iNode_shift

      feature = -2
      threshold = 0
      child_left = -1
      child_right = -1
      value = float(data[1].replace('=',''))
    else:
      # Looks like:
      # 'i:[f{feature[i]}<{threshold[i]} yes={children_left[i]},no={children_right[i]}...'
      iNode = int(node.split(':')[0]) # index comes before ':'
      # split around 'feature<threshold'
      data = node.split('<')
      feature = int(data[0].split('[')[-1].replace('f',''))
      threshold = float(data[1].split(']')[0])
      child_left = int(node.split('yes=')[1].split(',')[0])
      child_right = int(node.split('no=')[1].split(',')[0])
      if child_left > min_pruned_nodes:
        child_left -= iNode_shift
      if child_right > min_pruned_nodes:
        child_right -= iNode_shift
      value = 0
    features[iNode] = feature
    thresholds[iNode] = threshold
    children_left[iNode] = child_left
    children_right[iNode] = child_right
    values[iNode] = value

  if len(pruned_nodes) > 0:
    for iNode in pruned_nodes:
      del features[iNode]
      del thresholds[iNode] 
      del children_left[iNode]
      del children_right[iNode] 
      del values[iNode]

    
  treeDict = {'feature' : features, 'threshold' : thresholds, 'children_left' : children_left,
              'children_right' : children_right, 'value' : values}
  return treeDict
