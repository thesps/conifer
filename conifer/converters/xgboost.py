import numpy as np
import json
from .converter import addParentAndDepth, padTree
from ..model import model

def convert(bdt):
    meta = json.loads(bdt.save_config())
    max_depth = int(meta['learner']['gradient_booster']['updater']['grow_colmaker']['train_param']['max_depth'])
    n_classes = int(meta['learner']['learner_model_param']['num_class'])
    n_classes = 2 if n_classes == 0 else n_classes
    n_features = int(meta['learner']['learner_model_param']['num_feature'])
    ensembleDict = {'max_depth' : max_depth,
                    'n_trees' : len(bdt.get_dump()),
                    'n_classes' : n_classes,
                    'n_features' : n_features,
                    'trees' : [],
                    'init_predict' : 0,
                    'norm' : 1}
    for tree in bdt.get_dump():
        tree = treeToDict(bdt, tree)
        tree = addParentAndDepth(tree)
        tree = padTree(ensembleDict, tree)
        ensembleDict['trees'].append([tree])
    return ensembleDict

def treeToDict(bdt, tree):
  # First of all make the tree sklearn-like
  # split by newline, ignore the last line
  nodes = tree.split('\n')[:-1]
  # remove tab characters
  nodes = list(map(lambda x: x.replace('\t',''), nodes))
  nNodes = len(nodes)
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
      value = 0
    features[iNode] = feature
    thresholds[iNode] = threshold
    children_left[iNode] = child_left
    children_right[iNode] = child_right
    values[iNode] = value
  treeDict = {'feature' : features, 'threshold' : thresholds, 'children_left' : children_left,
              'children_right' : children_right, 'value' : values}
  return treeDict
