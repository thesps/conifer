import numpy as np
from .converter import addParentAndDepth, padTree
from ..model import model

def convert(bdt):
  ensembleDict = {'max_depth' : bdt.max_depth, 'n_trees' : bdt.n_estimators,
                  'n_features' : bdt.n_features_,
                  'n_classes' : bdt.n_classes_, 'trees' : [],
                  'init_predict' : bdt._raw_predict_init(np.zeros(bdt.n_features_).reshape(1, -1))[0].tolist(),
                  'norm' : 1}
  for trees in bdt.estimators_:
    treesl = []
    for tree in trees:
      tree = treeToDict(bdt, tree.tree_)
      tree = padTree(ensembleDict, tree)
      treesl.append(tree)
    ensembleDict['trees'].append(treesl)

  return ensembleDict
  #return model(ensembleDict)

def treeToDict(bdt, tree):
  # Extract the relevant tree parameters
  # NB node values are multiplied by the learning rate here, saving work in the FPGA
  treeDict = {'feature' : tree.feature.tolist(), 'threshold' : tree.threshold.tolist(), 'value' : (tree.value[:,0,0] * bdt.learning_rate).tolist()}
  treeDict['children_left'] = tree.children_left.tolist()
  treeDict['children_right'] = tree.children_right.tolist()
  treeDict = addParentAndDepth(treeDict)
  return treeDict

