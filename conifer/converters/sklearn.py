import numpy as np

def convert_bdt(bdt):
  ensembleDict = {'max_depth' : bdt.max_depth, 'n_trees' : bdt.n_estimators,
                  'n_features' : bdt.n_features_in_,
                  'n_classes' : bdt.n_classes_, 'trees' : [],
                  'init_predict' : bdt._raw_predict_init(np.zeros(bdt.n_features_in_).reshape(1, -1))[0].tolist(),
                  'norm' : 1}
  for trees in bdt.estimators_:
    treesl = []
    for tree in trees:
      tree = treeToDict(bdt, tree.tree_)
      # NB node values are multiplied by the learning rate here, saving work in the FPGA
      tree['value'] = (np.array(tree['value'])[:,0,0] * bdt.learning_rate).tolist()
      treesl.append(tree)
    ensembleDict['trees'].append(treesl)

  return ensembleDict

def convert_random_forest(bdt):
  ensembleDict = {'max_depth' : bdt.max_depth, 'n_trees' : bdt.n_estimators,
                  'n_features' : bdt.n_features_in_,
                  'n_classes' : bdt.n_classes_, 'trees' : [],
                  'init_predict' : [0] * bdt.n_classes_, 
                  'norm' : 1}
  for tree in bdt.estimators_:
    treesl = []
    tree = treeToDict(bdt, tree.tree_)
    # Random forest takes the mean prediction, do that here
    # Also need to scale the values by their sum
    v = np.array(tree['value'])
    tree['value'] = (v / v.sum(axis=2)[:, np.newaxis] / bdt.n_estimators)[:,0,0].tolist()
    treesl.append(tree)
    ensembleDict['trees'].append(treesl)

  return ensembleDict

def convert(bdt):
    if 'GradientBoosting' in bdt.__class__.__name__:
        return convert_bdt(bdt)
    elif 'RandomForest' in bdt.__class__.__name__:
        return convert_random_forest(bdt)

def treeToDict(bdt, tree):
  # Extract the relevant tree parameters
  treeDict = {'feature' : tree.feature.tolist(), 'threshold' : tree.threshold.tolist(), 'value' : tree.value.tolist()}
  treeDict['children_left'] = tree.children_left.tolist()
  treeDict['children_right'] = tree.children_right.tolist()
  return treeDict

