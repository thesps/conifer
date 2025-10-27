import numpy as np
from conifer.converters import splitting_conventions

def convert_bdt(bdt):
  ensembleDict = {'max_depth' : bdt.max_depth, 'n_trees' : bdt.n_estimators,
                  'n_features' : bdt.n_features_in_,
                  'n_classes' : bdt.n_classes_, 'trees' : [],
                  'init_predict' : bdt._raw_predict_init(np.zeros(bdt.n_features_in_).reshape(1, -1))[0].tolist(),
                  'norm' : 1,
                  'library':'sklearn',
                  'splitting_convention': splitting_conventions['sklearn']}
  for trees in bdt.estimators_:
    treesl = []
    for tree in trees:
      tree = treeToDict(bdt, tree.tree_,bdt.n_features_in_)
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
                  'norm' : 1,
                  'library':'sklearn',
                  'splitting_convention': splitting_conventions['sklearn']}
  for tree in bdt.estimators_:
    treesl = []
    tree = treeToDict(bdt, tree.tree_,bdt.n_features_in_)
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

def treeToDict(bdt, tree, n_features):
  # Extract the relevant tree parameters
  treeDict = {'feature' : tree.feature.tolist(), 'threshold' : tree.threshold.tolist(), 'value' : tree.value.tolist()}
  weight_list = []
  for feature in tree.feature.tolist():
    weights = [0 for i in range(n_features)]
    if feature >= 0:
      weights[feature] = 1
    weight_list.append(weights)
  treeDict['weight'] = weight_list
  treeDict['children_left'] = tree.children_left.tolist()
  treeDict['children_right'] = tree.children_right.tolist()
  return treeDict

