import json
import xgboost as xgb
import pandas
from scipy.special import logit
import numpy as np
from packaging import version
from typing import Union
import logging
from conifer.converters import splitting_conventions
logger = logging.getLogger(__name__)
__xgb_version = version.parse(xgb.__version__)

def convert(bdt : Union[xgb.core.Booster, xgb.XGBClassifier, xgb.XGBRegressor]):
    assert isinstance(bdt, (xgb.core.Booster, xgb.XGBClassifier, xgb.XGBRegressor))
    if isinstance(bdt, xgb.core.Booster):
      bst = bdt
    elif isinstance(bdt, (xgb.XGBClassifier, xgb.XGBRegressor)):
      bst = bdt.get_booster()
    meta = json.loads(bst.save_config())
    if __xgb_version >= version.parse('2'):
      logger.warning(f'Some prediction disagreements are observed for xgboost versions >= 2.0.0. You have xgboost {__xgb_version}. Extra validation is advised.')
      max_depth = int(meta.get('learner').get('gradient_booster').get('tree_train_param').get('max_depth'))
    else:
      updater = meta.get('learner').get('gradient_booster').get('gbtree_train_param').get('updater').split(',')[0]
      max_depth = int(meta.get('learner').get('gradient_booster').get('updater').get(updater).get('train_param').get('max_depth'))
    n_classes = int(meta.get('learner').get('learner_model_param').get('num_class'))
    fn_classes = 1 if n_classes == 0 else n_classes # the number of learners
    n_classes = 2 if n_classes == 0 else n_classes # the actual number of classes
    n_trees = int(int(meta.get('learner').get('gradient_booster').get('gbtree_model_param').get('num_trees')) / fn_classes)
    n_features = int(meta.get('learner').get('learner_model_param').get('num_feature'))
    
    feature_names = {}
    if bst.feature_names is None:
      for i in range(n_features):
        feature_names[f'f{i}'] = i
    else:
      for i, feature_name in enumerate(bst.feature_names):
        feature_names[feature_name] = i
    
    ensembleDict = {'max_depth' : max_depth,
                    'n_trees' : n_trees,
                    'n_classes' : n_classes,
                    'n_features' : n_features,
                    'trees' : [],
                    'init_predict' : parse_base_score(meta, fn_classes),
                    'norm' : 1,
                    'library':'xgboost',
                    'splitting_convention': splitting_conventions['xgboost'],
                    'feature_map' : feature_names}
    
    trees = bst.trees_to_dataframe()
    for i in range(ensembleDict['n_trees']):
        treesl = []
        for j in range(fn_classes):
            tree = trees[trees.Tree == fn_classes * i + j]
            tree = treeToDict(tree, feature_names)
            treesl.append(tree)
        ensembleDict['trees'].append(treesl)
    return ensembleDict

def treeToDict(tree : pandas.DataFrame, feature_names):
  assert isinstance(tree, pandas.DataFrame), "This method expects the tree as a pandas DataFrame"
  thresholds = tree.Split.fillna(0).tolist()
  features = tree.Feature.map(lambda x : -2 if x == 'Leaf' else feature_names[x]).tolist()
  children_left = tree.Yes.map(lambda x : int(x.split('-')[1]) if isinstance(x, str) else -1).tolist()
  children_right = tree.No.map(lambda x : int(x.split('-')[1]) if isinstance(x, str) else -1).tolist()
  values = tree.Gain.tolist()
  
  weight_list = []
  for feature in features:
    weights = [0 for i in range(len(feature_names))]
    if feature >= 0:
      weights[feature] = 1
    weight_list.append(weights)
    
  treeDict = {'feature'        : features,
              'threshold'      : thresholds,
              'weight'         : weight_list,
              'children_left'  : children_left,
              'children_right' : children_right,
              'value'          : values
              }
  return treeDict

def parse_base_score(meta_json, n_classes):
  '''
  Extract the base score from the learner
  '''
  if __xgb_version >= version.parse('2'):

    objective = meta_json.get('learner').get('objective').get('name')
    _proba_objectives = ['reg:logistic', 'binary:logistic', 'multi:softmax', 'multi:softprob']
    base_score_string = meta_json.get('learner').get('learner_model_param').get('base_score')
    base_score_string_values = base_score_string.strip('[]').split(',')
    base_score_float = list(map(float, base_score_string_values))
    all_zeros = not np.any(base_score_float)
    if not all_zeros:
      # for objectives predicting probabilities, take the logit to get a raw base score
      # otherwise the base score is already a raw value (e.g. regression)
      if objective in _proba_objectives:
        base_score = logit(base_score_float).tolist()
      else:
        base_score = base_score_float
      assert len(base_score) == n_classes, f'Expected {n_classes} base_scores values, got {len(base_score)}'
      return base_score
    else:
      return [0.] * n_classes
  else:
    return [0.] * n_classes