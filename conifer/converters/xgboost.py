import json
import xgboost as xgb
import pandas
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
      logger.warning(f'xgboost versions >= 2.0.0 are not yet fully supported. You have xgboost {__xgb_version}')
      max_depth = int(meta.get('learner').get('gradient_booster').get('tree_train_param').get('max_depth'))
    else:
      updater = meta.get('learner').get('gradient_booster').get('gbtree_train_param').get('updater').split(',')[0]
      max_depth = int(meta.get('learner').get('gradient_booster').get('updater').get(updater).get('train_param').get('max_depth'))
    n_classes = int(meta.get('learner').get('learner_model_param').get('num_class'))
    fn_classes = 1 if n_classes == 0 else n_classes # the number of learners
    n_classes = 2 if n_classes == 0 else n_classes # the actual number of classes
    n_trees = int(int(meta.get('learner').get('gradient_booster').get('gbtree_model_param').get('num_trees')) / fn_classes)
    n_features = int(meta.get('learner').get('learner_model_param').get('num_feature'))
    ensembleDict = {'max_depth' : max_depth,
                    'n_trees' : n_trees,
                    'n_classes' : n_classes,
                    'n_features' : n_features,
                    'trees' : [],
                    'init_predict' : [0] * fn_classes,
                    'norm' : 1,
                    'library':'xgboost',
                    'splitting_convention': splitting_conventions['xgboost']}
    
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
                    'init_predict' : [0] * fn_classes,
                    'norm' : 1,
                    'feature_map' : feature_names
                    }

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
  treeDict = {'feature'        : features,
              'threshold'      : thresholds,
              'children_left'  : children_left,
              'children_right' : children_right,
              'value'          : values
              }
  return treeDict
