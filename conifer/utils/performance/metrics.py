'''
Module for obtaining extra model/tree metrics
'''
from conifer.model import ModelBase
import numpy as np

def array_summary_statistics(d):
  '''
  Get summary statistics from an array
  min, max, mean, std, quartiles
  '''
  mean, std, min, max, sum = np.mean(d), np.std(d), np.min(d), np.max(d), np.sum(d)
  quartiles = np.quantile(d, [0.25, 0.75])
  return {'mean' : mean,
          'std' : std,
          'min' : min,
          'max' : max,
          'sum' : sum,
          'quartile_1' : quartiles[0],
          'quartile_3' : quartiles[1]}

def __prefix_keys(d : dict, prefix : str):
  '''
  Add a prefix to the keys of dictionary d
  '''
  x = {}
  for k, v in d.items():
    x[f'{prefix}{k}'] = v
  return x

def get_sparsity_metrics(model):
  sparsity = np.array([1 - (tree.n_nodes() - tree.n_leaves()) / (2 ** model.max_depth - 1) for tree_c in model.trees for tree in tree_c])  
  return __prefix_keys(array_summary_statistics(sparsity), 'sparsity_')

def get_feature_frequency(model : ModelBase):
  def _get_feature_counts_tree(tree):
    features, counts = np.unique(tree.feature, return_counts=True)
    return features, counts
  counts = np.zeros(model.n_features)
  for trees_c in model.trees:
    for tree in trees_c:
      f, c = _get_feature_counts_tree(tree)
      counts[f[f != -2]] += c[f != -2]
  n_nodes = model.n_nodes() - model.n_leaves()
  return counts / n_nodes 
    
def get_feature_frequency_metrics(model : ModelBase):
  feature_frequency = get_feature_frequency(model)
  return __prefix_keys(array_summary_statistics(feature_frequency), 'feature_frequency_')

def get_model_metrics(model : ModelBase):
  results_d = {'max_depth' : model.max_depth,
               'n_trees' : model.n_trees,
               'n_features' : model.n_features,
               'n_nodes' : model.n_nodes() - model.n_leaves(),
               'n_leaves' : model.n_leaves(),
               'backend' : model.config.backend}
  results_d.update(get_sparsity_metrics(model))
  results_d.update(get_feature_frequency_metrics(model))
  return results_d
