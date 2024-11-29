import numpy as np
import conifer
from conifer.model import ModelBase
import os
import logging
logger = logging.getLogger(__name__)

# constants derived from fitting to a scan
_vhdl_latency_constants = np.array([2, 1, 1])
_vhdl_lut_constants = np.array([-54.69561067, 8.03061793, 13.99964128])
_vhdl_ff_constants = np.array([12.77986779, 33.56940453, 4.01345795])

def vhdl_latency(model : ModelBase, k: "list[float]" = _vhdl_latency_constants):
  assert isinstance(model, ModelBase), f"Expected conifer.model.ModelBase, got {type(model)}"
  k0, k1, k2 = k
  return int(np.round(k0 + k1 * model.max_depth + k2 * np.ceil(np.log2(model.n_trees))))

def _vhdl_resources(model : ModelBase, k: "list[float]"):
  assert isinstance(model, ModelBase), f"Expected conifer.model.ModelBase, got {type(model)}"
  k0, k1, k2 = k
  return int(np.round(k0 + k1 * model.n_trees + k2 * model.n_nodes()))

def vhdl_luts(model : ModelBase, k: "list[float]" = _vhdl_lut_constants):
  return _vhdl_resources(model, k)

def vhdl_ffs(model : ModelBase, k: "list[float]" = _vhdl_ff_constants):
  return _vhdl_resources(model, k)

hls_latency_model = None
hls_lut_model = None
hls_ff_model = None
hls_estimators = {'latency' : hls_latency_model, 'lut' : hls_lut_model, 'ff' : hls_ff_model}
hls_estimator_files = {'latency' : f'{os.path.dirname(__file__)}/performance_models/hls_latency_model.json',
                       'lut'     : f'{os.path.dirname(__file__)}/performance_models/hls_latency_model.json', # TODO provide the proper model
                       'ff'      : f'{os.path.dirname(__file__)}/performance_models/hls_latency_model.json'  # TODO provide the proper model
                       }

def _get_hls_estimator(name : str):
  if hls_estimators[name] is None:
    hls_estimators[name] = conifer.model.load_model(hls_estimator_files[name])
  return hls_estimators[name]

def _estimate_hls_performance(model : ModelBase, estimator : ModelBase):
  assert isinstance(model, ModelBase), f"Expected conifer.model.ModelBase, got {type(model)}"
  X = conifer.utils.performance.metrics.get_model_metrics(model)
  features = ['max_depth',
              'n_trees',
              'n_features',
              'n_nodes',
              'n_leaves',
              'sparsity_mean',
              'sparsity_std',
              'sparsity_min',
              'sparsity_max',
              'sparsity_sum',
              'sparsity_quartile_1',
              'sparsity_quartile_3',
              'feature_frequency_mean',
              'feature_frequency_std',
              'feature_frequency_min',
              'feature_frequency_max',
              'feature_frequency_sum',
              'feature_frequency_quartile_1',
              'feature_frequency_quartile_3']
  X = np.array([[X[k] for k in features]])
  return estimator.decision_function(X)

def hls_latency(model : ModelBase):
  return _estimate_hls_performance(model, _get_hls_estimator('latency'))

def hls_luts(model : ModelBase):
  return _estimate_hls_performance(model, _get_hls_estimator('lut'))


def hls_ffs(model : ModelBase):
  return _estimate_hls_performance(model, _get_hls_estimator('ff'))

_estimators = {
  'vhdl' : {
    'latency' : vhdl_latency,
    'lut'     : vhdl_luts,
    'ff'      : vhdl_ffs
  },
  'xilinxhls' : {
    'latency' : hls_latency,
    'lut'     : hls_luts,
    'ff'      : hls_ffs
  }
}

_estimators['vivadohls'] = _estimators['xilinxhls']
_estimators['vitishls'] = _estimators['xilinxhls']

def performance_estimates(model : ModelBase, backend : str = None):
  '''
  Get estimates of latency and resource usage for a conifer model
  Parameters:
  ----------
  model : conifer.model.ModelBase
    The model to provide performance estimates for

  backend : string (optional)
    Get estimates for a backend other than the provided model's backend

  Returns:
  ----------
  dictionary of parameter estimates
  '''
  logger.warning('Performance estimation is experimental, use with caution!')
  if backend is None:
    backend = model.config.backend
  if backend in _estimators.keys():
    results = {param : _estimators[backend][param](model) for param in _estimators[backend].keys()}
  else:
    logger.warn(f'Performance estimates are not available for {backend} backend')
    results = {}
  return results