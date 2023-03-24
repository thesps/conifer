import numpy as np
from conifer.model import ModelBase
import logging
logger = logging.getLogger(__name__)

# constants derived from fitting to a scan
_vhdl_latency_constants = np.array([1, 1, 1])
_vhdl_lut_constants = np.array([-22.68347473,   5.83935035,  -1.03714058,   2.39507861])
_vhdl_ff_constants = np.array([14.32104831,  2.20940089, 13.31831411,  1.81902467])

def vhdl_latency(n_trees: int, max_depth: int, k: "list[float]" = _vhdl_latency_constants):
    k0, k1, k2 = k
    return int(np.round(k0 + k1 * max_depth + k2 * np.ceil(np.log2(n_trees))))

def _vhdl_resources(n_trees: int, max_depth: int, k: "list[float]"):
    k0, k1, k2, k3 = k
    return int(np.round(k0 + k1 * n_trees * (k2 + k3 * 2 ** max_depth)))

def vhdl_luts(n_trees: int, max_depth: int, k: "list[float]" = _vhdl_lut_constants):
  return _vhdl_resources(n_trees, max_depth, k)

def vhdl_ffs(n_trees: int, max_depth: int, k: "list[float]" = _vhdl_ff_constants):
  return _vhdl_resources(n_trees, max_depth, k)

_estimators = {
   'vhdl' : {
    'latency' : vhdl_latency,
    'lut'     : vhdl_luts,
    'ff'      : vhdl_ffs
   }
}

def performance_estimates(model : ModelBase):
  '''
  Get estimates of latency and resource usage for a conifer model
  Parameters:
  ----------
  model : conifer model 

  Returns:
  ----------
  dictionary of parameter estimates
  '''
  logger.warn('Performance estimation is experimental, use with caution!')
  backend = model.config.backend
  if backend in _estimators.keys():
    results = {param : _estimators[backend][param](model.n_trees, model.max_depth) for param in _estimators[backend].keys()}
  else:
    logger.warn(f'Performance estimates are not available for {backend} backend')
    results = {}
  return results   