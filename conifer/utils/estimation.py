import numpy as np
from conifer.model import ModelBase
import logging
logger = logging.getLogger(__name__)

# constants derived from fitting to a scan
_vhdl_latency_constants = np.array([2, 1, 1])
_vhdl_lut_constants = np.array([-54.69561067, 8.03061793, 13.99964128])
_vhdl_ff_constants = np.array([12.77986779, 33.56940453, 4.01345795])

def vhdl_latency(n_trees: int, n_nodes: int, max_depth: int, k: "list[float]" = _vhdl_latency_constants):
    k0, k1, k2 = k
    return int(np.round(k0 + k1 * max_depth + k2 * np.ceil(np.log2(n_trees))))

def _vhdl_resources(n_trees: int, n_nodes: int, max_depth: int, k: "list[float]"):
    k0, k1, k2 = k
    return int(np.round(k0 + k1 * n_trees + k2 * n_nodes))

def vhdl_luts(n_trees: int, n_nodes: int, max_depth: int, k: "list[float]" = _vhdl_lut_constants):
  return _vhdl_resources(n_trees, n_nodes, max_depth, k)

def vhdl_ffs(n_trees: int, n_nodes: int, max_depth: int, k: "list[float]" = _vhdl_ff_constants):
  return _vhdl_resources(n_trees, n_nodes, max_depth, k)

# constants derived from fitting to a scan
_hls_latency_constants = np.array([-6.28150311, 1.3875, 0.72214752])
_hls_ff_constants = np.array([50.06785475, 6.93775665, 0.69295728])
_hls_lut_constants = np.array([-21.3043842, 21.60668614, 7.05345757])

def hls_latency(n_trees: int, n_nodes: int, max_depth: int, k: "list[float]" = _hls_latency_constants):
   k0, k1, k2 = k
   t = int(np.ceil(k0 + k1 * max_depth + k2 * np.log2(n_trees)))
   t = t if t > 0 else 1
   return t

def _hls_resources(n_trees: int, n_nodes: int, max_depth: int, k: "list[float]"):
   k0, k1, k2 = k
   r = int(np.round(k0 + k1 * n_trees + k2 * n_nodes))
   r = r if r > 0 else 0
   return r

def hls_luts(n_trees: int, n_nodes: int, max_depth: int, k: "list[float]" = _hls_lut_constants):
   return _hls_resources(n_trees, max_depth, n_nodes, k)

def hls_ffs(n_trees: int, n_nodes: int, max_depth: int, k: "list[float]" = _hls_ff_constants):
   return _hls_resources(n_trees, max_depth, n_nodes, k)

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
    results = {param : _estimators[backend][param](model.n_trees, model.n_nodes() - model.n_leaves(), model.max_depth) for param in _estimators[backend].keys()}
  else:
    logger.warn(f'Performance estimates are not available for {backend} backend')
    results = {}
  return results   