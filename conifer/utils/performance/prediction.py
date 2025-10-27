import numpy as np
import conifer
from conifer.model import ModelBase
import os
import logging
logger = logging.getLogger(__name__)

class PerformanceEstimator:
  def __init__(self):
    pass
  def predict(self, model : ModelBase):
    raise NotImplementedError
  
class VHDLLatencyEstimator(PerformanceEstimator):
  '''
  Estimator of VHDL backend latency
  Model is derived from prior knowledge of implementation
  '''
  latency_constants = np.array([2, 1, 1])
  def __init__(self):
    super().__init__()

  def predict(self, model : ModelBase):
    assert isinstance(model, ModelBase), f"Expected conifer.model.ModelBase, got {type(model)}"
    k0, k1, k2 = VHDLLatencyEstimator.latency_constants
    return int(np.round(k0 + k1 * model.max_depth + k2 * np.ceil(np.log2(model.n_trees))))
  
class VHDLLUTEstimator(PerformanceEstimator):
  '''
  Estimator of VHDL backend LUT usage
  Model is derived from prior knowledge of implementation and functional fit to data
  '''
  lut_constants = np.array([-54.69561067, 8.03061793, 13.99964128])
  def __init__(self):
    super().__init__()

  def predict(self, model : ModelBase):
    assert isinstance(model, ModelBase), f"Expected conifer.model.ModelBase, got {type(model)}"
    k0, k1, k2 = VHDLLUTEstimator.lut_constants
    return int(np.round(k0 + k1 * model.n_trees + k2 * model.n_nodes()))

class VHDLFFEstimator(PerformanceEstimator):
  '''
  Estimator of VHDL backend FF usage
  Model is derived from prior knowledge of implementation and functional fit to data
  '''
  ff_constants = np.array([12.77986779, 33.56940453, 4.01345795])
  def __init__(self):
    super().__init__()

  def predict(self, model : ModelBase):
    assert isinstance(model, ModelBase), f"Expected conifer.model.ModelBase, got {type(model)}"
    k0, k1, k2 = VHDLFFEstimator.ff_constants
    return int(np.round(k0 + k1 * model.n_trees + k2 * model.n_nodes()))

class HLSEstimator(PerformanceEstimator):
  '''
  HLS backend estimator base class
  '''
  _model_file = None
  def __init__(self):
    self.model = None
    super(HLSEstimator, self).__init__()

  def _get_model(self):
    '''
    Get the estimator model object
    Load the model if not already loaded
    '''
    if self.model is None:
      self.model = conifer.model.load_model(self._model_file)
    return self.model
  
  def predict(self, model : ModelBase):
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
    estimator = self._get_model()
    return int(np.round(estimator.decision_function(X)))
  
class HLSLatencyEstimator(HLSEstimator):
  '''
  Estimator of HLS backend Latency
  Model is derived from BDT fit to data
  '''
  _model_file = f'{os.path.dirname(__file__)}/performance_models/hls_latency_model.json'

  def __init__(self):
    super(HLSLatencyEstimator, self).__init__()

class HLSLUTEstimator(HLSEstimator):
  '''
  Estimator of HLS backend LUT usage
  Model is derived from BDT fit to data
  '''
  _model_file = f'{os.path.dirname(__file__)}/performance_models/hls_lut_model.json'

  def __init__(self):
    super(HLSLUTEstimator, self).__init__()    

class HLSFFEstimator(HLSEstimator):
  '''
  Estimator of HLS backend FF usage
  Model is derived from BDT fit to data
  '''
  _model_file = f'{os.path.dirname(__file__)}/performance_models/hls_ff_model.json'

  def __init__(self):
    super(HLSFFEstimator, self).__init__()

# make instances of all of the estimators
vhdlLatencyEstimator = VHDLLatencyEstimator()
vhdlLUTEstimator = VHDLLUTEstimator()
vhdlFFEstimator = VHDLFFEstimator()

hlsLatencyEstimator = HLSLatencyEstimator()
hlsLUTEstimator = HLSLUTEstimator()
hlsFFEstimator = HLSFFEstimator()

_estimators = {
  'vhdl' : {
    'latency' : vhdlLatencyEstimator,
    'lut'     : vhdlLUTEstimator,
    'ff'      : vhdlFFEstimator
  },
  'xilinxhls' : {
    'latency' : hlsLatencyEstimator,
    'lut'     : hlsLUTEstimator,
    'ff'      : hlsFFEstimator
  }
}

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
  if model.is_oblique():
    logger.warning('Performance estimation is not yet available for oblique splitting')
  if backend is None:
    backend = model.config.backend
  if backend in _estimators.keys():
    results = {param : _estimators[backend][param].predict(model) for param in _estimators[backend].keys()}
  else:
    logger.warning(f'Performance estimates are not available for {backend} backend. Backends with estimates available are: {str(list(_estimators.keys()))}')
    results = {}
  return results