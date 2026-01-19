import numpy as np
from conifer_xrt_runtime import ConiferFPUXRTRuntime, ConiferModelShapeInfo
from conifer.backends.fpu.writer import FPUConfig
import logging
logger = logging.getLogger(__name__)

class XrtDriver:

  def __init__(self, xclbin_path, kernel_name='FPU_Alveo', device_index=0):
    self.device = ConiferFPUXRTRuntime(device_index, xclbin_path, kernel_name)
    fpu_info = self.device.fpu_info
    config_dictionary = {
        'features': fpu_info.features,
        'tree_engines': fpu_info.tree_engines,
        'nodes': fpu_info.nodes,
        'threshold_type' : fpu_info.threshold_type,
        'score_type' : fpu_info.score_type,
        'dynamic_scaler': fpu_info.dynamic_scaler,
    }
    self.config = FPUConfig(config_dictionary)
    logger.info(f'Initialized FPU driver for {kernel_name} with configuration: {self.config}')

  def load(self, nodes: np.ndarray, scales: np.ndarray, n_features=1, n_classes=2, batch_size=1):
    '''
    Load packed model onto FPU

    Parameters
    ----------
    nodes: ndarray of shape (FPU TEs, FPU nodes, 7), dtype int32
      Packed nodes, from FPUModel.pack
    scales: ndarray of shape (FPU features + 1), dtype float32
      Packed scale factors, from FPUModel._scales
    n_features: integer (optional)
      Number of model features (must be less than FPU features)
    n_classes: integer (optional)
      Number of model classes (Only binary classification is currently supported)
    batch_size: integer (optional)
      Batch size for allocating buffers
    '''
    assert n_classes == 2, "Only binary classification is currently supported"
    model_shape_info = ConiferModelShapeInfo(n_features, n_classes)
    self.device.load(nodes, scales, batch_size, model_shape_info)
    logger.info(f'Model with {n_features} features and {n_classes} classes loaded onto FPU')

  def allocate_buffers(self, batch_size=1):
    '''
    Allocate input/output buffers on device.
    A model must be loaded before calling this.
    
    batch_size: integer
      Batch size for allocating buffers
    '''
    self.device.allocate_buffers(batch_size)

  def decision_function(self, X: np.ndarray) -> np.ndarray:
    '''
    Execute inference

    Parameters
    ----------
    X: ndarray of shape (batch_size, n_features), dtype float32
      Input sample. Shape must match allocated buffers

    Returns
    ----------
    score: ndarray of shape (batch_size, n_classes)
    '''
    return self.device.decision_function(X)