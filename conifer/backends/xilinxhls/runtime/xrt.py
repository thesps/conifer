import logging
logger = logging.getLogger(__name__)
import numpy as np
from conifer_xrt_runtime import ConiferXilinxHLSXRTRuntime

class XrtDriver:

  def __init__(self, xclbin_path, kernel_name, device_index=0, batch_size=1):
    self.device = ConiferXilinxHLSXRTRuntime(device_index, xclbin_path, kernel_name)
    self.device.allocate_buffers(batch_size)

  def allocate_buffers(self, batch_size=1):
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