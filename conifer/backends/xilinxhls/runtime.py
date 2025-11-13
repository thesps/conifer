import logging
logger = logging.getLogger(__name__)
from typing import Union
import pynq
import pyxrt
import numpy as np

class ZynqDriver:
  def __init__(self, bitfile, ip_name=None, batch_size=1):
    self.overlay = pynq.Overlay(bitfile)
    ips = [at for at in dir(self.overlay) if 'conifer' in at]
    if ip_name is None:
      assert len(ips) > 0, 'No conifer IPs found in overlay'
    ip_name = ip_name if ip_name is not None else ips[0]
    self.ip = getattr(self.overlay, ip_name, None)
    assert self.ip is not None, 'No conifer IPs found in overlay'
    self.get_info()
    self._init_buffers(batch_size)

  def get_info(self):
    self.ip.write(self.ip.register_map.N.address, 0)
    self.ip.write(self.ip.register_map.CTRL.AP_START, 1)
    n_f = self.ip.read(self.ip.register_map.n_f.address)
    n_c = self.ip.read(self.ip.register_map.n_c.address)
    self.n_features = n_f
    self.n_classes = n_c

  def _init_Xy_buffers(self, X_shape, y_shape, x_dtype='float32', y_dtype='float32'):
    self.Xbuf = pynq.allocate(X_shape, dtype=x_dtype)
    self.ybuf = pynq.allocate(y_shape, dtype=y_dtype)
    self.ip.write(self.ip.register_map.x.address, self.Xbuf.physical_address)
    self.ip.write(self.ip.register_map.score.address, self.ybuf.physical_address)

  def _init_buffers(self, batch_size=1, x_dtype='float32', y_dtype='float32'):
    X_shape = (batch_size, self.n_features)
    y_shape = (batch_size, self.n_classes)
    self._init_Xy_buffers(X_shape=X_shape, y_shape=y_shape, x_dtype=x_dtype, y_dtype=y_dtype)

  def decision_function(self, X):
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
    assert X.ndim == 2, "Expected 2D inputs."
    assert X.shape[0] == self.Xbuf.shape[0], "Batch size must match"
    assert X.shape[1] == self.Xbuf.shape[1], "More inputs were provided than this FPU supports ({} vs {})".format(X.shape[1], self.config['features'])
    assert X.dtype == self.Xbuf.dtype, f"Cannot copy {X.dtype} data into {self.Xbuf.dtype} buffer"
    self.Xbuf[:] = X
    self.ip.write(self.ip.register_map.N.address, X.shape[0])
    self.ip.write(self.ip.register_map.CTRL.AP_START, 1)
    while not self.ip.register_map.CTRL.AP_IDLE:
      pass
    return self.ybuf
  
class PynqAlveoDriver:
  def __init__(self, bitfile, ip_name=None, batch_size=1):
    self.overlay = pynq.Overlay(bitfile)
    ips = [at for at in dir(self.overlay) if 'conifer' in at]
    if ip_name is None:
      assert len(ips) > 0, 'No conifer IPs found in overlay'
    ip_name = ip_name if ip_name is not None else ips[0]
    self.ip = getattr(self.overlay, ip_name, None)
    assert self.ip is not None, 'No conifer IPs found in overlay'
    self.tmp_buff = pynq.allocate(1)
    self.get_info()
    self._init_buffers(batch_size)

  def get_info(self):
    tmp = self.tmp_buff
    self.ip.call(0, tmp, tmp, tmp, tmp)
    n_f = self.ip.read(self.ip.register_map.n_f.address)
    n_c = self.ip.read(self.ip.register_map.n_c.address)
    self.n_features = n_f
    self.n_classes = n_c
    tmp.freebuffer()

  def _init_Xy_buffers(self, X_shape, y_shape, x_dtype='float32', y_dtype='float32'):
    self.Xbuf = pynq.allocate(X_shape, dtype=x_dtype)
    self.ybuf = pynq.allocate(y_shape, dtype=y_dtype)

  def _init_buffers(self, batch_size=1, x_dtype='float32', y_dtype='float32'):
    X_shape = (batch_size, self.n_features)
    y_shape = (batch_size, self.n_classes)
    self._init_Xy_buffers(X_shape=X_shape, y_shape=y_shape, x_dtype=x_dtype, y_dtype=y_dtype)

  def decision_function(self, X):
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
    assert X.ndim == 2, "Expected 2D inputs."
    assert X.shape[0] == self.Xbuf.shape[0], "Batch size must match"
    assert X.shape[1] == self.Xbuf.shape[1], "More inputs were provided than this FPU supports ({} vs {})".format(X.shape[1], self.config['features'])
    assert X.dtype == self.Xbuf.dtype, f"Cannot copy {X.dtype} data into {self.Xbuf.dtype} buffer"
    self.Xbuf[:] = X
    self.Xbuf.sync_to_device()
    self.ip.call(X.shape[0], self.tmp_buff, self.tmp_buff, self.Xbuf, self.ybuf)
    while not self.ip.register_map.CTRL.AP_IDLE:
      pass
    self.ybuf.sync_from_device()
    return self.ybuf

class PyXrtAlveoDriver:
  def __init__(self,
               xclbin : Union[str, pyxrt.xclbin],
               device : Union[int, pyxrt.device],
               ip_name=None,
               n_features=None,
               n_classes=None,
               batch_size=1):
    
    # load the device
    if isinstance(device, int):
      self.device = pyxrt.device(device)
    elif isinstance(device, pyxrt.device):
      self.device = device
    else:
      logger.error(f"Can't handle device '{device}' (type {type(device)})")

    # load the xclbin
    if isinstance(xclbin, str):
      self.xclbin = pyxrt.xclbin(xclbin)
    elif isinstance(xclbin, pyxrt.xclbin):
      self.xclbin = xclbin
    else:
      logger.error(f"Can't handle xclbin '{xclbin}' (type {type(xclbin)})")

    logger.debug('Loading xclbin onto device')
    uuid = self.device.load_xclbin(xclbin)

    kernels = self.xclbin.get_kernels()
    logger.debug(f'Kernels in xclbin: {kernels}')
    assert len(kernels) == 1, f"Can only handle 1 kernel in xclbin, found {len(kernels)}"
    kernel = kernels[0]

    self.kernel = pyxrt.kernel(self.device, uuid, kernel.get_name(), pyxrt.kernel.cu_access_mode.exclusive)

    # TODO
    # Until register reading is added, need to specify the n_features and n_classes
    self.n_features = n_features
    self.n_classes = n_classes
    self._init_buffers(batch_size)

  def _init_Xy_buffers(self, X_shape, y_shape, x_dtype='float32', y_dtype='float32'):
    Xsize = np.prod(X_shape) * np.dtype(x_dtype).itemsize
    ysize = np.prod(y_shape) * np.dtype(y_dtype).itemsize
    self.XbufHandle = pyxrt.bo(self.device, Xsize, pyxrt.bo.normal, self.kernel.group_id(3))
    self.ybufHandle = pyxrt.bo(self.device, ysize, pyxrt.bo.normal, self.kernel.group_id(3))
    self.Xbuf = self.XbufHandle.map()
    self.ybuf = self.ybufHandle.map()
    self.X_dtype = x_dtype
    self.y_dtype = y_dtype

  def _init_buffers(self, batch_size=1, x_dtype='float32', y_dtype='float32'):
    X_shape = (batch_size, self.n_features)
    y_shape = (batch_size, self.n_classes)
    self._init_Xy_buffers(X_shape=X_shape, y_shape=y_shape, x_dtype=x_dtype, y_dtype=y_dtype)

  def decision_function(self, X):
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
    assert X.ndim == 2, "Expected 2D inputs."
    assert np.prod(X.shape) * X.dtype.itemsize == self.Xbuf.nbytes, f"Cannot copy {np.prod(X.shape) * X.dtype.itemsize} bytes data into {self.Xbuf.nbytes} buffer"
    assert X.dtype == np.dtype(self.X_dtype), f"Got X dtype {X.dtype}, expected {self.X_dtype}"
    return self._decision_function(X)
  
  def _decision_function(self, X):
    '''
    Execute inference without checks on X shape and dtype

    Parameters
    ----------
    X: ndarray

    Returns
    ----------
    score: ndarray
    '''
    self.XbufHandle.write(X.tobytes(), 0)
    self.XbufHandle.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    run = self.kernel(X.shape[0], 0, 0, self.XbufHandle, self.ybufHandle)
    run.wait()
    self.ybufHandle.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
    return np.frombuffer(self.ybuf, dtype=self.y_dtype)