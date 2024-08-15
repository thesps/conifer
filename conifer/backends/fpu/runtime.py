import pynq
import numpy as np
import json

class ZynqDriver:
  def __init__(self, bitfile, fpu_name=None, batch_size=1):
    self.overlay = pynq.Overlay(bitfile)
    self.fpus = [at for at in dir(self.overlay) if 'FPU' in at]
    if fpu_name is None:
      assert len(self.fpus) > 0, "No FPUs found in bitfile"
    fpu_name = self.fpus[0] if fpu_name is None else fpu_name
    self.fpu = getattr(self.overlay, fpu_name, None)
    assert self.fpu is not None, f'No FPU {fpu_name} found in bitfle'
    info = json.loads(self.get_info())
    self.config = info['configuration']
    self.metadata = info['metadata']

    self.interfaceNodes = pynq.buffer.allocate((self.config['tree_engines'], self.config['nodes'], 7), dtype='int32')
    self.scales = pynq.buffer.allocate(self.config['features']+1, dtype='float')
    self._init_buffers(batch_size=batch_size)

  def get_info_len(self):
    self.fpu.write(self.fpu.register_map.instruction.address, 0)
    self.fpu.write(self.fpu.register_map.CTRL.address, 1)
    infoLen = self.fpu.read(self.fpu.register_map.infoLength.address)
    return infoLen

  def get_info(self):
    '''
    Get FPU configuration from device

    Returns
    ----------
    configuration: dictionary
    '''
    infoLen = self.get_info_len()
    infoLen = self.get_info_len()
    info = pynq.buffer.allocate(infoLen, dtype='byte')
    self.fpu.write(self.fpu.register_map.info.address, info.physical_address)
    self.fpu.write(self.fpu.register_map.instruction.address, 0)
    self.fpu.write(self.fpu.register_map.CTRL.address, 1)
    return "".join([chr(i) for i in info])

  def _init_Xy_buffers(self, X_shape, y_shape):
    dtype = 'float32' if self.config.get('dynamic_scaler', False) else 'int32'
    self.Xbuf = pynq.allocate(X_shape, dtype=dtype)
    self.ybuf = pynq.allocate(y_shape, dtype=dtype)
    self.fpu.write(self.fpu.register_map.X.address, self.Xbuf.physical_address)
    self.fpu.write(self.fpu.register_map.y.address, self.ybuf.physical_address)

  def _init_buffers(self, batch_size=1):
    cfg = getattr(self, 'config', None)
    assert cfg is not None, 'Configuration not loaded'
    self._init_Xy_buffers((batch_size, cfg['features']), (batch_size, 1))
    self.interfaceNodes = pynq.allocate((self.config['tree_engines'], self.config['nodes'], 7), dtype='int32')
    self.scales = pynq.allocate(self.config['features'] + 1, dtype='float32') # todo 1 is placehold for number of classes

  def load(self, nodes, scales, n_features=1, n_classes=2, batch_size=None):
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
    self.interfaceNodes[:] = nodes
    self.scales[:] = scales
    # load the nodes
    self.fpu.write(self.fpu.register_map.nodes_in.address, self.interfaceNodes.physical_address)
    self.fpu.write(self.fpu.register_map.scales_in.address, self.scales.physical_address)
    # load
    self.fpu.write(self.fpu.register_map.instruction.address, 1)
    self.fpu.write(self.fpu.register_map.CTRL.address, 1)
    if batch_size is None:
      batch_size = self.Xbuf.shape[0]
    nc = 1 if n_classes == 2 else n_classes
    self._init_Xy_buffers((batch_size, n_features), (batch_size, nc))

  def read(self):
    '''
    Read packed model from FPU
    Sets device attributes interfaceNodes and scales
    '''
    # read back the nodes
    self.fpu.write(self.fpu.register_map.nodes_out.address, self.interfaceNodes.physical_address)
    self.fpu.write(self.fpu.register_map.scales_out.address, self.scales.physical_address)
    # read
    self.fpu.write(self.fpu.register_map.instruction.address, 2)
    self.fpu.write(self.fpu.register_map.CTRL.address, 1)

  def predict(self, X):
    '''
    Execute inference on FPU

    Parameters
    ----------
    X: ndarray of shape (batch_size, n_features), dtype float32 or int32
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
    self.fpu.write(self.fpu.register_map.batch_size.address, X.shape[0])
    self.fpu.write(self.fpu.register_map.n_features.address, X.shape[1])
    self.fpu.write(self.fpu.register_map.instruction.address, 3)
    self.fpu.write(self.fpu.register_map.CTRL.address, 1)
    return self.ybuf[:]

class AlveoDriver:
  def __init__(self, bitfile, fpu_name=None, batch_size=1):
    self.overlay = pynq.Overlay(bitfile)
    self.fpus = [getattr(self.overlay, at) for at in dir(self.overlay) if 'FPU' in at]
    fpu_name = 'FPU_Alveo_1' if fpu_name is None else fpu_name
    self.fpu = getattr(self.overlay, fpu_name, None)
    assert self.fpu is not None, f'No FPU {fpu_name} found in bitfle'
    info = json.loads(self.get_info())
    self.config = info['configuration']
    self.metadata = info['metadata']
    self._init_buffers(batch_size=batch_size)

  def get_info(self):
    '''
    Get FPU configuration from device

    Returns
    ----------
    configuration: dictionary
    '''
    self.fpu.write(self.fpu.register_map.CTRL.address, 1)
    infoLen =self. fpu.read(self.fpu.register_map.infoLength.address)
    info = pynq.allocate(infoLen, dtype='byte')
    dummy = pynq.allocate(1)
    self.fpu.call(dummy, dummy, 0, 0, 0, dummy, dummy, dummy, dummy, info, dummy)
    info.sync_from_device()
    return "".join([chr(i) for i in info])

  def _init_Xy_buffers(self, X_shape, y_shape):
    dtype = 'float32' if self.config.get('dynamic_scaler', False) else 'int32'
    self.Xbuf = pynq.allocate(X_shape, dtype=dtype)
    self.ybuf = pynq.allocate(y_shape, dtype=dtype)

  def _init_buffers(self, batch_size=1):
    cfg = getattr(self, 'config', None)
    assert cfg is not None, 'Configuration not loaded'
    self._init_Xy_buffers((batch_size, cfg['features']), (batch_size, 1))
    self.interfaceNodes = pynq.allocate((self.config['tree_engines'], self.config['nodes'], 7), dtype='int32')
    self.scales = pynq.allocate(self.config['features'] + 1, dtype='float32') # todo 1 is placehold for number of classes
    self._dummy_buf = pynq.allocate(1)

  def load(self, nodes, scales, n_features=1, n_classes=2, batch_size=None):
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
    self.interfaceNodes[:] = nodes
    self.scales[:] = scales
    self.interfaceNodes.sync_to_device()
    self.scales.sync_to_device()
    dummy = self._dummy_buf
    self.fpu.call(dummy, dummy, 1, 0, 0, self.interfaceNodes, dummy, self.scales, dummy, dummy, dummy)
    if batch_size is None:
      batch_size = self.Xbuf.shape[0]
    nc = 1 if n_classes == 2 else n_classes
    self._init_Xy_buffers((batch_size, n_features), (batch_size, nc))

  def predict(self, X):
    '''
    Execute inference on FPU

    Parameters
    ----------
    X: ndarray of shape (batch_size, n_features), dtype float32 or int32
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
    dummy = self._dummy_buf
    self.fpu.call(self.Xbuf, self.ybuf, 3, X.shape[0], X.shape[1], dummy, dummy, dummy, dummy, dummy, dummy)
    while not self.fpu.register_map.CTRL.AP_IDLE:
      pass
    self.ybuf.sync_from_device()
    return self.ybuf

  