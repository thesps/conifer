import pynq

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
  
class AlveoDriver:
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