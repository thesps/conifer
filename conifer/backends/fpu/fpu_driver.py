import pynq
import numpy as np
import json

class ZynqDriver:
  def __init__(self, bitfile, fpu_name=None):
    self.overlay = pynq.Overlay(bitfile)
    self.fpus = [getattr(self.overlay, at) for at in dir(self.overlay) if 'FPU' in at]
    fpu_name = 'FPU_0' if fpu_name is None else fpu_name
    self.fpu = getattr(self.overlay, fpu_name, None)
    assert self.fpu is not None, f'No FPU {fpu_name} found in bitfle'
    info = json.loads(self.get_info())
    self.config = info['configuration']
    self.metadata = info['metadata']

    self.interfaceNodes = pynq.buffer.allocate((self.config['tree_engines'], self.config['nodes'], 7), dtype='int32')
    self.scales = pynq.buffer.allocate(self.config['features'], dtype='float')

    self.Xbuf = pynq.buffer.allocate((self.config['features']), dtype='int')
    self.ybuf = pynq.buffer.allocate((1), dtype='int')
    self.fpu.write(self.fpu.register_map.X.address, self.Xbuf.physical_address)
    self.fpu.write(self.fpu.register_map.y.address, self.ybuf.physical_address)

  def get_info(self):
    self.fpu.write(self.fpu.register_map.CTRL.address, 1)
    infoLen = self.fpu.read(self.fpu.register_map.infoLength.address)
    info = pynq.buffer.allocate(infoLen, dtype='byte')
    self.fpu.write(self.fpu.register_map.info.address, info.physical_address)
    self.fpu.write(self.fpu.register_map.instruction.address, 0)
    self.fpu.write(self.fpu.register_map.CTRL.address, 1)
    return "".join([chr(i) for i in info])

  def load(self, nodes, scales):
    self.interfaceNodes[:] = nodes
    self.scales[:] = scales
    # load the nodes
    self.fpu.write(self.fpu.register_map.nodes_in.address, self.interfaceNodes.physical_address)
    self.fpu.write(self.fpu.register_map.scales_in.address, self.scales.physical_address)
    # load
    self.fpu.write(self.fpu.register_map.instruction.address, 1)
    self.fpu.write(self.fpu.register_map.CTRL.address, 1)

  def read(self):
    # read back the nodes
    self.fpu.write(self.fpu.register_map.nodes_out.address, self.interfaceNodes.physical_address)
    self.fpu.write(self.fpu.register_map.scales_out.address, self.scales.physical_address)
    # read
    self.fpu.write(self.fpu.register_map.instruction.address, 2)
    self.fpu.write(self.fpu.register_map.CTRL.address, 1)

  def predict(self, X):
    assert X.ndim == 1, "Expected 1D inputs. Batched inference is not currently supported"
    assert X.shape[0] <= self.config['features'], "More inputs were provided than this FPU supports ({} vs {})".format(X.shape[0], self.config['features'])
    self.Xbuf[:] = np.zeros(self.Xbuf.shape, dtype='int32')
    self.Xbuf[:X.shape[0]] = X
    self.fpu.write(self.fpu.register_map.instruction.address, 3)
    self.fpu.write(self.fpu.register_map.CTRL.address, 1)
    return self.ybuf[:]

class AlveoDriver:
  def __init__(self, bitfile, fpu_name=None):
    self.overlay = pynq.Overlay(bitfile)
    self.fpus = [getattr(self.overlay, at) for at in dir(self.overlay) if 'FPU' in at]
    fpu_name = 'FPU_Alveo_1' if fpu_name is None else fpu_name
    self.fpu = getattr(self.overlay, fpu_name, None)
    assert self.fpu is not None, f'No FPU {fpu_name} found in bitfle'
    info = json.loads(self.get_info())
    self.config = info['configuration']
    self.metadata = info['metadata']
    self._init_buffers()

  def get_info(self):
    self.fpu.write(self.fpu.register_map.CTRL.address, 1)
    infoLen =self. fpu.read(self.fpu.register_map.infoLength.address)
    info = pynq.allocate(infoLen, dtype='byte')
    dummy = pynq.allocate(1)
    self.fpu.call(dummy, dummy, 0, dummy, dummy, dummy, dummy, info, dummy)
    info.sync_from_device()
    return "".join([chr(i) for i in info])

  def _init_buffers(self):
    cfg = getattr(self, 'config', None)
    assert cfg is not None, 'Configuration not loaded'
    self.Xbuf = pynq.allocate(cfg['features'], dtype='int32')
    self.ybuf = pynq.allocate(1, dtype='int32')
    self.interfaceNodes = pynq.allocate((self.config['tree_engines'], self.config['nodes'], 7), dtype='int32')
    self.scales = pynq.allocate(self.config['features'], dtype='float')
    self._dummy_buf = pynq.allocate(1)

  def load(self, nodes, scales):
    self.interfaceNodes[:] = nodes
    self.scales[:] = scales
    self.interfaceNodes.sync_to_device()
    self.scales.sync_to_device()
    dummy = self._dummy_buf
    self.fpu.call(dummy, dummy, 1, self.interfaceNodes, dummy, self.scales, dummy, dummy, dummy)

  def predict(self, X):
    assert X.ndim == 1, "Expected 1D inputs. Batched inference is not currently supported"
    assert X.shape[0] <= self.config['features'], "More inputs were provided than this FPU supports ({} vs {})".format(X.shape[0], self.config['features'])
    self.Xbuf[:] = np.zeros(self.Xbuf.shape, dtype='int32')
    self.Xbuf[:X.shape[0]] = X
    self.Xbuf.sync_to_device()
    dummy = self._dummy_buf
    self.fpu.call(self.Xbuf, self.ybuf, 3, dummy, dummy, dummy, dummy, dummy, dummy)
    self.ybuf.sync_from_device()
    return self.ybuf

  