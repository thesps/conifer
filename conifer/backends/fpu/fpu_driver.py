import pynq
import numpy as np
import json

class ZynqDriver:
  def __init__(self, bitfile):
    self.overlay = pynq.Overlay(bitfile)
    self.fpu = self.overlay.FPU_0
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


  