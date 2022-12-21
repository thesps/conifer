"""
import pynq

class FPUDriver(pynq.DefaultIP):
  def __init__(self, description):
    super().__init__(description=description)

  bindto = ['xilinx.com:hls:FPU:1.0']

  def get_info(self):
    self.write(self.register_map.CTRL.address, 1)
    infoLen = self.read(self.register_map.infoLength.address)
    info = pynq.buffer.allocate(infoLen, dtype='byte')
    self.write(self.register_map.info.address, info.physical_address)
    self.write(self.register_map.instruction.address, 0)
    self.write(self.register_map.CTRL.address, 1)

  def load(self, nodes):


  def read(self, nodes):
"""

import pynq
import numpy as np

class ZynqDriver:
  def __init__(self, bitfile, config, X_shape, y_shape):
    self.overlay = pynq.Overlay(bitfile)
    self.fpu = self.overlay.FPU_0
    self.config = config

    self.interfaceNodes = pynq.buffer.allocate(np.product([config['tree_engines'], config['nodes'], 7]), dtype='int32')
    self.scales = pynq.buffer.allocate(config['features'], dtype='float')

    self.Xbuf = pynq.buffer.allocate(X_shape, dtype='int')
    self.ybuf = pynq.buffer.allocate(y_shape, dtype='int')
    self.fpu.write(self.fpu.register_map.X.address, self.Xbuf.physical_address)
    self.fpu.write(self.fpu.register_map.y.address, self.ybuf.physical_address)

  def get_info(self):
    self.fpu.write(self.fpu.register_map.CTRL.address, 1)
    infoLen = self.read(self.fpu.register_map.infoLength.address)
    info = pynq.buffer.allocate(infoLen, dtype='byte')
    self.fpu.write(self.fpu.register_map.info.address, info.physical_address)
    self.fpu.write(self.fpu.register_map.instruction.address, 0)
    self.fpu.write(self.fpu.register_map.CTRL.address, 1)
    return info

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
    self.Xbuf[:] = X
    self.fpu.write(self.fpu.register_map.instruction.address, 3)
    self.fpu.write(self.fpu.register_map.CTRL.address, 1)
    return self.ybuf[:]


  