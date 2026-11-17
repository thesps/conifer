'''
The memory map of the configuration port of the RTL FPU (FPU_rtl), mirroring src/vhdl/fpu_rtl_pkg.vhd, and the
configuration writes loading a packed model. Independent of the bus connecting the port.
'''
import numpy as np
from conifer.utils import address_bits
from conifer.backends.fpu.core import _VHDLWriter

class FPUMemoryMap:
  '''
  Addresses of 32 bit words, with the fields from the least significant bit: word | slot | Tree Engine | region
    - region INFO (0): read only words describing the FPU, at the index in INFO
    - region ROOTS (1): slot r of a Tree Engine's roots memory (the number of trees, then the root node addresses)
    - region NODES (2): word k of node slot of a Tree Engine's node memory. A node is node_words words, least
      significant first, written in order: the write of the last word writes the node
  '''
  data_bits = 32
  region_info, region_roots, region_nodes = 0, 1, 2
  version = 1
  info_names = ['version', 'tree_engines', 'nodes', 'roots', 'features', 'threshold_bits', 'score_bits', 'node_bits']

  def __init__(self, cfg):
    '''cfg: FPUConfig of the FPU'''
    self.cfg = cfg
    self.core = _VHDLWriter(cfg)
    c = self.core
    self.node_words = -(-c.nodebits // self.data_bits)
    self.word_bits = address_bits(self.node_words)
    self.slot_bits = max(c.nawidth, c.rawidth)
    self.te_bits = address_bits(c.nte)
    self.slot_lo = self.word_bits
    self.te_lo = self.slot_lo + self.slot_bits
    self.region_lo = self.te_lo + self.te_bits
    self.address_bits = self.region_lo + 2
    # the node fields of FPUInterfaceNode.pack, and their widths: threshold score feature child_left child_right iclass is_leaf
    self.field_bits = [c.tbits, c.ubits, c.featbits, c.addrbits, c.addrbits, c.classbits, 1]

  def address(self, region, te=0, slot=0, word=0):
    return (region << self.region_lo) | (te << self.te_lo) | (slot << self.slot_lo) | word

  def info(self):
    '''The expected INFO words, by name'''
    c = self.core
    return dict(zip(self.info_names, [self.version, c.nte, self.cfg.nodes, self.cfg.roots, c.nf, c.tbits, c.ubits, c.nodebits]))

  def info_addresses(self):
    return [self.address(self.region_info, word=i) for i in range(len(self.info_names))]

  def node_word(self, fields):
    '''The node memory word (an int) from the 7 node fields, packed from the least significant bit'''
    w, lo = 0, 0
    for v, b in zip(fields, self.field_bits):
      w |= (int(v) & ((1 << b) - 1)) << lo
      lo += b
    return w

  def writes(self, nodes, roots):
    '''
    The configuration writes loading a packed model, in order
    Parameters
    ----------
    nodes, roots: the packed model, from FPUModel.pack
    Returns
    ----------
    writes: ndarray of shape (n_writes, 2), dtype uint64: address, data
    '''
    c = self.core
    mask_d, mask_r = (1 << self.data_bits) - 1, (1 << c.addrbits) - 1
    writes = []
    for te in range(c.nte):
      for r, v in enumerate(roots[te]):
        writes.append((self.address(self.region_roots, te, r), int(v) & mask_r))
      for n, fields in enumerate(nodes[te]):
        w = self.node_word(fields)
        for k in range(self.node_words):
          writes.append((self.address(self.region_nodes, te, n, k), (w >> (k * self.data_bits)) & mask_d))
    return np.array(writes, dtype='uint64')
