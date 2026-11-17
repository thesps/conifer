import logging
logger = logging.getLogger(__name__)
from conifer.backends.fpu.writer import FPUConfig
from conifer.backends.fpu.memory_map import FPUMemoryMap

class RTLDriver:
  '''
  Load models onto an RTL FPU (FPU_rtl) through its configuration port, over any bus: the bus is given by functions
  reading and writing a 32 bit word at a configuration port address. Inference is done by the design containing the
  FPU, so there is no decision_function. Use with FPUModel.attach_device.
  '''
  def __init__(self, read, write, verify=True):
    '''
    Parameters
    ----------
    read: function(address) -> int
    write: function(address, data)
    verify: bool
      Read back the model after loading it
    '''
    self.read, self.write, self.verify = read, write, verify
    self.config = self.get_info()
    self.memory_map = FPUMemoryMap(FPUConfig(self.config))

  def get_info(self):
    '''
    The FPU configuration (an FPUConfig dictionary), from its INFO words. INFO is region 0, so INFO word i is at
    address i whatever the configuration
    '''
    info = {name : self.read(i) for i, name in enumerate(FPUMemoryMap.info_names)}
    assert info['version'] == FPUMemoryMap.version, f'FPU memory map version {info["version"]}, expected {FPUMemoryMap.version}'
    config = {'backend' : 'fpu', 'tree_engines' : info['tree_engines'], 'nodes' : info['nodes'], 'roots' : info['roots'],
              'features' : info['features'], 'threshold_type' : info['threshold_bits'], 'score_type' : info['score_bits'],
              'dynamic_scaler' : False}
    node_bits = FPUMemoryMap(FPUConfig(config)).info()['node_bits']
    assert info['node_bits'] == node_bits, f'FPU node word of {info["node_bits"]} bits, expected {node_bits} for its configuration'
    return config

  def load(self, nodes, roots, scales=None, n_features=None, n_classes=None, batch_size=None):
    '''
    Load a packed model (FPUModel.pack). The scales, n_features, n_classes and batch_size of the device interface are
    unused: the RTL FPU has no dynamic scaler
    '''
    fpu = (self.config['tree_engines'], self.config['nodes'], self.config['roots'] + 1)
    assert nodes.shape[:2] + roots.shape[1:] == fpu, f'The model is for an FPU of (Tree Engines, nodes, roots + 1) {nodes.shape[:2] + roots.shape[1:]}, the FPU is {fpu}'
    writes = self.memory_map.writes(nodes, roots)
    addresses, data = [int(a) for a in writes[:, 0]], [int(d) for d in writes[:, 1]]
    self.write_all(addresses, data)
    if self.verify:
      read = self.read_all(addresses)
      bad = [(a, d, r) for a, d, r in zip(addresses, data, read) if r != d]
      assert not bad, f'{len(bad)} of {len(writes)} words read back differ, first (address, written, read): {bad[0]}'
    logger.info(f'Loaded the model with {len(writes)} configuration writes')

  def write_all(self, addresses, data):
    '''Write each word of data to its address, in order. Override to batch the writes on the bus'''
    for a, d in zip(addresses, data):
      self.write(a, d)

  def read_all(self, addresses):
    '''Read each address, in order. Override to batch the reads on the bus'''
    return [self.read(a) for a in addresses]

  def decision_function(self, X):
    raise NotImplementedError('The RTL FPU is run by the design containing it')
