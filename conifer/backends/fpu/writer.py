import copy
import math
import numpy as np
import json
import shutil
import os
import zipfile
from typing import List
import logging
logger = logging.getLogger(__name__)
from conifer.model import ModelBase, ConfigBase, ModelMetaData
try:
    from conifer.backends.fpu.fpu_driver import ZynqDriver
except ImportError:
    FPUDriver = None

class FPUInterfaceNode:
  '''
  Python representation of the Node sent to/from the FPU
  '''
  def __init__(self,
               threshold: int, 
               score: int,
               feature: int,
               child_left: int,
               child_right: int,
               iclass: int,
               is_leaf: int):
    self.threshold = threshold
    self.score = score
    self.feature = feature
    self.child_left = child_left
    self.child_right = child_right
    self.iclass = iclass
    self.is_leaf = is_leaf

  def scale(self, threshold, score):
    self.threshold *= threshold
    self.score *= score

  def pack(self) -> List[int] :
    '''
    Pack the fields as expected by the FPU
    '''
    fields = np.zeros(7, dtype='int')
    fields[0] = self.threshold
    fields[1] = self.score
    fields[2] = self.feature
    fields[3] = self.child_left
    fields[4] = self.child_right
    fields[5] = self.iclass
    fields[6] = self.is_leaf
    return fields

  def unpack(fields: List[int]) :
    return FPUInterfaceNode(*fields)

  def _null_node():
    return FPUInterfaceNode(0, 0, -2, -1, -1, 0, 1)

class FPUInterfaceTree:
  def __init__(self, nodes: List[FPUInterfaceNode]):
    self.nodes = nodes

  def n_nodes(self):
    return len(self.nodes)

  def pad_to(self, n: int):
    '''Pad the tree up to n nodes with null nodes'''
    self.nodes = self.nodes + [FPUInterfaceNode._null_node()] * (n - self.n_nodes())

  def scale(self, threshold: float, score: float):
    for node in self.nodes:
      node.scale(threshold, score)

  def pack(self):
    '''Pack the tree for sending to the FPU'''
    data = np.zeros((self.n_nodes(), 7), dtype='int')
    for i, node in enumerate(self.nodes):
      data[i] = node.pack()
    return data

  def unpack(data):
    nodes = []
    n_nodes = data.ravel().shape[0] // 7
    for d in data.reshape((n_nodes, 7)):
      nodes.append(FPUInterfaceNode(*d))
    return FPUInterfaceTree(nodes)

  def from_flat_tree_dictionary(tree, iclass):
    n_nodes = len(tree['feature'])
    nodes = []
    for i in range(n_nodes):
      nodes.append(FPUInterfaceNode(tree['threshold'][i],
                                    tree['value'][i],
                                    tree['feature'][i],
                                    tree['children_left'][i],
                                    tree['children_right'][i],
                                    iclass,
                                    tree['feature'][i] == -2))
    return FPUInterfaceTree(nodes)

  def _null_tree(n: int):
    nodes = [FPUInterfaceNode._null_node()] * n
    return FPUInterfaceTree(nodes)

class FPUConfig(ConfigBase):
  backend = 'fpu'
  _config_fields = ConfigBase._config_fields + ['nodes', 'tree_engines', 'features', 'threshold_type', 'score_type', 'dynamic_scaler']
  _config_fields.remove('output_dir')
  _config_fields.remove('project_name')
  _fpu_alts = {'nodes'          : ['Nodes'],
               'tree_engines'   : ['TreeEngines'],
               'features'       : ['Features'],
               'threshold_type' : ['ThresholdType'],
               'score_type'     : ['ScoreType'],
               'dynamic_scaler' : ['DynamicScaler']
               }
  _alternates = {**ConfigBase._alternates, **_fpu_alts}
  _fpu_defaults = {'nodes'          : 512,
                   'tree_engines'   : 100,
                   'features'       : 16,
                   'threshold_type' : 16,
                   'score_type'     : 16,
                   'dynamic_scaler' : True
                    }
  _defaults = {**ConfigBase._defaults, **_fpu_defaults}
  def __init__(self, configDict, validate=True):
    super(FPUConfig, self).__init__(configDict, validate=False)
    if validate:
      self._validate()

  def default_config():
    return copy.deepcopy(FPUConfig._defaults)

class FPUBuilderConfig(FPUConfig):
  backend = 'fpu_builder'
  _config_fields = ConfigBase._config_fields + FPUConfig._config_fields + ['part']
  _fpu_builder_alts = {'part' : ['Part']}
  _alternates = {**ConfigBase._alternates, **FPUConfig._alternates, **_fpu_builder_alts}
  _fpu_builder_defaults = {'part' : 'xc7z020clg400-1'}
  _defaults = {**ConfigBase._defaults, **FPUConfig._defaults, **_fpu_builder_defaults}
  def __init__(self, configDict, validate=True):
    super(FPUBuilderConfig, self).__init__(configDict, validate=False)
    if validate:
      self._validate()

  def default_config():
    return copy.deepcopy(FPUBuilderConfig._defaults) 
  

class FPUModelConfig(ConfigBase):
  backend = 'fpu'
  _config_fields = ConfigBase._config_fields + ['fpu']
  _fpu_alts = {'fpu' : ['FPU']}
  _alternates = {**ConfigBase._alternates, **_fpu_alts}
  _fpu_defaults = {'fpu' : FPUConfig.default_config()}
  _defaults = {**ConfigBase._defaults, **_fpu_defaults}
  def __init__(self, configDict, validate=True):
    super(FPUModelConfig, self).__init__(configDict, validate=False)
    self.fpu = FPUConfig(self.fpu)
    if validate:
      self._validate()

  def default_config():
    return copy.deepcopy(FPUModelConfig._defaults)

class FPUModel(ModelBase):

  def __init__(self, ensembleDict, config, metadata=None):
    super(FPUModel, self).__init__(ensembleDict, config, metadata)
    self.config = FPUModelConfig(config)
    #assert len(ensembleDict['trees']) == 1, 'Only binary classification models are currently supported'
    interface_trees = []
    for ic, tree_class in enumerate(ensembleDict['trees']):
      for tree in tree_class:
        interface_trees.append(FPUInterfaceTree.from_flat_tree_dictionary(tree, ic))
    self.interface_trees = interface_trees

    fpu_cfg = self.config.fpu
    self.pad_to(fpu_cfg.tree_engines, fpu_cfg.nodes)

  def attach_device(self, bitfile):
    self.driver = ZynqDriver(bitfile, self.config, self.config.features, 1)
    
  def pad_to(self, n_trees, n_nodes):
    for tree in self.interface_trees:
      tree.pad_to(n_nodes)
    self.interface_trees += [FPUInterfaceTree._null_tree(n_nodes)] * (n_trees - self.n_trees)

  def scale(self, threshold: float, score: float):
    for tree in self.interface_trees:
      tree.scale(threshold, score)

  def pack(self):
    data = np.zeros((self.config.fpu.tree_engines, self.config.fpu.nodes, 7), dtype='int32')
    for i, tree in enumerate(self.interface_trees):
      data[i] = tree.pack()
    return data

  def load(self):
    self.driver.load(self.pack(), np.ones(self.config.fpu.features, dtype='float'))

  def decision_function(self, X):
    return self.driver.predict(X)

  def write(self):
    self.save()
    with open(f'{self.config.output_dir}/nodes.json', 'w') as f:
      d = {'nodes' : self.pack().tolist(), 'scales' : np.ones(self.config.fpu.features, dtype='float').tolist()}
      json.dump(d, f)

def make_model(ensembleDict, config):
    return FPUModel(ensembleDict, config)

def auto_config():
    config = {'Backend'     : 'fpu',
              'ProjectName' : 'my-prj',
              'OutputDir'   : 'my-conifer-prj',
              'Precision'   : 'ap_fixed<18,8>'}
    fpu_cfg = {
      "nodes": 512,
      "tree_engines": 100,
      "features": 16,
      "threshold_type": 16,
      "score_type": 16,
      "dynamic_scaler": True
    }
    config['FPU'] = fpu_cfg
    return config

def _resolve_type(t):
  if isinstance(t, int):
    return f'ap_fixed<{t},{t}>'
  elif isinstance(t, str):
    return t

class FPUBuilder:

  def __init__(self, cfg,):
    self.cfg = FPUBuilderConfig(cfg)
    for key, value in FPUBuilderConfig.default_config().items():
      setattr(self, key, getattr(self.cfg, key, value))
    self.output_dir = os.path.abspath(self.output_dir)
    self._metadata = ModelMetaData()

  def default_cfg():
    return FPUBuilderConfig.default_config()

  def write_params(self):
    import conifer
    with open(f'{self.output_dir}/parameters.h', 'w') as f:
      f.write('#ifndef CONIFER_FPU_PARAMS_H_\n#define CONIFER_FPU_PARAMS_H_\n')
      f.write('#include "fpu.h"\n')
      f.write(f'typedef {_resolve_type(self.threshold_type)} T;\n')
      f.write(f'typedef {_resolve_type(self.score_type)} U;\n')
      f.write(f'static const int NFEATURES={self.features};\n')
      f.write(f'static const int NTE={self.tree_engines};\n')
      f.write(f'static const int NNODES={self.nodes};\n')
      f.write(f'static const int ADDRBITS={math.ceil(np.log2(self.nodes))+1};\n')
      f.write(f'static const int FEATBITS={math.ceil(np.log2(self.features))+1};\n')
      f.write(f'static const int CLASSBITS={1};\n')
      f.write(f'static const bool SCALER={"true" if self.dynamic_scaler else "false"};\n')
      f.write(f'typedef DecisionNode<T,U,FEATBITS,ADDRBITS,CLASSBITS> DN;\n')
      info = {'configuration' : self.cfg._to_dict(), 'metadata' : self._metadata._to_dict()}
      info = json.dumps(info)
      info_fmt = info.replace('"', r'\"')
      f.write(f'static const char* theInfo = "{info_fmt}";\n')
      f.write(f'static const int theInfoLength = {len(info)};\n')
      f.write('#endif\n')

    with open(f'{self.output_dir}/parameters.tcl', 'w') as f:
      f.write(f'set prj_name {self.project_name}\n')
      f.write(f'set part {self.part}\n')

  def write(self):
    filedir = os.path.dirname(os.path.abspath(__file__))
    logger.info(f"Writing project to {self.output_dir}")
    os.makedirs(self.output_dir, exist_ok=True)
    shutil.copyfile(f'{filedir}/src/fpu.cpp', f'{self.output_dir}/fpu.cpp')
    shutil.copyfile(f'{filedir}/src/fpu.h', f'{self.output_dir}/fpu.h')
    shutil.copyfile(f'{filedir}/src/build_hls.tcl', f'{self.output_dir}/build_hls.tcl')
    shutil.copyfile(f'{filedir}/src/build_bit.tcl', f'{self.output_dir}/build_bit.tcl')
    with open(f'{self.output_dir}/{self.project_name}.json', 'w') as f:
      json.dump(self.cfg._to_dict(), f)
    self.write_params()

  def build(self, csynth=True, bitfile=True):
    self.write()
    cwd = os.getcwd()
    os.chdir(self.output_dir)
    if csynth:
      logger.info(f"Building FPU HLS")
      os.system('vitis_hls -f build_hls.tcl')
    if bitfile:
      logger.info(f"Building FPU bitfile")
      os.system('vivado -mode batch -source build_bit.tcl')
      self.package()
    os.chdir(cwd)

  def package(self):
    logger.info(f'Packaging FPU bitfile to {self.output_dir}/{self.project_name}.zip')
    with zipfile.ZipFile(f'{self.output_dir}/{self.project_name}_vivado/fpu.xsa', 'r') as zip:
      zip.extractall(f'{self.output_dir}/{self.project_name}_vivado/package')
    os.makedirs(f'{self.output_dir}/package', exist_ok=True)
    shutil.copyfile(f'{self.output_dir}/{self.project_name}_vivado/package/fpu.bit', f'{self.output_dir}/package/fpu.bit')
    shutil.copyfile(f'{self.output_dir}/{self.project_name}_vivado/package/design_1.hwh', f'{self.output_dir}/package/fpu.hwh')
    shutil.copyfile(f'{self.output_dir}/{self.project_name}.json', f'{self.output_dir}/package/fpu.json')
    with zipfile.ZipFile(f'{self.output_dir}/{self.project_name}.zip', 'w') as zip:
      zip.write(f'{self.output_dir}/package/fpu.bit', 'fpu.bit')
      zip.write(f'{self.output_dir}/package/fpu.hwh', 'fpu.hwh')
      zip.write(f'{self.output_dir}/package/fpu.json', 'fpu.json')

def old_info():
      info =  'static const FPUInfo theInfo = {{"{v}",\n'
      info += '{s}"{t}",\n{s}"{u}",\n{s}{NF},\n{s}{NT},\n{s}{NN},\n'
      info += '{s}{AB},\n{s}{FB},\n{s}{CB},\n{s}{S}\n{s}}};\n'
      def _char32(s):
        # crop to 31 to allow for char array terminator
        c = s[:31] if len(s) > 31 else s
        c = ' ' * (31 - len(s)) + s if len(s) < 31 else s
        return c
      v = conifer.__version__
      info = info.format(s = space,
                         v = _char32(v),
                         t = _char32(_resolve_type(self.threshold_type)),
                         u = _char32(_resolve_type(self.score_type)),
                         NF = self.features,
                         NT = self.tree_engines,
                         NN = self.nodes,
                         AB = math.ceil(np.log2(self.nodes))+1,
                         FB = math.ceil(np.log2(self.features))+1,
                         CB = 1,
                         S = 1 if self.dynamic_scaler else 0)
      logger.debug(info)