import copy
import math
import numpy as np
import json
import shutil
import time
import os
import zipfile
from typing import List
import logging
logger = logging.getLogger(__name__)
from conifer.model import ModelBase, ConfigBase, ModelMetaData
from conifer.utils import copydocstring

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
    if isinstance(threshold, np.ndarray):
      self.threshold *= threshold[self.feature]
    else:
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

  def generate_codename(self):
    import conifer
    template = 'fpu_{te}TE_{n}N_{f}F_{tt}T_{st}S_{ds}DS'
    codename = template.format(te = self.tree_engines,
                               n = self.nodes,
                               f = self.features,
                               tt = self.threshold_type,
                               st = self.score_type,
                               ds = '' if self.dynamic_scaler else 'N')
    return codename

class FPUBuilderConfig(FPUConfig):
  backend = 'fpu_builder'
  _config_fields = ConfigBase._config_fields + FPUConfig._config_fields + ['part', 'clock_period']
  _fpu_builder_alts = {'part' : ['Part'], 'clock_period' : ['ClockPeriod']}
  _alternates = {**ConfigBase._alternates, **FPUConfig._alternates, **_fpu_builder_alts}
  _fpu_builder_defaults = {'part' : 'xc7z020clg400-1', 'clock_period' : 10}
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

    if self.config.fpu.dynamic_scaler:
      t, s = self.derive_scales()
      self.scale(t, s)

  def attach_device(self, device, batch_size=None):
    '''
    Load model onto FPU device

    Parameters
    ----------
    device
      FPU runtime device (AlveoDriver or ZynqDriver)
    batch_size: integer
      batch size used for allocating buffers
    '''
    self.device = device
    self.load(batch_size=batch_size)
    
  def pad_to(self, n_trees, n_nodes):
    for tree in self.interface_trees:
      tree.pad_to(n_nodes)
    self.interface_trees += [FPUInterfaceTree._null_tree(n_nodes)] * (n_trees - self.n_trees)

  def derive_scales(self):
    '''
    Derive threshold and score scale factors from static analysis of model parameters, and configured precision.
    Returns
    ----------
    threshold_scales: ndarray of shape (n_features)
      Scale factors derived for thresholds
    score_scales: ndarray of shape (n_classes)
      Scale factors derived for scores
    '''
    # only scale thresholds of non-leaf nodes
    thresholds = np.array([t for trees_c in self.trees for tree in trees_c for t, f in zip(tree.threshold, tree.feature) if f != -2])
    features = np.array([f for trees_c in self.trees for tree in trees_c for f in tree.feature if f != -2])
    threshold_scales = np.zeros(shape=self.n_features, dtype='float32')
    h = 2**(self.config.fpu.threshold_type-1)-1
    for i in range(self.n_features):
      t = np.abs(thresholds[features == i])
      t = t[t != 0]
      threshold_scales[i] = 1. if len(t) == 0 else h / t.max()
    # only scale the scores of leaf nodes
    v = np.array([v for trees_c in self.trees for tree in trees_c for v, f in zip(tree.value, tree.feature) if f == -2])
    v = np.abs(v[v != 0])
    h = (2**(self.config.fpu.score_type-1)-1) / self.n_trees
    score_scales = np.array([h / v.max()])
    return threshold_scales, score_scales

  def scale(self, threshold: float, score: float):
    '''
    Scale model tresholds and scores by scale factors
    Parameters
    ----------
    threshold: ndarray of shape (n_features) or scalar
      scale factors by which to multiply thresholds
    score: ndarray of shape (n_classes) or scalar
      scale factors by which to divide scores
    '''
    logger.info(f'Scaling model with threshold scales {threshold}, score scales {score}')
    self.threshold_scale = threshold
    self.score_scale = 1. / score
    for tree in self.interface_trees:
      tree.scale(threshold, score)

  def pack(self):
    '''
    Pack model into FPU InterfaceDecisionTrees
    Returns
    ----------
    data: ndarray of shape (FPU TEs, FPU nodes, 7), dtype int32
      The packed InterfaceDecisionTrees
    '''
    assert self.n_trees <= self.config.fpu.tree_engines, f'Cannot pack model with {self.n_trees} trees to FPU target with {self.config.fpu.tree_engines} Tree Engines'
    assert 2**self.max_depth <= self.config.fpu.nodes, f'Cannot pack model with max_depth {self.max_depth} to FPU target with {self.nodes} nodes'
    data = np.zeros((self.config.fpu.tree_engines, self.config.fpu.nodes, 7), dtype='int32')
    for i, tree in enumerate(self.interface_trees):
      data[i] = tree.pack()
    return data

  def load(self, batch_size=None):
    '''
    Load model onto attached FPU device
    '''
    assert self.device is not None, 'No device attached! Did you load the driver and attach_device first?'
    self.device.load(self.pack(), self._scales(), self.n_features, self.n_classes, batch_size)

  @copydocstring(ModelBase.write)
  def decision_function(self, X):
    assert self.device is not None, 'No device attached! Did you load the driver and attach_device first?'
    return self.device.predict(X)

  def write(self):
    '''
    Write the FPU packed model (InterfaceDecisionTrees and scales) to JSON files.
    These files can be used to execute inference in contexts without access to the model object.
    '''
    self.save()
    with open(f'{self.config.output_dir}/nodes.json', 'w') as f:
      d = {'nodes' : self.pack().tolist(), 'scales' : self._scales().tolist()}
      json.dump(d, f)

  def _scales(self):
    '''
    Get the scales packed for FPU loading
    Returns
    ----------
    scales: ndarray of shape (FPU features + 1)
    '''
    scales = np.ones(self.config.fpu.features + 1, dtype='float32') # todo 1 is a placeholder for classes
    scales[:self.n_features] = self.threshold_scale
    scales[-1] = self.score_scale
    return scales

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
      f.write(f'static const int NCLASSES={1};\n')
      f.write(f'static const int CLASSBITS={1};\n')
      f.write(f'static const bool SCALER={"true" if self.dynamic_scaler else "false"};\n')
      f.write(f'typedef DecisionNode<T,U,FEATBITS,ADDRBITS,CLASSBITS> DN;\n')
      info = {'configuration' : self.cfg._to_dict(), 'metadata' : self._metadata._to_dict()}
      info = json.dumps(info)
      info_fmt = info.replace('"', r'\"')
      f.write(f'static const char* theInfo = "{info_fmt}";\n')
      f.write(f'static const int theInfoLength = {len(info)};\n')
      f.write('#endif\n')

  def write(self):
    filedir = os.path.dirname(os.path.abspath(__file__))
    logger.info(f"Writing project to {self.output_dir}")
    os.makedirs(self.output_dir, exist_ok=True)
    shutil.copyfile(f'{filedir}/src/fpu.cpp', f'{self.output_dir}/fpu.cpp')
    shutil.copyfile(f'{filedir}/src/fpu.h', f'{self.output_dir}/fpu.h')
    with open(f'{self.output_dir}/{self.project_name}.json', 'w') as f:
      json.dump(self.cfg._to_dict(), f)
    self.write_params()

  def build(self, csynth=True):
    raise NotImplementedError

  def package(self):
    raise NotImplementedError

class ZynqFPUBuilderConfig(FPUBuilderConfig):
  backend = 'fpu_builder'
  _config_fields = FPUBuilderConfig._config_fields + ['board_part', 'processing_system_ip', 'processing_system']
  _zynq_fpu_builder_alts = {'board_part' : ['BoardPart'],
                            'processing_system_ip' : ['ProcessingSystemIP'],
                            'processing_system' : ['ProcessingSystem']}
  _alternates = {**FPUBuilderConfig._alternates, **_zynq_fpu_builder_alts}
  _zynq_fpu_builder_defaults = {'board_part' : 'tul.com.tw:pynq-z2:part0:1.0',
                                'processing_system_ip' : 'xilinx.com:ip:processing_system7:5.5',
                                'processing_system' : 'processing_system7'}
  _defaults = {**FPUBuilderConfig._defaults, **_zynq_fpu_builder_defaults}
  def __init__(self, configDict, validate=True):
    super(ZynqFPUBuilderConfig, self).__init__(configDict, validate=False)
    if validate:
      self._validate()

  def default_config():
    return copy.deepcopy(ZynqFPUBuilderConfig._defaults) 

class ZynqFPUBuilder(FPUBuilder):
  def __init__(self, cfg): 
    super(ZynqFPUBuilder, self).__init__(cfg)
    self.cfg = ZynqFPUBuilderConfig(cfg)
    for key, value in ZynqFPUBuilderConfig.default_config().items():
      setattr(self, key, getattr(self.cfg, key, value))

  def default_cfg():
    return ZynqFPUBuilderConfig.default_config()

  def write(self):
    '''
    Write project files
    '''
    super().write()
    filedir = os.path.dirname(os.path.abspath(__file__))
    shutil.copyfile(f'{filedir}/src/build_hls.tcl', f'{self.output_dir}/build_hls.tcl')
    shutil.copyfile(f'{filedir}/src/build_bit.tcl', f'{self.output_dir}/build_bit.tcl')
    self.write_tcl()

  def write_tcl(self):
    import conifer
    with open(f'{self.output_dir}/parameters.tcl', 'w') as f:
      f.write(f'set prj_name {self.project_name}\n')
      f.write(f'set part {self.part}\n')
      f.write(f'set board_part {self.board_part}\n')
      f.write(f'set processing_system_ip {self.processing_system_ip}\n')
      f.write(f'set processing_system {self.processing_system}\n')
      f.write(f'set clock_period {self.clock_period}\n')
      f.write(f'set m_axi_addr64 false\n')
      f.write(f'set top FPU_Zynq\n')
      f.write(f'set flow_target vivado\n')
      f.write(f'set version {conifer.__version__.major}.{conifer.__version__.minor}\n')
      f.write(f'set export_format ip_catalog\n')

  def build(self, csynth=True, bitfile=True):
    '''
    Build FPU project
    Parameters
    ----------
    csynth: boolean (optional)
      Run HLS C Synthesis
    bitfile: boolean (optional)
      Create Vivado IPI project, run synthesis and implementation
    '''
    self.write()
    cwd = os.getcwd()
    os.chdir(self.output_dir)
    success = True
    if csynth:
      logger.info(f"Building FPU HLS")
      success = success and os.system('vitis_hls -f build_hls.tcl')==0
    if success and bitfile:
      logger.info(f"Building FPU bitfile")
      success = success and os.system('vivado -mode batch -source build_bit.tcl')==0
    os.chdir(cwd)
    return success

  def package(self, retry: int = 6, retries: int = 10):
    '''
    Collect build products and compress to a zip file
    Parameters
    ----------
    retry: int (optional)
      wait time in seconds before retrying
    retries: int (optional)
      number of retries before exiting
    '''
    logger.info(f'Packaging FPU bitfile to {self.output_dir}/{self.project_name}.zip')

    for attempt in range(retries):
      if os.path.exists(f'{self.output_dir}/{self.project_name}_vivado/fpu.xsa'):
        with zipfile.ZipFile(f'{self.output_dir}/{self.project_name}_vivado/fpu.xsa', 'r') as zip:
          zip.extractall(f'{self.output_dir}/{self.project_name}_vivado/package')
        os.makedirs(f'{self.output_dir}/package', exist_ok=True)
        shutil.copyfile(f'{self.output_dir}/{self.project_name}_vivado/package/fpu.bit', f'{self.output_dir}/package/fpu.bit')
        shutil.copyfile(f'{self.output_dir}/{self.project_name}_vivado/package/design_1.hwh', f'{self.output_dir}/package/fpu.hwh')
        shutil.copyfile(f'{self.output_dir}/{self.project_name}.json', f'{self.output_dir}/package/fpu.json')
        filedir = os.path.dirname(os.path.abspath(__file__))
        with zipfile.ZipFile(f'{self.output_dir}/{self.project_name}.zip', 'w') as zip:
          zip.write(f'{self.output_dir}/package/fpu.bit', 'fpu.bit')
          zip.write(f'{self.output_dir}/package/fpu.hwh', 'fpu.hwh')
          zip.write(f'{self.output_dir}/package/fpu.json', 'fpu.json')
          zip.write(f'{filedir}/src/LICENSE', 'LICENSE')
        break
      else:
        logger.info(f'Bitfile not found, waiting {retry} seconds for retry')
        time.sleep(retry)      

class AlveoFPUBuilderConfig(FPUBuilderConfig):
  backend = 'fpu_builder'
  _config_fields = FPUBuilderConfig._config_fields + ['platform']
  _alveo_fpu_builder_alts = {'platform' : ['Platform']}
  _alternates = {**FPUBuilderConfig._alternates, **_alveo_fpu_builder_alts}
  _alveo_fpu_builder_defaults = {'platform' : 'xilinx_u200_gen3x16_xdma_2_202110_1'}
  _defaults = {**FPUBuilderConfig._defaults, **_alveo_fpu_builder_defaults}
  _defaults['clock_period'] = 3
  def __init__(self, configDict, validate=True):
    super(AlveoFPUBuilderConfig, self).__init__(configDict, validate=False)
    if validate:
      self._validate()

  def default_config():
    return copy.deepcopy(AlveoFPUBuilderConfig._defaults) 

class AlveoFPUBuilder(FPUBuilder):
  def __init__(self, cfg): 
    super(AlveoFPUBuilder, self).__init__(cfg)
    self.cfg = AlveoFPUBuilderConfig(cfg)
    for key, value in AlveoFPUBuilderConfig.default_config().items():
      setattr(self, key, getattr(self.cfg, key, value))

  def default_cfg():
    return AlveoFPUBuilderConfig.default_config()

  def write(self):
    '''
    Write project files
    '''
    super().write()
    filedir = os.path.dirname(os.path.abspath(__file__))
    shutil.copyfile(f'{filedir}/src/build_hls.tcl', f'{self.output_dir}/build_hls.tcl')
    shutil.copyfile(f'{filedir}/src/build_bit.tcl', f'{self.output_dir}/build_bit.tcl')
    self.write_tcl()

  def write_tcl(self):
    import conifer
    with open(f'{self.output_dir}/parameters.tcl', 'w') as f:
      f.write(f'set prj_name {self.project_name}\n')
      f.write(f'set part {self.part}\n')
      f.write(f'set clock_period {self.clock_period}\n')
      f.write(f'set m_axi_addr64 true\n')
      f.write(f'set top FPU_Alveo\n')
      f.write(f'set flow_target vitis\n')
      f.write(f'set version {conifer.__version__.major}.{conifer.__version__.minor}\n')
      f.write(f'set export_format xo\n')

  def build(self, csynth=True, bitfile=True):
    '''
    Build FPU project
    Parameters
    ----------
    csynth: boolean (optional)
      Run HLS C Synthesis
    bitfile: boolean (optional)
      Run v++ linkage (Place and Route)
    '''
    self.write()
    cwd = os.getcwd()
    os.chdir(self.output_dir)
    success = True
    if csynth:
      logger.info(f"Building FPU HLS")
      success = success and os.system('vitis_hls -f build_hls.tcl')==0
    if success and bitfile:
      logger.info(f"Building FPU bitfile")
      pn = self.project_name
      vitis_cmd = f'v++ -t hw --platform {self.platform} --link {pn}/solution1/impl/export.xo -o {pn}.xclbin'
      success = success and os.system(vitis_cmd)==0
    os.chdir(cwd)
    return success

  def package(self, retry: int = 6, retries: int = 10):
    '''
    Collect build products and compress to a zip file
    Parameters
    ----------
    retry: int (optional)
      wait time in seconds before retrying
    retries: int (optional)
      number of retries before exiting
    '''
    logger.info(f'Packaging FPU bitfile to {self.output_dir}/{self.project_name}.zip')

    for attempt in range(retries):
      if os.path.exists(f'{self.output_dir}/{self.project_name}.xclbin'):
        filedir = os.path.dirname(os.path.abspath(__file__))
        with zipfile.ZipFile(f'{self.output_dir}/{self.project_name}.zip', 'w') as zip:
          zip.write(f'{self.output_dir}/{self.project_name}.xclbin', f'{self.project_name}.xclbin')
          zip.write(f'{self.output_dir}/{self.project_name}.json', f'{self.project_name}.json')
          zip.write(f'{filedir}/src/LICENSE', 'LICENSE')
        break
      else:
        logger.info(f'Bitfile not found, waiting {retry} seconds for retry')
        time.sleep(retry)
