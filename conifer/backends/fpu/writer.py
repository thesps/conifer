import copy
import math
import numpy as np
import json
import shutil
import os
from typing import List, Union
import logging
logger = logging.getLogger(__name__)
from conifer.model import ModelBase, ConfigBase, ModelMetaData
from conifer.utils import copydocstring
from conifer.backends.common import _TOOLS, get_hls, get_hls_build_command
from conifer.backends.boards import get_board_config, get_builder, BoardConfig, ZynqConfig, AlveoConfig

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

  def offset(self, base: int):
    '''Return a copy of the tree with child addresses moved to start at node address base'''
    nodes = []
    for node in self.nodes:
      cl = node.child_left + base if not node.is_leaf else node.child_left
      cr = node.child_right + base if not node.is_leaf else node.child_right
      nodes.append(FPUInterfaceNode(node.threshold, node.score, node.feature, cl, cr, node.iclass, node.is_leaf))
    return FPUInterfaceTree(nodes)

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
  _config_fields = ConfigBase._config_fields + ['nodes', 'tree_engines', 'roots', 'features', 'threshold_type', 'score_type', 'dynamic_scaler']
  _config_fields.remove('output_dir')
  _config_fields.remove('project_name')
  _fpu_alts = {'nodes'          : ['Nodes'],
               'tree_engines'   : ['TreeEngines'],
               'features'       : ['Features'],
               'threshold_type' : ['ThresholdType'],
               'score_type'     : ['ScoreType'],
               'dynamic_scaler' : ['DynamicScaler'],
               'roots'          : ['Roots']
               }
  _alternates = {**ConfigBase._alternates, **_fpu_alts}
  _fpu_defaults = {'nodes'          : 512,
                   'tree_engines'   : 100,
                   'roots'          : 16,
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

  def _validate(self):
    super(FPUConfig, self)._validate()
    assert 1 <= self.roots <= self.nodes, f'FPU roots must be between 1 and nodes ({self.nodes}), got {self.roots}'

  def default_config():
    return copy.deepcopy(FPUConfig._defaults)

  def generate_codename(self):
    template = 'fpu_{te}TE_{n}N_{r}R_{f}F_{tt}T_{st}S_{ds}DS'
    codename = template.format(te = self.tree_engines,
                               n = self.nodes,
                               r = self.roots,
                               f = self.features,
                               tt = self.threshold_type,
                               st = self.score_type,
                               ds = '' if self.dynamic_scaler else 'N')
    return codename

class FPUBuilderConfig(FPUConfig):
  backend = 'fpu_builder'
  _config_fields = ConfigBase._config_fields + FPUConfig._config_fields + ['board', 'clock_period']
  _fpu_builder_alts = {'board' : ['Board'], 'clock_period' : ['ClockPeriod']}
  _alternates = {**ConfigBase._alternates, **FPUConfig._alternates, **_fpu_builder_alts}
  _fpu_builder_defaults = {'board' : 'pynq-z2', 'clock_period' : 10}
  _defaults = {**ConfigBase._defaults, **FPUConfig._defaults, **_fpu_builder_defaults}
  def __init__(self, configDict, validate=True):
    super(FPUBuilderConfig, self).__init__(configDict, validate=False)
    if isinstance(self.board, str):
      self.board_config = get_board_config(self.board)
    elif isinstance(self.board, dict):
      self.board_config = get_board_config(self.board.get('name', None))
    elif isinstance(self.board, BoardConfig):
      self.board_config = self.board
    if validate:
      self._validate()

  def default_config():
    return copy.deepcopy(FPUBuilderConfig._defaults) 
  
  def top_name(self):
    top_names = {ZynqConfig  : 'FPU_Zynq',
                 AlveoConfig : 'FPU_Alveo',
                 }
    return top_names.get(type(self.board_config), None)

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
    for trees_t in ensembleDict['trees']:
      for ic, tree in enumerate(trees_t):
        interface_trees.append(FPUInterfaceTree.from_flat_tree_dictionary(tree, ic))
    self.interface_trees = interface_trees

    assert not self.is_oblique(), f'Oblique splits are not supported by the FPU backend, please use the hls backend'
    self.tree_engine_assignment = self.assign_tree_engines()

    if self.config.fpu.dynamic_scaler:
      t, s = self.derive_scales()
      self.scale(t, s)

  def attach_device(self, device, batch_size=None):
    '''
    Load model onto FPU device

    Parameters
    ----------
    device
      FPU runtime device (XrtDriver or ZynqDriver)
    batch_size: integer
      batch size used for allocating buffers
    '''
    self.device = device
    self.load(batch_size=batch_size)
    
  def assign_tree_engines(self):
    '''
    Assign the model trees to the FPU Tree Engines.
    When the model has no more trees than the FPU has Tree Engines, each tree gets its own Tree Engine.
    Otherwise trees are packed into the Tree Engines as multiple roots, with the Tree Engine inference latency
    (the sum of the depths of its trees) balanced as evenly as possible.
    Returns
    ----------
    assignment: list of length (FPU TEs) of lists of tree indices into interface_trees
    '''
    fpu_cfg = self.config.fpu
    n_tes, n_nodes, n_roots = fpu_cfg.tree_engines, fpu_cfg.nodes, fpu_cfg.roots
    tree_nodes = [tree.n_nodes() for tree in self.interface_trees]
    tree_depths = [tree.max_depth() for trees_c in self.trees for tree in trees_c]
    assert len(self.interface_trees) <= n_tes * n_roots, f'Cannot pack model with {len(self.interface_trees)} trees to FPU target with {n_tes} Tree Engines and {n_roots} roots (maximum {n_tes * n_roots} trees)'
    for i, n in enumerate(tree_nodes):
      assert n <= n_nodes, f'Cannot pack tree {i} with {n} nodes to FPU target with {n_nodes} nodes'

    def _assign(order, key):
      assignment = [[] for _ in range(n_tes)]
      nodes_used = [0] * n_tes
      latency = [0] * n_tes
      for i in order:
        candidates = [te for te in range(n_tes) if len(assignment[te]) < n_roots and nodes_used[te] + tree_nodes[i] <= n_nodes]
        if len(candidates) == 0:
          return None
        te = min(candidates, key=lambda te: key(te, latency, nodes_used))
        assignment[te].append(i)
        nodes_used[te] += tree_nodes[i]
        latency[te] += tree_depths[i] + 1
      return assignment

    trees = range(len(self.interface_trees))
    if len(trees) <= n_tes:
      return [[i] for i in trees] + [[] for _ in range(n_tes - len(trees))]
    # balance the latency, taking the deepest trees first
    assignment = _assign(sorted(trees, key=lambda i: (tree_depths[i], tree_nodes[i]), reverse=True),
                         lambda te, latency, nodes_used: (latency[te], nodes_used[te], te))
    if assignment is None:
      # fall back to first fit decreasing on the nodes, which packs tighter but may have worse latency
      logger.warning('Could not balance trees over Tree Engines, falling back to first fit packing')
      assignment = _assign(sorted(trees, key=lambda i: tree_nodes[i], reverse=True),
                           lambda te, latency, nodes_used: te)
    assert assignment is not None, f'Cannot pack model with {sum(tree_nodes)} nodes in {len(tree_nodes)} trees to FPU target with {n_tes} Tree Engines of {n_nodes} nodes and {n_roots} roots'
    return assignment

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
    nodes: ndarray of shape (FPU TEs, FPU nodes, 7), dtype int32
      The packed InterfaceDecisionTrees. The trees assigned to each TE are placed one after another.
    roots: ndarray of shape (FPU TEs, FPU roots + 1), dtype int32
      The number of trees assigned to each TE, followed by the address of each tree's root node
    '''
    fpu_cfg = self.config.fpu
    nodes = np.zeros((fpu_cfg.tree_engines, fpu_cfg.nodes, 7), dtype='int32')
    roots = np.zeros((fpu_cfg.tree_engines, fpu_cfg.roots + 1), dtype='int32')
    for te, tree_indices in enumerate(self.tree_engine_assignment):
      te_nodes = []
      roots[te, 0] = len(tree_indices)
      for r, i in enumerate(tree_indices):
        roots[te, r + 1] = len(te_nodes)
        te_nodes += self.interface_trees[i].offset(len(te_nodes)).nodes
      te_tree = FPUInterfaceTree(te_nodes)
      te_tree.pad_to(fpu_cfg.nodes)
      nodes[te] = te_tree.pack()
    return nodes, roots

  def load(self, batch_size=None):
    '''
    Load model onto attached FPU device
    '''
    assert self.device is not None, 'No device attached! Did you load the driver and attach_device first?'
    nodes, roots = self.pack()
    self.device.load(nodes, roots, self._scales(), self.n_features, self.n_classes, batch_size)

  @copydocstring(ModelBase.write)
  def decision_function(self, X):
    assert self.device is not None, 'No device attached! Did you load the driver and attach_device first?'
    return self.device.decision_function(X)

  def write(self):
    '''
    Write the FPU packed model (InterfaceDecisionTrees and scales) to JSON files.
    These files can be used to execute inference in contexts without access to the model object.
    '''
    self.save()
    with open(f'{self.config.output_dir}/nodes.json', 'w') as f:
      nodes, roots = self.pack()
      d = {'nodes' : nodes.tolist(), 'roots' : roots.tolist(), 'scales' : self._scales().tolist()}
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
      "roots": 16,
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
    top_name = self.cfg.top_name()
    ip_name = f'{top_name}_0'
    self.board_builder = get_builder(self.cfg, self.cfg.board_config, top_name=top_name, ip_name=ip_name)
    self.output_dir = os.path.abspath(self.output_dir)
    self._metadata = ModelMetaData()
    self._csim, self._cosim = False, False

  def default_cfg():
    return FPUBuilderConfig.default_config()

  def _info(self):
    '''The FPU info string, returned by the device'''
    info = {'configuration' : self.cfg._to_dict(), 'metadata' : self._metadata._to_dict()}
    return json.dumps(info)

  def write_params(self):
    with open(f'{self.output_dir}/parameters.h', 'w') as f:
      f.write('#ifndef CONIFER_FPU_PARAMS_H_\n#define CONIFER_FPU_PARAMS_H_\n')
      f.write('#include "fpu.h"\n')
      f.write(f'typedef {_resolve_type(self.threshold_type)} T;\n')
      f.write(f'typedef {_resolve_type(self.score_type)} U;\n')
      f.write(f'static const int NFEATURES={self.features};\n')
      f.write(f'static const int NTE={self.tree_engines};\n')
      f.write(f'static const int NNODES={self.nodes};\n')
      f.write(f'static const int NROOTS={self.roots};\n')
      f.write(f'static const int ADDRBITS={math.ceil(np.log2(self.nodes))+1};\n')
      f.write(f'static const int FEATBITS={math.ceil(np.log2(self.features))+1};\n')
      f.write(f'static const int NCLASSES={1};\n')
      f.write(f'static const int CLASSBITS={1};\n')
      f.write(f'static const bool SCALER={"true" if self.dynamic_scaler else "false"};\n')
      f.write(f'typedef DecisionNode<T,U,FEATBITS,ADDRBITS,CLASSBITS> DN;\n')
      info = self._info()
      info_fmt = info.replace('"', r'\"')
      f.write(f'static const char* theInfo = "{info_fmt}";\n')
      f.write(f'static const int theInfoLength = {len(info)};\n')
      f.write('#endif\n')

  def write_tcl(self):
    import conifer
    with open(f'{self.output_dir}/hls_parameters.tcl', 'w') as f:
      f.write(f'set prj_name {self.project_name}\n')
      f.write(f'set top {self.cfg.top_name()}\n')
      f.write(f'set part {self.cfg.board_config.xilinx_part}\n')
      f.write(f'set clock_period {self.clock_period}\n')
      f.write(f'set flow_target {self.board_builder.get_flow_target()}\n')
      f.write(f'set export_format {self.board_builder.get_export_format()}\n')
      f.write(f'set m_axi_addr64 {str(self.board_builder.get_maxi64()).lower()}\n')
      f.write(f'set version {conifer.__version__.major}.{conifer.__version__.minor}\n')
      f.write(f'set csim {int(self._csim)}\n')
      f.write(f'set cosim {int(self._cosim)}\n')
      if self._cosim:
        with open(f'{self.output_dir}/tb_data/X.dat') as fX:
          batch_size, n_features = [int(v) for v in fX.readline().split()]
        f.write(f'set cosim_X_depth {max(batch_size * n_features, 1)}\n')
        f.write(f'set cosim_y_depth {max(batch_size, 1)}\n')
        f.write(f'set cosim_info_depth {len(self._info())}\n')

  def write_testbench_data(self, model, X):
    '''
    Write a packed model and inputs for the HLS testbench, used for C Simulation and Cosimulation (see build).
    The testbench writes its predictions to {output_dir}/{project_name}/solution1/csim/build/tb_data/y.dat for C Simulation.
    Parameters
    ----------
    model: FPUModel
      Model targeting this FPU configuration
    X: ndarray of shape (batch_size, n_features)
      Inputs, float32 for FPUs with the dynamic scaler, otherwise int32
    '''
    for key in ['tree_engines', 'nodes', 'roots', 'features', 'dynamic_scaler']:
      assert getattr(model.config.fpu, key) == getattr(self.cfg, key), f'Model FPU {key} ({getattr(model.config.fpu, key)}) does not match the FPU being built ({getattr(self.cfg, key)})'
    tb_dir = f'{self.output_dir}/tb_data'
    os.makedirs(tb_dir, exist_ok=True)
    nodes, roots = model.pack()
    np.savetxt(f'{tb_dir}/nodes.dat', nodes.reshape(-1, 7), fmt='%d')
    np.savetxt(f'{tb_dir}/roots.dat', roots, fmt='%d')
    np.savetxt(f'{tb_dir}/scales.dat', model._scales(), fmt='%.9g')
    with open(f'{tb_dir}/X.dat', 'w') as f:
      f.write(f'{X.shape[0]} {X.shape[1]}\n')
      np.savetxt(f, X, fmt='%.9g' if self.dynamic_scaler else '%d')

  def write(self):
    filedir = os.path.dirname(os.path.abspath(__file__))
    logger.info(f"Writing project to {self.output_dir}")
    os.makedirs(self.output_dir, exist_ok=True)
    shutil.copyfile(f'{filedir}/src/build_hls.tcl', f'{self.output_dir}/build_hls.tcl')
    shutil.copyfile(f'{filedir}/src/fpu.cpp', f'{self.output_dir}/fpu.cpp')
    shutil.copyfile(f'{filedir}/src/fpu.h', f'{self.output_dir}/fpu.h')
    shutil.copyfile(f'{filedir}/src/fpu_tb.cpp', f'{self.output_dir}/fpu_tb.cpp')
    with open(f'{self.output_dir}/{self.project_name}.json', 'w') as f:
      json.dump(self.cfg._to_dict(), f)
    self.write_params()
    self.write_tcl()

  def build(self, csynth=True, bitfile=True, csim=False, cosim=False, **build_kwargs):
    '''
    Build FPU project
    Parameters
    ----------
    csynth: boolean (optional)
      Run HLS C Synthesis
    bitfile: boolean (optional)
      Create Vivado IPI project, run synthesis and implementation
    csim: boolean (optional)
      Run HLS C Simulation of the testbench. Call write_testbench_data first.
    cosim: boolean (optional)
      Run HLS C/RTL Cosimulation of the testbench. Call write_testbench_data first.
    '''
    self._csim, self._cosim = csim, cosim
    if csim or cosim:
      assert os.path.exists(f'{self.output_dir}/tb_data/X.dat'), 'No testbench data found, call write_testbench_data first'
    self.write()
    cwd = os.getcwd()
    os.chdir(self.output_dir)
    success = True
    if csynth:
      hls_tool = get_hls()
      if hls_tool is None:
        logger.error("No HLS in PATH (looked for {}). Did you source the appropriate Xilinx Toolchain?".format(', '.join(_TOOLS.values())))
        success = False
      else:
        cmd = f'{get_hls_build_command(hls_tool, "build_hls.tcl")} > hls_build.log'
        logger.info(f'Building FPU HLS with command "{cmd}"')
        success = success and os.system(cmd)==0
    if success and bitfile:
      success = success and self.board_builder.build(**build_kwargs)
    os.chdir(cwd)
    return success
  
  def package(self):
    self.board_builder.package()
