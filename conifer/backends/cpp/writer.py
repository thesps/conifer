import os
import numpy as np
from shutil import copyfile
import copy
from conifer.utils import _ap_include, _json_include, _gcc_opts, _py_executable, copydocstring
from conifer.model import ModelBase
from conifer.backends.common import MultiPrecisionConfig
import logging
logger = logging.getLogger(__name__)

# traversal algorithms
_algorithms = ('treewalk', 'quickscorer', 'blockwise-quickscorer')

class CPPConfig(MultiPrecisionConfig):
  backend = 'cpp'
  _config_fields = MultiPrecisionConfig._config_fields + ['algorithm', 'tau', 'delta']
  _alternates = {**MultiPrecisionConfig._alternates,
                 'algorithm' : ['Algorithm'],
                 'tau'       : ['Tau'],
                 'delta'     : ['Delta'],
                 }
  # default block shape for 'blockwise-quickscorer'
  # NOTE: tau/delta need to be tuned per ensemble and per machine, cf. Table 4 of the QuickScorer paper.
  # NOTE: delta is much smaller than the python backend's, where it is the numpy vectorization width.
  _defaults = {**MultiPrecisionConfig._defaults,
               'algorithm' : 'treewalk',
               'tau'       : 128,
               'delta'     : 16,
               }

  def __init__(self, configDict):
    configDict = dict(configDict)
    for key in ['algorithm', 'tau', 'delta']:
      if configDict.get(key) is None and configDict.get(key.capitalize()) is None:
        configDict[key] = self._defaults[key]
    super(CPPConfig, self).__init__(configDict, validate=True)
    self._extra_validate()

  def _extra_validate(self):
    # TODO: proagate different precisions properly through backend
    # for now enforce that all the precisions are equal
    assert self.input_precision == self.threshold_precision, f'input & threshold precision must be equal, got: {self.input_precision} & {self.threshold_precision}'
    assert self.algorithm in _algorithms, f'Unknown algorithm "{self.algorithm}", expected one of {_algorithms}'
    for key in ('tau', 'delta'):
      val = getattr(self, key)
      assert int(val) > 0, f'{key} must be a positive integer, got: {val}'

class CPPModel(ModelBase):
  def __init__(self, ensembleDict, config, metadata=None):
    super(CPPModel, self).__init__(ensembleDict, config, metadata)
    self.config = CPPConfig(config)
    # block shape the compiled bridge's QuickScorer structures are built for, None until built
    self._qs_block_shape = None

  @copydocstring(ModelBase.write)
  def write(self):
    '''
    Write BDT for CPP backend
    '''

    #######################
    # my_project.json
    #######################
    self.save()

    cfg = self.config
    filedir = os.path.dirname(os.path.abspath(__file__))
    logger.info(f"Writing project to {cfg.output_dir}")

    #######################
    # bridge.cpp
    #######################

    copyfile(f'{filedir}/template/bridge.cpp',
            f"{cfg.output_dir}/bridge_tmp.cpp")

    fin = open(f"{cfg.output_dir}/bridge_tmp.cpp", 'r')
    fout = open(f"{cfg.output_dir}/bridge.cpp", 'w')
    for line in fin.readlines():
      newline = line
      if '// conifer insert typedef' in line:
        
        newline =  "struct BDTConfig : conifer::ConiferConfiguration {\n"
        newline += f"    typedef {cfg.threshold_precision} threshold_t;\n"  
        newline += f"    typedef {cfg.input_precision} input_t;\n"
        newline += f"    typedef {cfg.weight_precision} weight_t;\n"  
        newline += f"    typedef {cfg.score_precision} score_t;\n"
        newline += f"    static constexpr bool useAddTree = false;\n"
        newline += f"{'}'};\n"
      elif 'PYBIND11_MODULE' in line:
        newline = line.replace('conifer_bridge', f'conifer_bridge_{self._stamp}')
      elif '// conifer insert include' in line:
        newline = '#include "ap_fixed.h"' if cfg.any_ap_types() else ''
      fout.write(newline)
    fin.close()
    fout.close()
    os.remove(f"{cfg.output_dir}/bridge_tmp.cpp")

  @copydocstring(ModelBase.compile)
  def compile(self):
    self.write()
    cfg = self.config
    curr_dir = os.getcwd()
    os.chdir(cfg.output_dir)

    # include the ap_ headers, but only if needed (e.g. float/double precision doesn't need them)
    ap_include = ""
    if cfg.any_ap_types():
      ap_include = _ap_include()
      if ap_include is None:
        os.chdir(curr_dir)
        raise Exception("Couldn't find Xilinx ap_ headers. Source the Vivado/Vitis HLS toolchain, or set XILINX_AP_INCLUDE environment variable.")
    #include the JSON headers
    json_include = _json_include()
    if json_include is None:
      os.chdir(curr_dir)
      raise Exception("Couldn't find the JSON headers. Install nlohmman JSON, and set JSON_ROOT")
    # find the conifer.h header
    filedir = os.path.dirname(os.path.abspath(__file__))
    conifer_include = f'-I{filedir}/include/'

    # Do the compile
    cmd = f"g++ -O3 -shared -std=c++14 -fPIC $({_py_executable()} -m pybind11 --includes) {ap_include} {json_include} {conifer_include} {_gcc_opts()} bridge.cpp -o conifer_bridge_{self._stamp}.so"
    logger.debug(f'Compiling with command {cmd}')
    try:
      ret_val = os.system(cmd)
      if ret_val != 0:
        raise Exception(f'Failed to compile project {cfg.project_name}')
    except:
      os.chdir(curr_dir)
      raise Exception(f'Failed to compile project {cfg.project_name}')

    try:
      logger.debug(f'Importing conifer_bridge_{self._stamp} from conifer_bridge_{self._stamp}.so')
      import importlib.util
      spec = importlib.util.spec_from_file_location(f'conifer_bridge_{self._stamp}', f'./conifer_bridge_{self._stamp}.so')
      module = importlib.util.module_from_spec(spec)
      spec.loader.exec_module(module)
      self.bridge = module.BDT(f"{cfg.project_name}.json")
      self._qs_block_shape = None
    except ImportError:
      os.chdir(curr_dir)
      raise Exception("Can't import pybind11 bridge, is it compiled?")
    finally:
      os.chdir(curr_dir)
    if cfg.algorithm != 'treewalk':
      self._init_quickscorer(cfg.algorithm)

  def _init_quickscorer(self, algorithm):
    '''
    Build the QuickScorer structures of the compiled bridge for the block shape of
    `algorithm`, unless they are already built for it. Since tau & delta are runtime
    arguments of the bridge, changing the block shape doesn't recompile the project.
    '''
    cfg = self.config
    # plain quickscorer is the degenerate blocking: one block of all the trees, scored
    # over all the samples at once
    tau, delta = (0, 0) if algorithm == 'quickscorer' else (int(cfg.tau), int(cfg.delta))
    if self._qs_block_shape == (tau, delta):
      return
    max_leaves = max(tree.n_leaves() for trees_class in self.trees for tree in trees_class)
    if max_leaves > 64:
      raise NotImplementedError(f'QuickScorer bitvectors are 64-bit words, but a tree has {max_leaves} leaves')
    self.bridge.init_quickscorer(tau, delta)
    self._qs_block_shape = (tau, delta)

  def decision_function(self, X, trees=False, algorithm=None):
    '''
    Compute the decision function of `X`.

    Parameters
    ----------
    X: array-like of shape (n_samples, n_features)
        Input sample

    trees: bool, optional
        If True, returns the decision function of each tree in the ensemble. Otherwise, returns the sum of the decision function of all trees. Defaults to False.

    algorithm: string, optional
        Traversal algorithm, 'treewalk', 'quickscorer' or 'blockwise-quickscorer'.
        Overrides the 'Algorithm' of the model configuration for this call only.

    Returns
    ----------
    score: ndarray of shape (n_samples, n_classes) or (n_samples,)
    '''
    cfg = self.config
    algorithm = cfg.algorithm if algorithm is None else algorithm
    assert algorithm in _algorithms, f'Unknown algorithm "{algorithm}", expected one of {_algorithms}'

    if algorithm == 'treewalk':
      curr_dir = os.getcwd()
      os.chdir(cfg.output_dir)
      if len(X.shape) == 1:
        y = np.array(self.bridge.decision_function(X))
      elif len(X.shape) == 2:
        y = np.array([self.bridge.decision_function(xi) for xi in X])
      else:
        os.chdir(curr_dir)
        raise Exception(f"Can't handle data shape {X.shape}, expected 1D or 2D shape")
      os.chdir(curr_dir)
    else:
      # (bw)quickscorer, scoring the whole batch in one call
      self._init_quickscorer(algorithm)
      X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
      if len(X.shape) == 1:
        y = np.array(self.bridge.decision_function_batch(X.reshape(1, -1)))[0]
      elif len(X.shape) == 2:
        y = np.array(self.bridge.decision_function_batch(X))
      else:
        raise Exception(f"Can't handle data shape {X.shape}, expected 1D or 2D shape")

    if len(y.shape) == 2 and y.shape[1] == 1:
      y = y.reshape(y.shape[0])
    return y

  def nbytes(self, algorithm=None):
    '''
    Total size in bytes of the QuickScorer traversal data structures
    (cf. Table 1 of the paper), for the block shape of `algorithm`.
    Defaults to the 'Algorithm' of the model configuration, or to plain
    'quickscorer' if that is 'treewalk', which builds no such structures.
    '''
    algorithm = self.config.algorithm if algorithm is None else algorithm
    assert algorithm in _algorithms, f'Unknown algorithm "{algorithm}", expected one of {_algorithms}'
    self._init_quickscorer('quickscorer' if algorithm == 'treewalk' else algorithm)
    return self.bridge.nbytes()

  def build(self):
    raise NotImplementedError

def auto_config():
    config = {'Backend' : 'cpp',
              'ProjectName': 'my_prj',
              'OutputDir': 'my-conifer-prj',
              'Precision': 'ap_fixed<18,8>',
              'Algorithm': 'treewalk'}
    return config

def make_model(ensembleDict, config):
    return CPPModel(ensembleDict, config)
