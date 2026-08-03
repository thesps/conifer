import os
import numpy as np
from shutil import copyfile
from conifer.utils import _ap_include, _json_include, _gcc_opts, _py_executable, copydocstring
from conifer.model import ModelBase
from conifer.backends.common import MultiPrecisionConfig
import logging
logger = logging.getLogger(__name__)

class CPPQSConfig(MultiPrecisionConfig):
  backend = 'cpp_qs'
  _config_fields = MultiPrecisionConfig._config_fields + ['tau', 'delta']
  _alternates = {**MultiPrecisionConfig._alternates,
                 'tau'   : ['Tau'],
                 'delta' : ['Delta'],
                 }
  _defaults = {**MultiPrecisionConfig._defaults,
               'tau'   : None,
               'delta' : None,
               }
  # tau/delta left unset means plain QuickScorer
  _allow_undefined = MultiPrecisionConfig._allow_undefined + ['tau', 'delta']

  def __init__(self, configDict):
    super(CPPQSConfig, self).__init__(configDict, validate=True)
    self._extra_validate()

  def _extra_validate(self):
    # TODO: proagate different precisions properly through backend
    # for now enforce that all the precisions are equal
    assert self.input_precision == self.threshold_precision, f'input & threshold precision must be equal, got: {self.input_precision} & {self.threshold_precision}'
    for key in ('tau', 'delta'):
      val = getattr(self, key, None)
      assert val is None or int(val) > 0, f'{key} must be a positive integer or None, got: {val}'

class CPPQSModel(ModelBase):
  def __init__(self, ensembleDict, config, metadata=None):
    super(CPPQSModel, self).__init__(ensembleDict, config, metadata)
    self.config = CPPQSConfig(config)
    if self.is_oblique():
      raise NotImplementedError(
        'QuickScorer supports only axis-aligned trees (one feature per split), '
        'but this model contains oblique splits')
    max_leaves = max(tree.n_leaves() for trees_class in self.trees for tree in trees_class)
    if max_leaves > 64:
      raise NotImplementedError(
        f'QuickScorer bitvectors are 64-bit words, but a tree has {max_leaves} leaves')

  @copydocstring(ModelBase.write)
  def write(self):
    '''
    Write BDT for cpp_qs backend
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

        newline =  "struct BDTConfig : conifer_qs::ConiferConfiguration {\n"
        newline += f"    typedef {cfg.threshold_precision} threshold_t;\n"
        newline += f"    typedef {cfg.input_precision} input_t;\n"
        newline += f"    typedef {cfg.weight_precision} weight_t;\n"
        newline += f"    typedef {cfg.score_precision} score_t;\n"
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
    # find the conifer_qs.h header
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
      tau = 0 if getattr(cfg, 'tau', None) is None else int(cfg.tau)
      delta = 0 if getattr(cfg, 'delta', None) is None else int(cfg.delta)
      self.bridge = module.BDT(f"{cfg.project_name}.json", tau, delta)
    except ImportError:
      os.chdir(curr_dir)
      raise Exception("Can't import pybind11 bridge, is it compiled?")
    finally:
      os.chdir(curr_dir)

  @copydocstring(ModelBase.decision_function)
  def decision_function(self, X, trees=False):
    X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
    if len(X.shape) == 1:
      y = np.array(self.bridge.decision_function(X.reshape(1, -1)))[0]
    elif len(X.shape) == 2:
      y = np.array(self.bridge.decision_function(X))
    else:
      raise Exception(f"Can't handle data shape {X.shape}, expected 1D or 2D shape")
    if len(y.shape) == 2 and y.shape[1] == 1:
      y = y.reshape(y.shape[0])
    return y

  def nbytes(self):
    '''Total size in bytes of the QuickScorer traversal data structures (cf. Table 1 of the paper)'''
    return self.bridge.nbytes()

  def build(self):
    raise NotImplementedError

def auto_config():
    config = {'Backend' : 'cpp_qs',
              'ProjectName': 'my_prj',
              'OutputDir': 'my-conifer-prj',
              'Precision': 'ap_fixed<18,8>'}
    return config

def make_model(ensembleDict, config):
    return CPPQSModel(ensembleDict, config)
