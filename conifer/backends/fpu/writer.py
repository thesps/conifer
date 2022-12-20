import copy
import math
import numpy as np
import json
import shutil
import os
import zipfile
import logging
logger = logging.getLogger(__name__)

def _pack_node(tree, inode, cfg, scale):
  fields = np.zeros(7, dtype='int')
  t = tree['threshold'][inode] * scale
  fields[0] = t
  v = tree['value'][inode] * scale
  fields[1] = v
  fields[2] = tree['feature'][inode]
  fields[3] = tree['children_left'][inode]
  fields[4] = tree['children_right'][inode]
  fields[5] = 0 # class ID
  fields[6] = tree['feature'][inode] == -2
  return fields

def write(model):

  model.save()
  ensembleDict = copy.deepcopy(model._ensembleDict)
  cfg = copy.deepcopy(model.config)
  logger.info(f"Writing project to {cfg['OutputDir']}")
  fpu_cfg = cfg['FPU']

  dtype = cfg['Precision']
  dtype = dtype.replace('ap_fixed<', '').replace('>', '')
  dtype_n = int(dtype.split(',')[0].strip()) # total number of bits
  dtype_int = int(dtype.split(',')[1].strip()) # number of integer bits
  dtype_frac = dtype_n - dtype_int # number of fractional bits
  scale = 2**dtype_frac

  assert ensembleDict['n_trees'] <= fpu_cfg['tree_engines']
  assert ensembleDict['n_classes'] == 2

  # TODO: '7' is the number of fields of each tree, remove the magic number
  nodes = np.zeros(shape=(fpu_cfg['tree_engines'], fpu_cfg['nodes'], 7), dtype='int')
  for it, trees_c in enumerate(ensembleDict['trees']):
    for tree in trees_c:
      for inode in range(len(tree['threshold'])):
        nodes[it][inode] = _pack_node(tree, inode, fpu_cfg, scale)
  
  scales = np.ones(shape=(fpu_cfg['features'])) * 1024

  outcfg = {'nodes' : nodes.flatten().tolist(), 'scales' : scales.flatten().tolist()}
  with open(f'{cfg["OutputDir"]}/fpu_settings.json','w') as f:
    json.dump(outcfg, f)

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

  _cfg_defaults = {
    'part' : 'xc7z020clg400-1',
    'project_name' : 'conifer_fpu',
    'output_dir' : 'conifer_fpu_prj',
    'nodes' : 512,
    'tree_engines' : 64,
    'features' : 16,
    'threshold_type' : 16,
    'score_type' : 16,
    'dynamic_scaler' : True
  }

  def __init__(self, cfg):
    self.cfg = cfg
    for key, value in FPUBuilder._cfg_defaults.items():
      setattr(self, key, cfg.get(key, value))
    self.output_dir = os.path.abspath(self.output_dir)

  def default_cfg():
    cfg = {}
    for key, value in FPUBuilder._cfg_defaults.items():
      cfg[key] = value
    return cfg

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
      info = copy.deepcopy(self.cfg)
      info['version'] = conifer.__version__
      info = json.dumps(info).replace('"', r'\"')
      f.write(f'static const char* theInfo = "{info}";\n')
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
      json.dump(self.cfg, f)
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