import os
import numpy as np
import json
from shutil import copyfile

def write(ensemble_dict, cfg):
  '''
  Write BDT for CPP backend
  '''

  filedir = os.path.dirname(os.path.abspath(__file__))
  os.makedirs(f"{cfg['OutputDir']}")

  with open(f"{cfg['OutputDir']}/{cfg['ProjectName']}.json", 'w') as f:
    json.dump(ensemble_dict, f)
    f.close()

  copyfile(f'{filedir}/template/bridge.cpp',
           f"{cfg['OutputDir']}/bridge_tmp.cpp")

  fin = open(f"{cfg['OutputDir']}/bridge_tmp.cpp", 'r')
  fout = open(f"{cfg['OutputDir']}/bridge.cpp", 'w')
  for line in fin.readlines():
    newline = line
    if '// conifer insert typedef' in line:
      newline =  f"typedef {cfg['Precision']} T;\n"
      newline += f"typedef {cfg['Precision']} U;\n"
    fout.write(newline)
  fin.close()
  fout.close()
  os.remove(f"{cfg['OutputDir']}/bridge_tmp.cpp")

def sim_compile(cfg):
  curr_dir = os.getcwd()
  os.chdir(cfg['OutputDir'])
  cmd = "g++ -O3 -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) -I/cvmfs/cms.cern.ch/slc7_amd64_gcc900/external/hls/2019.08/include/ -I/cvmfs/cms.cern.ch/slc7_amd64_gcc900/external/json/3.10.2-8cd3895745e2546d082bb9980a719047/include/nlohmann -I../../conifer/backends/cpp/include bridge.cpp -o conifer_bridge.so"
  try:
    ret_val = os.system(cmd)
    if ret_val != 0:
      raise Exception(f'Failed to compile project {cfg["ProjectName"]}')
  finally:
    os.chdir(curr_dir)

def decision_function(X, cfg, trees=False):
  curr_dir = os.getcwd()
  os.chdir(cfg['OutputDir'])
  try:
    import importlib.util
    spec = importlib.util.spec_from_file_location('conifer_bridge', './conifer_bridge.so')
    conifer_bridge = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(conifer_bridge)
  except ImportError:
    os.chdir(curr_dir)
    raise Exception("Can't import pybind11 bridge, is it compiled?")
  bdt = conifer_bridge.BDT(f"{cfg['ProjectName']}.json")
  if len(X.shape) == 1:
    y = np.array(bdt.decision_function(X))
  elif len(X.shape) == 2:
    y = np.array([bdt.decision_function(xi) for xi in X])
  else:
    raise Exception(f"Can't handle data shape {X.shape}, expected 1D or 2D shape")
  os.chdir(curr_dir)
  return y

def build():
  raise NotImplementedError

def auto_config():
    config = {'ProjectName': 'my_prj',
              'OutputDir': 'my-conifer-prj',
              'Precision': 'ap_fixed<18,8>'}
    return config
