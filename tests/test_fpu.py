'''
Tests of the FPU backend without place and route, so they run in a few minutes. Need Vivado and Vitis HLS in the PATH.
The reference is the cpp backend: with power of two FPU scales, and fixed point types matching those of the FPU
(threshold and score types truncated and rounded as the FPU scaling), it is bit exact with the FPU.
'''
import os
import re
import datetime
import subprocess
import numpy as np
import pytest
import conifer
from conifer.backends.fpu import FPUBuilder
from sklearn.datasets import make_hastie_10_2
from sklearn.ensemble import GradientBoostingClassifier

# FPU threshold and score types are 16 bit integers. Scale the thresholds and scores by these powers of two to fill them
T_FRAC, U_FRAC = 11, 8
FPU = {'tree_engines' : 4, 'nodes' : 128, 'roots' : 8, 'features' : 10, 'threshold_type' : 16, 'score_type' : 16}

def prj_dir(name):
  return os.path.abspath(f'prj_fpu_{name}_{int(datetime.datetime.now().timestamp())}')

def run(cmd, cwd):
  p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
  assert p.returncode == 0, f'{" ".join(cmd)} failed:\n{(p.stdout + p.stderr)[-3000:]}'
  return p.stdout

@pytest.fixture(scope='module')
def model():
  '''A BDT, test inputs, and the reference predictions. The FPU does not add an initial prediction, so it is 0'''
  X, y = make_hastie_10_2(random_state=0)
  clf = GradientBoostingClassifier(n_estimators=20, learning_rate=1.0, max_depth=3, random_state=0, init='zero').fit(X[:2000], y[:2000])
  X = X[2000:].astype('float32')
  cfg = {'Backend' : 'cpp', 'OutputDir' : prj_dir('cpp'), 'ProjectName' : 'cpp',
         'InputPrecision' : f'ap_fixed<16,{16-T_FRAC},AP_TRN,AP_WRAP>', 'ScorePrecision' : f'ap_fixed<16,{16-U_FRAC},AP_RND_CONV,AP_WRAP>'}
  reference = conifer.converters.convert_from_sklearn(clf, cfg)
  reference.compile()
  return clf, X, reference.decision_function(X).reshape(-1)

def fpu_project(clf, X, dynamic_scaler):
  '''Write an FPU project for the model, with testbench data for the inputs X'''
  cfg = conifer.backends.fpu.auto_config()
  cfg['FPU'].update({**FPU, 'dynamic_scaler' : dynamic_scaler})
  fpu_model = conifer.converters.convert_from_sklearn(clf, cfg)
  fpu_model.set_scales(2**T_FRAC, 2**U_FRAC)
  builder = FPUBuilder({**FPUBuilder.default_cfg(), **FPU, 'dynamic_scaler' : dynamic_scaler, 'project_name' : 'fpu',
                        'board' : 'xilinx_u200_gen3x16_xdma_2_202110_1', 'clock_period' : 3.33,
                        'output_dir' : prj_dir(f'scaler{int(dynamic_scaler)}')})
  os.makedirs(builder.output_dir)
  # without the dynamic scaler, the inputs are scaled to the threshold type before sending them to the FPU
  builder.write_testbench_data(fpu_model, X if dynamic_scaler else np.floor(X * 2**T_FRAC).astype(int))
  builder.write()
  return builder

@pytest.fixture(scope='module')
def vhdl_project(model):
  return fpu_project(*model[:2], dynamic_scaler=False)

def test_vhdl_sim(model, vhdl_project):
  '''Simulate the VHDL decision function FPU_df'''
  run(['sh', 'vhdl/xsim_tb.sh'], cwd=vhdl_project.output_dir)
  y = np.loadtxt(f'{vhdl_project.output_dir}/vhdl/y.dat') / 2**U_FRAC
  np.testing.assert_array_equal(y, model[2])

def test_vhdl_synth(vhdl_project):
  '''Out of context synthesis of the VHDL decision function as integrated in HLS (FPU_df_hls)'''
  log = run(['vivado', '-mode', 'batch', '-nojournal', '-source', 'vhdl/synth_ooc.tcl'], cwd=vhdl_project.output_dir)
  assert 'CRITICAL WARNING' not in log, re.findall(r'CRITICAL WARNING.*', log)
  assert '[Synth 8-327]' not in log, 'Latches inferred'
  wns = float(re.search(r'FPU_df_hls WNS (\S+) ns', log).group(1))
  assert wns >= 0, f'Post synthesis timing not met, WNS {wns} ns'

def test_hls_csim_cosim(model):
  '''C Simulation and Cosimulation of the HLS FPU with the VHDL decision function, with the dynamic scaler'''
  clf, X, y_ref = model
  # fewer samples, as cosimulation is slower than the VHDL simulation
  X, y_ref = X[:32], y_ref[:32]
  builder = fpu_project(clf, X, dynamic_scaler=True)
  assert builder.build(csynth=True, bitfile=False, csim=True, cosim=True), f'HLS build failed, see {builder.output_dir}/hls_build.log'
  odir = builder.output_dir
  assert 'C/RTL co-simulation finished: PASS' in open(f'{odir}/hls_build.log').read()
  for tb_dir in [f'{odir}/fpu/solution1/csim/build/tb_data', f'{odir}/fpu/solution1/sim/wrapc/tb_data']:
    # predictions are written as float32
    np.testing.assert_array_equal(np.loadtxt(f'{tb_dir}/y.dat', dtype='float32'), y_ref.astype('float32'), err_msg=tb_dir)
    # the model read back from the FPU is the one loaded
    for f in ['nodes', 'roots']:
      np.testing.assert_array_equal(np.loadtxt(f'{tb_dir}/{f}_out.dat'), np.loadtxt(f'{odir}/tb_data/{f}.dat'), err_msg=f'{tb_dir} {f}')
