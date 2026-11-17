'''
Tests of the RTL FPU (the rtl interface: FPU_rtl), without place and route. Need Vivado in the PATH.
The model, inputs and reference predictions are those of test_fpu.py.
'''
import os
import re
import json
import numpy as np
import pytest
import conifer
from conifer.backends.fpu import FPUBuilder, FPUMemoryMap, RTLDriver
from tests.test_fpu import model, prj_dir, FPU, T_FRAC, U_FRAC

PART = 'xcu200-fsgd2104-2-e'

def rtl_cfg(**kwargs):
  return {**FPUBuilder.default_cfg(), **FPU, 'dynamic_scaler' : False, 'interface' : 'rtl', 'xilinx_part' : PART,
          'clock_period' : 3.33, 'project_name' : 'fpu', **kwargs}

def fpu_model(clf):
  cfg = conifer.backends.fpu.auto_config()
  cfg['FPU'].update({**FPU, 'dynamic_scaler' : False})
  fpu_model = conifer.converters.convert_from_sklearn(clf, cfg)
  fpu_model.set_scales(2**T_FRAC, 2**U_FRAC)
  return fpu_model

@pytest.fixture(scope='module')
def rtl_project(model):
  clf, X, _ = model
  builder = FPUBuilder(rtl_cfg(output_dir=prj_dir('rtl')))
  os.makedirs(builder.output_dir)
  builder.write_testbench_data(fpu_model(clf), np.floor(X * 2**T_FRAC).astype(int))
  return builder

def hex_words(f):
  return [int(v, 16) for v in np.loadtxt(f, dtype=str, ndmin=1)]

def test_rtl_sim(model, rtl_project):
  '''Load the model through the configuration port, read it back, and run inference, in simulation'''
  assert rtl_project.build(sim=True, synth=False), f'See {rtl_project.output_dir}/vhdl/xsim.log'
  vdir, tdir = f'{rtl_project.output_dir}/vhdl', f'{rtl_project.output_dir}/tb_data'
  assert 'configuration writes, 0 errors' in open(f'{vdir}/xsim.log').read()
  np.testing.assert_array_equal(np.loadtxt(f'{vdir}/y.dat') / 2**U_FRAC, model[2])
  writes = np.loadtxt(f'{tdir}/cfg_writes.dat', dtype=str)
  assert hex_words(f'{vdir}/readback.dat') == [int(v, 16) for v in writes[:, 1]]
  assert hex_words(f'{vdir}/info.dat') == list(FPUMemoryMap(rtl_project.cfg).info().values())

def test_rtl_synth(rtl_project):
  '''Out of context synthesis of FPU_rtl, with asynchronous FPU and configuration clocks'''
  assert rtl_project.build(sim=False, synth=True), f'See {rtl_project.output_dir}/vhdl/synth.log'
  vdir = f'{rtl_project.output_dir}/vhdl'
  log = open(f'{vdir}/synth.log').read()
  assert 'CRITICAL WARNING' not in log, re.findall(r'CRITICAL WARNING.*', log)
  assert '[Synth 8-327]' not in log, 'Latches inferred'
  wns = float(re.search(r'FPU_rtl WNS (\S+) ns', log).group(1))
  assert wns >= 0, f'Post synthesis timing not met, WNS {wns} ns'
  # every clock domain crossing is constrained, and none is unsafe
  crossings = [l for l in open(f'{vdir}/cdc.rpt') if re.match(r'\s+\d+\s+CDC-\d+', l)]
  assert crossings and all('Max Delay Datapath Only' in l for l in crossings), 'Unconstrained clock domain crossings'
  assert not any('Critical' in l for l in crossings), 'Unsafe clock domain crossings'

def test_rtl_config_errors(tmp_path):
  '''The rtl interface takes xilinx_part, not board, and has no dynamic scaler'''
  for bad, msg in [({'board' : 'pynq-z2'}, 'not board'), ({'xilinx_part' : None}, 'set xilinx_part'),
                   ({'dynamic_scaler' : True}, 'no dynamic scaler')]:
    with pytest.raises(AssertionError, match=msg):
      FPUBuilder(rtl_cfg(output_dir=str(tmp_path), **bad))
  with pytest.raises(AssertionError, match='not xilinx_part'):
    FPUBuilder({**FPUBuilder.default_cfg(), 'xilinx_part' : PART, 'output_dir' : str(tmp_path)})
  # the hls build options are ignored with a warning
  with pytest.warns(UserWarning, match='csim'):
    assert FPUBuilder(rtl_cfg(output_dir=str(tmp_path))).build(csim=True, sim=False, synth=False)

def test_rtl_driver(model, tmp_path):
  '''RTLDriver loads and verifies a model over a bus, here a dictionary with the INFO words'''
  builder = FPUBuilder(rtl_cfg(output_dir=str(tmp_path)))
  fpu_model_ = fpu_model(model[0])
  mm = FPUMemoryMap(builder.cfg)
  bus = dict(zip(mm.info_addresses(), mm.info().values()))
  driver = RTLDriver(bus.get, bus.__setitem__)
  assert driver.config['tree_engines'] == FPU['tree_engines']
  fpu_model_.attach_device(driver)
  writes = mm.writes(*fpu_model_.pack())
  assert all(bus[int(a)] == int(d) for a, d in writes)
