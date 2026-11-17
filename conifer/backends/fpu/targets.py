'''
FPU targets: what an FPUBuilder builds around the VHDL core FPU_df (core.py), selected by the 'interface' config key.
  - 'hls' (HLSTarget): the HLS FPU, with FPU_df as an RTL blackbox, built for a board (Zynq or Alveo, see conifer.backends.boards)
'''
import json
import math
import os
import shutil
import numpy as np
import logging
logger = logging.getLogger(__name__)
from conifer.backends.common import _TOOLS, get_hls, get_hls_build_command
from conifer.backends.boards import get_builder, ZynqConfig, AlveoConfig
from conifer.backends.fpu.core import _VHDLWriter, _resolve_type, _fill_template

class FPUTarget:
  '''
  Base of the FPU targets. A target writes and builds its project around the core, which the FPUBuilder writes
  first. It reaches the FPU configuration and project metadata through the builder.
  '''
  def __init__(self, builder):
    self.builder = builder
    self.cfg = builder.cfg

  def write(self):
    raise NotImplementedError

  def build(self, **kwargs):
    raise NotImplementedError

  def package(self):
    raise NotImplementedError

class _HLSBlackboxWriter:
  '''
  Write the RTL blackbox wrapping FPU_df for HLS. From the templates in src and src/vhdl:
    - vhdl/FPU_df_hls.vhd: FPU_df with the HLS port names, copied
    - FPU_df_bb.v: the Verilog shim from the HLS blackbox ports to the VHDL FPU_df_hls, with the ports written
    - fpu_bb.h, fpu_bb.cpp: the C declaration and model of the blackbox, copied
  and FPU_df_bb.json, the HLS blackbox description.
  The blackbox takes each feature as a scalar, and each Tree Engine's node and roots memories as arrays.
  The macros listing the features and Tree Engines for fpu_bb.h are written to parameters.h with parameter_macros.
  '''
  vhdl_files = ['FPU_df_hls.vhd']
  # out of context synthesis of FPU_df_hls with the project part and clock
  vhdl_scripts = ['synth_ooc.tcl']

  def __init__(self, cfg):
    '''cfg: FPUConfig'''
    self.core = _VHDLWriter(cfg)

  def all_vhdl_files(self):
    '''The VHDL sources of the blackbox in compile order: the core's, then FPU_df_hls'''
    return self.core.vhdl_files + self.vhdl_files

  def parameter_macros(self):
    '''The lines for parameters.h listing the features and Tree Engines, for fpu_bb.h'''
    each = lambda n: ' '.join(f'M({i})' for i in range(n))
    return [f'#define FPU_FOR_EACH_FEATURE(M) {each(self.core.nf)}', f'#define FPU_FOR_EACH_TE(M) {each(self.core.nte)}']

  def _memories(self):
    '''The blackbox memory arguments, in argument order: (name, address width, data width) of roots_i then nodes_i'''
    c = self.core
    return ([(f'roots_{i}', c.rawidth, c.addrbits) for i in range(c.nte)] +
            [(f'nodes_{i}', c.nawidth, c.nodebits) for i in range(c.nte)])

  def write_shim(self, fname):
    '''
    Write the Verilog shim FPU_df_bb from the template src/FPU_df_bb.v, inserting:
      - ports: those HLS connects to the blackbox arguments. x_i for each feature, and for each memory roots_i, nodes_i:
          output [AW-1:0] roots_0_address0, output roots_0_ce0, input [DW-1:0] roots_0_q0
      - connections: the ports to FPU_df_hls, which takes each of X, nodes_addr, nodes_en, ... as one vector
          with element i at bits [(i+1)*W-1 : i*W]. So the ports are concatenated last first:
          .nodes_addr({nodes_1_address0, nodes_0_address0})
    '''
    c = self.core
    ports = [f'input [31:0] x_{i}' for i in range(c.nf)]
    for name, awidth, dwidth in self._memories():
      ports += [f'output [{awidth-1}:0] {name}_address0',
                f'output {name}_ce0',
                f'input [{dwidth-1}:0] {name}_q0']

    def concat(signals):
      return '{' + ', '.join(reversed(signals)) + '}'

    connections = {'X' : concat([f'x_{i}[{c.tbits-1}:0]' for i in range(c.nf)])}
    for mem in ['nodes', 'roots']:
      for vhdl_port, verilog_port in [('addr', 'address0'), ('en', 'ce0'), ('q', 'q0')]:
        connections[f'{mem}_{vhdl_port}'] = concat([f'{mem}_{i}_{verilog_port}' for i in range(c.nte)])

    filedir = os.path.dirname(os.path.abspath(__file__))
    _fill_template(f'{filedir}/src/FPU_df_bb.v', fname,
                   {'ports' : ',\n'.join(f'  {p}' for p in ports),
                    'connections' : ',\n'.join(f'  .{port}({signal})' for port, signal in connections.items())})

  def write_json(self, fname):
    '''
    Write the HLS blackbox description FPU_df_bb.json from the template src/FPU_df_bb.json, inserting c_parameters:
    each argument of FPU_df_bb in order, and the shim ports HLS connects it to
    '''
    def scalar_in(name):
      return {'c_name' : name, 'c_port_direction' : 'in', 'rtl_ports' : {'data_read_in' : name}}

    def scalar_out(name):
      return {'c_name' : name, 'c_port_direction' : 'out',
              'rtl_ports' : {'data_write_out' : name, 'data_write_valid' : f'{name}_ap_vld'}}

    def memory(name):
      return {'c_name' : name, 'c_port_direction' : 'in', 'RAM_type' : 'RAM_1P',
              'rtl_ports' : {'RAM_address' : f'{name}_address0', 'RAM_clock_enable' : f'{name}_ce0', 'RAM_data_read_in' : f'{name}_q0'}}

    filedir = os.path.dirname(os.path.abspath(__file__))
    with open(f'{filedir}/src/FPU_df_bb.json') as f:
      bb = json.load(f)
    bb['c_parameters'] = ([scalar_in(f'x_{i}') for i in range(self.core.nf)] +
                          [scalar_out('y')] +
                          [memory(name) for name, _, _ in self._memories()])
    with open(fname, 'w') as f:
      json.dump(bb, f, indent=2)

  def write(self, odir):
    '''Write the blackbox to odir, where the core is already written to {odir}/vhdl'''
    filedir = os.path.dirname(os.path.abspath(__file__))
    _VHDLWriter.copy_sources(f'{odir}/vhdl', self.vhdl_files + self.vhdl_scripts)
    for f in ['fpu_bb.h', 'fpu_bb.cpp']:
      shutil.copyfile(f'{filedir}/src/{f}', f'{odir}/{f}')
    self.write_shim(f'{odir}/FPU_df_bb.v')
    self.write_json(f'{odir}/FPU_df_bb.json')

class HLSTarget(FPUTarget):
  '''
  The HLS FPU (src/fpu.cpp), with the VHDL FPU_df as an RTL blackbox, for a board: built with Vitis HLS, then by
  the board's builder (Vivado for Zynq boards, v++ for Alveo)
  '''
  top_names = {ZynqConfig  : 'FPU_Zynq',
               AlveoConfig : 'FPU_Alveo',
               }

  def __init__(self, builder):
    super().__init__(builder)
    self.top_name = self.top_names.get(type(self.cfg.board_config), None)
    self.board_builder = get_builder(self.cfg, self.cfg.board_config, top_name=self.top_name, ip_name=f'{self.top_name}_0')
    self._csim, self._cosim = False, False

  def write_params(self):
    b = self.builder
    with open(f'{b.output_dir}/parameters.h', 'w') as f:
      f.write('#ifndef CONIFER_FPU_PARAMS_H_\n#define CONIFER_FPU_PARAMS_H_\n')
      f.write('#include "fpu.h"\n')
      f.write(f'typedef {_resolve_type(b.threshold_type)} T;\n')
      f.write(f'typedef {_resolve_type(b.score_type)} U;\n')
      f.write(f'static const int NFEATURES={b.features};\n')
      f.write(f'static const int NTE={b.tree_engines};\n')
      f.write(f'static const int NNODES={b.nodes};\n')
      f.write(f'static const int NROOTS={b.roots};\n')
      f.write(f'static const int ADDRBITS={math.ceil(np.log2(b.nodes))+1};\n')
      f.write(f'static const int FEATBITS={math.ceil(np.log2(b.features))+1};\n')
      f.write(f'static const int NCLASSES={1};\n')
      f.write(f'static const int CLASSBITS={1};\n')
      f.write(f'static const bool SCALER={"true" if b.dynamic_scaler else "false"};\n')
      f.write(f'typedef DecisionNode<T,U,FEATBITS,ADDRBITS,CLASSBITS> DN;\n')
      f.write('// the features and Tree Engines, as lists of M(i) for the VHDL blackbox arguments (see fpu_bb.h)\n')
      for line in _HLSBlackboxWriter(self.cfg).parameter_macros():
        f.write(line + '\n')
      info = b._info()
      info_fmt = info.replace('"', r'\"')
      f.write(f'static const char* theInfo = "{info_fmt}";\n')
      f.write(f'static const int theInfoLength = {len(info)};\n')
      f.write('#endif\n')

  def write_tcl(self):
    import conifer
    b = self.builder
    with open(f'{b.output_dir}/hls_parameters.tcl', 'w') as f:
      f.write(f'set prj_name {b.project_name}\n')
      f.write(f'set top {self.top_name}\n')
      f.write(f'set part {self.cfg.board_config.xilinx_part}\n')
      f.write(f'set clock_period {b.clock_period}\n')
      f.write(f'set flow_target {self.board_builder.get_flow_target()}\n')
      f.write(f'set export_format {self.board_builder.get_export_format()}\n')
      f.write(f'set m_axi_addr64 {str(self.board_builder.get_maxi64()).lower()}\n')
      f.write(f'set version {conifer.__version__.major}.{conifer.__version__.minor}\n')
      f.write(f'set csim {int(self._csim)}\n')
      f.write(f'set cosim {int(self._cosim)}\n')
      f.write(f'set vhdl_files {{{" ".join(_HLSBlackboxWriter(self.cfg).all_vhdl_files())}}}\n')
      if self._cosim:
        with open(f'{b.output_dir}/tb_data/X.dat') as fX:
          batch_size, n_features = [int(v) for v in fX.readline().split()]
        f.write(f'set cosim_X_depth {max(batch_size * n_features, 1)}\n')
        f.write(f'set cosim_y_depth {max(batch_size, 1)}\n')
        f.write(f'set cosim_info_depth {len(b._info())}\n')

  def write(self):
    '''Write the HLS project, around the core already written'''
    filedir = os.path.dirname(os.path.abspath(__file__))
    od = self.builder.output_dir
    for f in ['build_hls.tcl', 'fpu.cpp', 'fpu.h', 'fpu_tb.cpp']:
      shutil.copyfile(f'{filedir}/src/{f}', f'{od}/{f}')
    self.write_params()
    self.write_tcl()
    _HLSBlackboxWriter(self.cfg).write(od)

  def build(self, csynth=True, bitfile=True, csim=False, cosim=False, **build_kwargs):
    '''See FPUBuilder.build'''
    b = self.builder
    self._csim, self._cosim = csim, cosim
    if csim or cosim:
      assert os.path.exists(f'{b.output_dir}/tb_data/X.dat'), 'No testbench data found, call write_testbench_data first'
    b.write()
    cwd = os.getcwd()
    os.chdir(b.output_dir)
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

targets = {'hls' : HLSTarget}
