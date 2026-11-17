'''
FPU targets: what an FPUBuilder builds around the VHDL core FPU_df (core.py), selected by the 'interface' config key.
  - 'hls' (HLSTarget): the HLS FPU, with FPU_df as an RTL blackbox, built for a board (Zynq or Alveo, see conifer.backends.boards)
  - 'rtl' (RTLTarget): FPU_rtl, the FPU with its memories and a configuration port, for RTL designs on a Xilinx part
Other targets can be added with register_target.
'''
import json
import math
import os
import shutil
import warnings
import zipfile
import numpy as np
import logging
logger = logging.getLogger(__name__)
from conifer.backends.common import _TOOLS, get_hls, get_hls_build_command
from conifer.backends.boards import get_builder, ZynqConfig, AlveoConfig
from conifer.backends.fpu.core import _VHDLWriter, _resolve_type, _fill_template
from conifer.backends.fpu.memory_map import FPUMemoryMap

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

  def write_testbench_data(self, model, X):
    '''Write any testbench data of the target, after FPUBuilder.write_testbench_data writes that of the core'''
    pass

  def write_synth_parameters(self, top, part, vhdl_files, clocks, xdc_files=[]):
    '''
    Write vhdl/synth_parameters.tcl, the parameters of the out of context synthesis script vhdl/synth_ooc.tcl
    Parameters
    ----------
    top: the top level entity
    part: the Xilinx part
    vhdl_files: the VHDL sources, in compile order
    clocks: dictionary of clock port name to period (ns). The first is the FPU clock
    xdc_files: constraint files, in the vhdl directory
    '''
    with open(f'{self.builder.output_dir}/vhdl/synth_parameters.tcl', 'w') as f:
      f.write(f'set top {top}\n')
      f.write(f'set part {part}\n')
      f.write(f'set clocks {{{" ".join(f"{k} {v}" for k, v in clocks.items())}}}\n')
      f.write(f'set vhdl_files {{{" ".join(vhdl_files)}}}\n')
      f.write(f'set xdc_files {{{" ".join(xdc_files)}}}\n')

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
    self.write_synth_parameters('FPU_df_hls', self.cfg.board_config.xilinx_part, _HLSBlackboxWriter(self.cfg).all_vhdl_files(), {'ap_clk' : self.builder.clock_period})

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

class RTLTarget(FPUTarget):
  '''
  FPU_rtl (src/vhdl/FPU_rtl.vhd): the FPU with its Tree Engine memories, and a configuration port to load the model
  from another clock domain, independent of any bus protocol. For RTL designs, on the config's xilinx_part.
  The memory map of the configuration port is in fpu_rtl_pkg.vhd, and its Python mirror in memory_map.py.
  Without the dynamic scaler: the inputs are scaled to the threshold type by the design.
  Writes to {output_dir}/vhdl: the core, the FPU_rtl sources, the constraints FPU_rtl_cdc.xdc, the testbench
  FPU_rtl_tb, and the parameters for synth_ooc.tcl.
  '''
  vhdl_files = ['fpu_rtl_pkg.vhd', 'FPU_mem.vhd', 'FPU_rtl.vhd']
  vhdl_tb_files = ['FPU_rtl_tb.vhd']
  # the out of context synthesis script, shared with the hls interface
  vhdl_scripts = ['synth_ooc.tcl']
  xdc_files = ['FPU_rtl_cdc.xdc']
  top = 'FPU_rtl'
  tb = 'FPU_rtl_tb'
  # the configuration clock period for out of context synthesis (ns): that of ipbus
  cfg_clock_period = 32
  # the build options of the HLS target, unused here
  _hls_build_options = {'csynth' : True, 'bitfile' : True, 'csim' : False, 'cosim' : False}

  def __init__(self, builder):
    super().__init__(builder)
    assert not self.cfg.dynamic_scaler, f'The {self.cfg.interface} FPU has no dynamic scaler, set dynamic_scaler to False'

  def target_vhdl_files(self):
    '''The VHDL sources of the target, in compile order after the core's: names in the vhdl directory, or paths'''
    return self.vhdl_files

  def all_vhdl_files(self):
    '''The VHDL sources in compile order: the core's, then the target's'''
    return _VHDLWriter.vhdl_files + self.target_vhdl_files()

  def clocks(self):
    '''The clock ports and their periods (ns) for out of context synthesis: the FPU clock first'''
    return {'clk' : self.builder.clock_period, 'cfg_clk' : self.cfg_clock_period}

  def write(self):
    '''Write the FPU_rtl sources, testbench, constraints and synthesis parameters, around the core already written'''
    filedir = os.path.dirname(os.path.abspath(__file__))
    vdir = f'{self.builder.output_dir}/vhdl'
    _VHDLWriter.copy_sources(vdir, self.vhdl_files + self.vhdl_tb_files + self.vhdl_scripts)
    _fill_template(f'{filedir}/src/vhdl/FPU_rtl_cdc.xdc', f'{vdir}/FPU_rtl_cdc.xdc',
                   {'max_delay' : f'set fpu_rtl_cdc_max_delay {self.builder.clock_period}'})
    self.write_synth_parameters(self.top, self.cfg.xilinx_part, self.all_vhdl_files(), self.clocks(), self.xdc_files)

  def write_testbench_data(self, model, X):
    '''Write the configuration writes loading the model to tb_data/cfg_writes.dat, for FPU_rtl_tb'''
    writes = FPUMemoryMap(self.cfg).writes(*model.pack())
    np.savetxt(f'{self.builder.output_dir}/tb_data/cfg_writes.dat', writes, fmt='%08x')

  def build(self, sim=False, synth=True, **build_kwargs):
    '''
    Build the RTL FPU. Writes the project, then:
    Parameters
    ----------
    sim: boolean (optional)
      Simulate the testbench (FPU_rtl_tb) with xsim, writing y.dat, readback.dat and info.dat to the vhdl directory.
      Call write_testbench_data first.
    synth: boolean (optional)
      Synthesize the top (FPU_rtl) out of context with Vivado, writing utilization.rpt, timing.rpt and cdc.rpt to the vhdl directory
    The options of the HLS target (csynth, bitfile, csim, cosim) are ignored, with a warning if not their default
    '''
    ignored = [k for k, v in build_kwargs.items() if k in self._hls_build_options and v != self._hls_build_options[k]]
    if ignored:
      warnings.warn(f'The build options {ignored} are for the hls interface, and are ignored by the {self.cfg.interface} interface')
    unknown = [k for k in build_kwargs if k not in self._hls_build_options]
    assert not unknown, f'Unknown build options {unknown} for the {self.cfg.interface} interface'
    b = self.builder
    if sim:
      assert os.path.exists(f'{b.output_dir}/tb_data/cfg_writes.dat'), 'No testbench data found, call write_testbench_data first'
    b.write()
    success = True
    vdir = f'{b.output_dir}/vhdl'
    if sim:
      cmd = f'sh {vdir}/xsim_tb.sh {self.tb} {" ".join(self.target_vhdl_files())} > {vdir}/xsim.log 2>&1'
      logger.info(f'Simulating the RTL FPU with command "{cmd}"')
      success = success and os.system(cmd) == 0
    if success and synth:
      cmd = f'cd {vdir} && vivado -mode batch -nojournal -nolog -source synth_ooc.tcl > synth.log 2>&1'
      logger.info(f'Synthesizing the RTL FPU with command "{cmd}"')
      success = success and os.system(cmd) == 0
    return success

  def package(self):
    '''Zip the RTL sources and constraints, with the configuration, to {output_dir}/{project_name}.zip'''
    b = self.builder
    with zipfile.ZipFile(f'{b.output_dir}/{b.project_name}.zip', 'w') as z:
      for f in self.all_vhdl_files() + self.xdc_files:
        z.write(os.path.join(b.output_dir, 'vhdl', f), os.path.basename(f))
      z.write(f'{b.output_dir}/{b.project_name}.json', f'{b.project_name}.json')

targets = {'hls' : HLSTarget, 'rtl' : RTLTarget}

def register_target(name : str, target):
  '''
  Register a new FPU target, selected by the 'interface' config key

  Parameters
  ----------
  name: str
      The interface name
  target: subclass of FPUTarget
  '''
  assert name not in targets, f'An FPU target with interface "{name}" is already registered'
  assert issubclass(target, FPUTarget), f'Expected a subclass of FPUTarget, got {target}'
  targets[name] = target
