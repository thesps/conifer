import os
import sys
from shutil import copyfile
import numpy as np
from enum import Enum
import copy
import datetime
import logging
logger = logging.getLogger(__name__)

class Simulators(Enum):
   modelsim = 0
   xsim = 1
   ghdl = 2

def write(model):

  ensembleDict = copy.deepcopy(model._ensembleDict)
  cfg = copy.deepcopy(model.config)

  array_header_text = """library ieee;
  use ieee.std_logic_1164.all;
  use ieee.std_logic_misc.all;
  use ieee.numeric_std.all;

  use work.Constants.all;
  use work.Types.all;
  """

  array_cast_text = """    constant value : tyArray2DnNodes(0 to nTrees - 1) := to_tyArray2D(value_int);
      constant threshold : txArray2DnNodes(0 to nTrees - 1) := to_txArray2D(threshold_int);"""

  bdt_instantiation_template = """  BDT{} : entity work.BDT
  generic map(
    iFeature => {},
    iChildLeft => {},
    iChildRight => {},
    iParent => {},
    iLeaf => {},
    depth => {},
    threshold => {},
    value => {},
    initPredict => {}
  )
  port map(
    clk => clk,
    X => X,
    X_vld => X_vld,
    y => {},
    y_vld => {}
  );

  """

  filedir = os.path.dirname(os.path.abspath(__file__))
  logger.info(f"Writing project to {cfg['OutputDir']}")
  os.makedirs('{}/firmware'.format(cfg['OutputDir']))
  copyfiles = ['AddReduce.vhd', 'BDT.vhd', 'BDTTestbench.vhd', 'SimulationInput.vhd', 'SimulationOutput.vhd',
               'TestUtil.vhd', 'Tree.vhd', 'Types.vhd']
  for f in copyfiles:
      copyfile('{}/firmware/{}'.format(filedir, f), '{}/firmware/{}'.format(cfg['OutputDir'], f))

  dtype = cfg['Precision']
  if not 'ap_fixed' in dtype:
    logger.error("Only ap_fixed is currently supported, exiting")
    sys.exit()
  dtype = dtype.replace('ap_fixed<', '').replace('>', '')
  dtype_n = int(dtype.split(',')[0].strip()) # total number of bits
  dtype_int = int(dtype.split(',')[1].strip()) # number of integer bits
  dtype_frac = dtype_n - dtype_int # number of fractional bits
  mult = 2**dtype_frac

  # binary classification only uses one set of trees
  n_classes = 1 if ensembleDict['n_classes'] == 2 else ensembleDict['n_classes']

  # Make a file for all the trees for each class
  fout = [open('{}/firmware/Arrays{}.vhd'.format(cfg['OutputDir'], i), 'w') for i in range(n_classes)]
  # Write the includes and package header for each file
  for i in range(n_classes):
    fout[i].write(array_header_text)
    fout[i].write('package Arrays{} is\n\n'.format(i))
    fout[i].write('    constant initPredict : ty := to_ty({});\n'.format(int(np.round(ensembleDict['init_predict'][i] * mult))))
    #fout[i].write('    constant initPredict : ty := to_ty({});\n'.format(int(np.round(ensembleDict['init_predict'][i] * mult))))


  # Loop over fields (childrenLeft, childrenRight, threshold...)
  for field in ensembleDict['trees'][0][0].keys():
    # Write the type for this field to each classes' file
    for iclass in range(n_classes):
      #dtype = 'txArray2D' if field == 'threshold' else 'tyArray2D' if field == 'value' else 'intArray2D'
      fieldName = field
      # The threshold and value arrays are declared as integers, then cast
      # So need a separate constant
      if field == 'threshold' or field == 'value':
        fieldName += '_int'
        # Convert the floating point values to integers
        for ii, trees in enumerate(ensembleDict['trees']):
          ensembleDict['trees'][ii][iclass][field] = np.round(np.array(ensembleDict['trees'][ii][iclass][field]) * mult).astype('int')
      nElem = 'nLeaves' if field == 'iLeaf' else 'nNodes'
      fout[iclass].write('    constant {} : intArray2D{}(0 to nTrees - 1) := ('.format(fieldName, nElem))
    # Loop over the trees within the class
    for itree, trees in enumerate(ensembleDict['trees']):
      for iclass, tree in enumerate(trees):
        fout[iclass].write('({})'.format(', '.join(map(str, tree[field]))))
        if itree < ensembleDict['n_trees'] - 1:
          fout[iclass].write(',')
        fout[iclass].write('\n{}'.format(' ' * 16))    
    for iclass in range(n_classes):
      fout[iclass].write(');\n')
  for i in range(n_classes):
    fout[i].write(array_cast_text + "\n")
    fout[i].write('end Arrays{};'.format(i))
    fout[i].close()

  write_sim_scripts(cfg, filedir, n_classes)

  f = open('{}/SimulationInput.txt'.format(cfg['OutputDir']), 'w')
  f.write(' '.join(map(str, [0] * ensembleDict['n_features'])))
  f.close()

  f = open(os.path.join(filedir,'./scripts/synth.tcl'),'r')
  fout = open('{}/synth.tcl'.format(cfg['OutputDir']), 'w')
  for line in f.readlines():
    if 'hls4ml' in line:
      newline = "synth_design -top BDTTop -part {}\n".format(cfg['XilinxPart'])
      fout.write(newline)
    else:
      fout.write(line)
  fout.write('write_edif -file {}'.format(cfg['ProjectName']))
  f.close()
  fout.close()

  f = open(os.path.join(filedir,'./firmware/BDTTop.vhd'),'r')
  fout = open('{}/firmware/BDTTop.vhd'.format(cfg['OutputDir']),'w')
  for line in f.readlines():
    if 'include arrays' in line:
        for i in range(n_classes):
          newline = 'use work.Arrays{};\n'.format(i)
          fout.write(newline)
    elif 'instantiate BDTs' in line:
      for i in range(n_classes):
        arr = 'Arrays{}.'.format(i)

        newline = bdt_instantiation_template.format('{}'.format(i),
                                                   '{}{}'.format(arr, 'feature'),
                                                   '{}{}'.format(arr, 'children_left'),
                                                   '{}{}'.format(arr, 'children_right'),
                                                   '{}{}'.format(arr, 'parent'),
                                                   '{}{}'.format(arr, 'iLeaf'),
                                                   '{}{}'.format(arr, 'depth'),
                                                   '{}{}'.format(arr, 'threshold') ,
                                                   '{}{}'.format(arr, 'value'),
                                                   '{}{}'.format(arr, 'initPredict'),
                                                   'y({})'.format(i),
                                                   'y_vld({})'.format(i))
        fout.write(newline)
    else:
      fout.write(line)
  f.close()
  fout.close()

  f = open(os.path.join(filedir, './firmware/Constants.vhd'), 'r')
  fout = open('{}/firmware/Constants.vhd'.format(cfg['OutputDir']), 'w')
  for line in f.readlines():
    if 'hls4ml' in line:
      newline = "  constant nTrees : integer := {};\n".format(ensembleDict['n_trees'])
      newline += "  constant maxDepth : integer := {};\n".format(ensembleDict['max_depth'])
      newline +=  "  constant nNodes : integer := {};\n".format(2 ** (ensembleDict['max_depth'] + 1) - 1)
      newline += "  constant nLeaves : integer := {};\n".format(2 ** ensembleDict['max_depth'])
      newline += "  constant nFeatures : integer := {};\n".format(ensembleDict['n_features'])
      newline += "  constant nClasses : integer := {};\n\n".format(n_classes)
      newline += "  subtype tx is signed({} downto 0);\n".format(dtype_n - 1)
      newline += "  subtype ty is signed({} downto 0);\n".format(dtype_n - 1)
      fout.write(newline)
    else:
      fout.write(line)
  f.close()
  fout.close()

def auto_config():
    config = {'ProjectName' : 'my-prj',
              'OutputDir'   : 'my-conifer-prj',
              'Precision'   : 'ap_fixed<18,8>',
              'XilinxPart' : 'xcvu9p-flgb2104-2L-e',
              'ClockPeriod' : '5'}
    return config

def sim_compile(model):
  from conifer.backends.vhdl import simulator
  config = copy.deepcopy(model.config)
  xsim_cmd = 'sh xsim_compile.sh > xsim_compile.log'
  msim_cmd = 'sh modelsim_compile.sh > modelsim_compile.log'
  ghdl_cmd = 'sh ghdl_compile.sh > ghdl_compile.log'
  cmdmap = {Simulators.modelsim : msim_cmd,
            Simulators.xsim : xsim_cmd,
            Simulators.ghdl : ghdl_cmd}
  cmd = cmdmap[simulator]
  logger.info(f'Compiling simulation for {simulator} simulator')
  logger.debug(f'Compiling simulation with command "{cmd}"')
  cwd = os.getcwd()
  os.chdir(config['OutputDir'])
  success = os.system(cmd)
  os.chdir(cwd)
  if(success > 0):
      logger.error("'sim_compile' failed, check {}_compile.log".format(simulator.name))
      sys.exit()
  return

def decision_function(X, model, trees=False):
    from conifer.backends.vhdl import simulator

    config = copy.deepcopy(model.config)
    msim_cmd = 'vsim -c -do "vsim -L BDT -L xil_defaultlib xil_defaultlib.testbench; run -all; quit -f" > vsim.log'
    xsim_cmd = 'xsim -R bdt_tb > xsim.log'
    ghdl_cmd = 'ghdl -r --std=08 --work=xil_defaultlib testbench > ghdl.log'
    cmdmap = {Simulators.modelsim : msim_cmd,
              Simulators.xsim : xsim_cmd,
              Simulators.ghdl : ghdl_cmd}
    cmd = cmdmap[simulator]
    msim_log = 'vsim.log'
    xsim_log = 'xsim.log'
    ghdl_log = 'ghdl.log'
    logmap = {Simulators.modelsim : msim_log,
              Simulators.xsim : xsim_log,
              Simulators.ghdl : ghdl_log}
    logfile = logmap[simulator]

    logger.info(f'Running simulation for {simulator} simulator')

    dtype = config['Precision']
    if not 'ap_fixed' in dtype:
        logger.error("Only ap_fixed is currently supported, exiting")
        sys.exit()
    dtype = dtype.replace('ap_fixed<', '').replace('>', '')
    dtype_n = int(dtype.split(',')[0].strip()) # total number of bits
    dtype_int = int(dtype.split(',')[1].strip()) # number of integer bits
    dtype_frac = dtype_n - dtype_int # number of fractional bits
    mult = 2**dtype_frac
    Xint = (X *  mult).astype('int')
    logger.debug(f'Converting X ({X.dtype}), to integers with scale factor {mult} from {config["Precision"]}')
    np.savetxt('{}/SimulationInput.txt'.format(config['OutputDir']),
               Xint, delimiter=' ', fmt='%d')
    cwd = os.getcwd()
    os.chdir(config['OutputDir'])
    logger.debug(f'Running simulation with command "{cmd}"')
    success = os.system(cmd)
    os.chdir(cwd)
    if(success > 0):
        logger.error("'decision_function' failed, see {}.log".format(logfile))
        sys.exit()
    y = np.loadtxt('{}/SimulationOutput.txt'.format(config['OutputDir'])) * 1. / mult
    if trees:
        logger.warn("Individual tree output (trees=True) not yet implemented for this backend")
    return y

def build(config, **kwargs):
    cmd = 'vivado -mode batch -source synth.tcl > build.log'
    cwd = os.getcwd()
    os.chdir(config['OutputDir'])
    start = datetime.datetime.now()
    logger.info(f'build starting {start:%H:%M:%S}')
    logger.debug(f'Running build with command "{cmd}"')
    success = os.system(cmd)
    os.chdir(cwd)
    stop = datetime.datetime.now()
    logger.info(f'build finished {stop:%H:%M:%S} - took {str(stop-start)}')
    if(success > 0):
        logger.error("build failed, check build.log")
        sys.exit()
            
def write_sim_scripts(cfg, filedir, n_classes):
  from conifer.backends.vhdl import simulator
  fmap = {Simulators.modelsim : write_modelsim_scripts,
          Simulators.xsim : write_xsim_scripts,
          Simulators.ghdl : write_ghdl_scripts,}
  fmap[simulator](cfg, filedir, n_classes)

def write_modelsim_scripts(cfg, filedir, n_classes):
  f = open(os.path.join(filedir,'./scripts/modelsim_compile.sh'),'r')
  fout = open('{}/modelsim_compile.sh'.format(cfg['OutputDir']),'w')
  for line in f.readlines():
    if 'insert arrays' in line:
      for i in range(n_classes):
        newline = 'vcom -2008 -work BDT ./firmware/Arrays{}.vhd\n'.format(i)
        fout.write(newline)
    else:
      fout.write(line)
  f.close()
  fout.close()

  f = open('{}/test.tcl'.format(cfg['OutputDir']), 'w')
  f.write('vsim -L BDT -L xil_defaultlib xil_defaultlib.testbench\n')
  f.write('run 100 ns\n')
  f.write('quit -f\n')
  f.close()

def write_xsim_scripts(cfg, filedir, n_classes):
  f = open(os.path.join(filedir, './scripts/xsim_compile.sh'), 'r')
  fout = open('{}/xsim_compile.sh'.format(cfg['OutputDir']), 'w')
  for line in f.readlines():
    if 'insert arrays' in line:
      for i in range(n_classes):
        newline = 'xvhdl -2008 -work BDT ./firmware/Arrays{}.vhd\n'.format(i)
        fout.write(newline)
    else:
      fout.write(line)
  f.close()
  fout.close()

def write_ghdl_scripts(cfg, filedir, n_classes):
  f = open(os.path.join(filedir, './scripts/ghdl_compile.sh'), 'r')
  fout = open('{}/ghdl_compile.sh'.format(cfg['OutputDir']), 'w')
  for line in f.readlines():
    if 'insert arrays' in line:
      for i in range(n_classes):
        newline = 'ghdl -a --std=08 --work=BDT ./firmware/Arrays{}.vhd\n'.format(i)
        fout.write(newline)
    else:
      fout.write(line)
  f.close()
  fout.close()