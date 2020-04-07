import os
import sys
from shutil import copyfile
import numpy as np

def write(ensembleDict, cfg):
  array_header_text = """library ieee;
  use ieee.std_logic_1164.all;
  use ieee.std_logic_misc.all;
  use ieee.numeric_std.all;

  use work.Constants.all;
  use work.Types.all;
  """

  array_cast_text = """    constant value : tyArray2D(nTrees-1 downto 0)(nNodes-1 downto 0) := to_tyArray2D(value_int);
      constant threshold : txArray2D(nTrees-1 downto 0)(nNodes-1 downto 0) := to_txArray2D(threshold_int);"""

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
  os.makedirs('{}/firmware'.format(cfg['OutputDir']))
  copyfiles = ['AddReduce.vhd', 'BDT.vhd', 'BDTTestbench.vhd', 'SimulationInput.vhd', 'SimulationOutput.vhd',
               'TestUtil.vhd', 'Tree.vhd', 'Types.vhd']
  for f in copyfiles:
      copyfile('{}/firmware/{}'.format(filedir, f), '{}/firmware/{}'.format(cfg['OutputDir'], f))

  dtype = cfg['Precision']
  if not 'ap_fixed' in dtype:
    print("Only ap_fixed is currently supported, exiting")
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
    fout[i].write('    constant initPredict : ty := to_ty({});\n'.format(int(np.round(ensembleDict['init_predict'] * mult))))
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
      fout[iclass].write('    constant {} : intArray2D(nTrees-1 downto 0)({}-1 downto 0) := ('.format(fieldName, nElem))
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

def sim_compile(config):
    cwd = os.getcwd()
    os.chdir(config['OutputDir'])
    cmd = 'sh modelsim_compile.sh > modelsim_compile.log'
    success = os.system(cmd)
    os.chdir(cwd)
    if(success > 0):
        print("'sim_compile' failed, check modelsim_compile.log")
        sys.exit()
    return

def decision_function(X, config):
    dtype = config['Precision']
    if not 'ap_fixed' in dtype:
        print("Only ap_fixed is currently supported, exiting")
        sys.exit()
    dtype = dtype.replace('ap_fixed<', '').replace('>', '')
    dtype_n = int(dtype.split(',')[0].strip()) # total number of bits
    dtype_int = int(dtype.split(',')[1].strip()) # number of integer bits
    dtype_frac = dtype_n - dtype_int # number of fractional bits
    mult = 2**dtype_frac
    Xint = (X *  mult).astype('int')
    np.savetxt('{}/SimulationInput.txt'.format(config['OutputDir']),
               Xint, delimiter=' ', fmt='%d')
    latency = 20
    t = 5 * (len(X) + latency)
    cmd = 'vsim -c -do "vsim -L BDT -L xil_defaultlib xil_defaultlib.testbench; run {} ns; quit -f" > vsim.log'.format(t)
    cwd = os.getcwd()
    os.chdir(config['OutputDir'])
    success = os.system(cmd)
    os.chdir(cwd)
    if(success > 0):
        print("'decision_function' failed")
        sys.exit()
    y = np.loadtxt('{}/SimulationOutput.txt'.format(config['OutputDir'])) * 1. / mult
    return y
