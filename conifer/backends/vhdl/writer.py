import os
from shutil import copyfile
import numpy as np
import math
from enum import Enum
from conifer.utils import FixedPointConverter, copydocstring
from conifer.backends.common import BottomUpDecisionTree, MultiPrecisionConfig, read_vsynth_report
from conifer.model import ModelBase
import copy
import datetime
import logging
logger = logging.getLogger(__name__)

class VHDLConfig(MultiPrecisionConfig):
  backend = 'vhdl'
  _config_fields = MultiPrecisionConfig._config_fields + ['xilinx_part']
  _vhdl_alts = {'xilinx_part'  : ['XilinxPart']}
  _alternates = {**MultiPrecisionConfig._alternates, **_vhdl_alts}
  _vhdl_defaults = {'precision'    : 'ap_fixed<18,8>',
                    'xilinx_part'  : 'xcvu9p-flgb2104-2L-e',
                    }
  _defaults = {**MultiPrecisionConfig._defaults, **_vhdl_defaults}
  def __init__(self, configDict, validate=True):
    super(VHDLConfig, self).__init__(configDict, validate=False)
    if validate:
      self._validate()

  def default_config():
    return copy.deepcopy(VHDLConfig._defaults)

  def _extra_validate(self):
    # TODO: proagate different precisions properly through backend
    # for now enforce that all the precisions are equal
    assert self.input_precision == self.threshold_precision, f'input & threshold precision must be equal, got: {self.input_precision} & {self.threshold_precision}'
    assert self.threshold_precision == self.score_precision, f'threshold & score precision must be equal, got: {self.threshold_precision} & {self.score_precision}'

class VHDLModel(ModelBase):

  def __init__(self, ensembleDict, config, metadata=None):
    super(VHDLModel, self).__init__(ensembleDict, config, metadata)
    self.config = VHDLConfig(config)
    self._fp_converter = FixedPointConverter(self.config.input_precision)
    trees = ensembleDict.get('trees', None)
    assert trees is not None, f'Missing expected key trees in ensembleDict'
    self.trees = [[BottomUpDecisionTree(treeDict) for treeDict in trees_class] for trees_class in trees]
    for trees_class in self.trees:
      for tree in trees_class:
        tree.padTree(self.max_depth)
        # Convert the floating point values to integers
        tree.threshold_int = np.array([self._fp_converter.to_int(x) for x in tree.threshold])
        tree.value_int = np.array([self._fp_converter.to_int(x) for x in tree.value])

  @copydocstring(ModelBase.write)
  def write(self):

    self.save()
    cfg = self.config

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
    logger.info(f"Writing project to {cfg.output_dir}")
    os.makedirs('{}/firmware'.format(cfg.output_dir), exist_ok=True)
    copyfiles = ['AddReduce.vhd', 'BDT.vhd', 'BDTTestbench.vhd', 'SimulationInput.vhd', 'SimulationOutput.vhd',
                'TestUtil.vhd', 'Tree.vhd', 'Types.vhd']
    for f in copyfiles:
        copyfile('{}/firmware/{}'.format(filedir, f), '{}/firmware/{}'.format(cfg.output_dir, f))

    # binary classification only uses one set of trees
    n_classes = 1 if self.n_classes == 2 else self.n_classes

    # Make a file for all the trees for each class
    fout = [open('{}/firmware/Arrays{}.vhd'.format(cfg.output_dir, i), 'w') for i in range(n_classes)]
    # Write the includes and package header for each file
    for i in range(n_classes):
      fout[i].write(array_header_text)
      fout[i].write('package Arrays{} is\n\n'.format(i))
      fout[i].write('    constant initPredict : ty := to_ty({});\n'.format(self._fp_converter.to_int(np.float64(self.init_predict[i]))))

    # Loop over fields (childrenLeft, childrenRight, threshold...)
    tree_fields = ['feature', 'threshold_int', 'value_int',
                   'children_left', 'children_right', 'parent',
                   'iLeaf', 'depth']
    for field in tree_fields:
      # Write the type for this field to each classes' file
      for iclass in range(n_classes):
        nElem = 'nLeaves' if field == 'iLeaf' else 'nNodes'
        fout[iclass].write('    constant {} : intArray2D{}(0 to nTrees - 1) := ('.format(field, nElem))
      # Loop over the trees within the class
      for itree, trees in enumerate(self.trees):
        for iclass, tree in enumerate(trees):
          fout[iclass].write('({})'.format(', '.join(map(str, getattr(tree, field)))))
          if itree < self.n_trees - 1:
            fout[iclass].write(',')
          fout[iclass].write('\n{}'.format(' ' * 16))    
      for iclass in range(n_classes):
        fout[iclass].write(');\n')
    for i in range(n_classes):
      fout[i].write(array_cast_text + "\n")
      fout[i].write('end Arrays{};'.format(i))
      fout[i].close()

    from conifer.backends.vhdl import simulator
    simulator.write_scripts(cfg.output_dir, filedir, n_classes)

    f = open('{}/SimulationInput.txt'.format(cfg.output_dir), 'w')
    f.write(' '.join(map(str, [0] * self.n_features)))
    f.close()

    f = open(os.path.join(filedir,'./scripts/synth.tcl'),'r')
    fout = open('{}/synth.tcl'.format(cfg.output_dir), 'w')
    for line in f.readlines():
      if 'hls4ml' in line:
        newline = "synth_design -top BDTTop -part {}\n".format(cfg.xilinx_part)
        fout.write(newline)
      else:
        fout.write(line)
    fout.write('write_edif -file {}'.format(cfg.project_name))
    f.close()
    fout.close()

    f = open(os.path.join(filedir,'./firmware/BDTTop.vhd'),'r')
    fout = open('{}/firmware/BDTTop.vhd'.format(cfg.output_dir),'w')
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
    fout = open('{}/firmware/Constants.vhd'.format(cfg.output_dir), 'w')
    for line in f.readlines():
      if 'hls4ml' in line:
        newline = "  constant nTrees : integer := {};\n".format(self.n_trees)
        newline += "  constant maxDepth : integer := {};\n".format(self.max_depth)
        newline +=  "  constant nNodes : integer := {};\n".format(2 ** (self.max_depth + 1) - 1)
        newline += "  constant nLeaves : integer := {};\n".format(2 ** self.max_depth)
        newline += "  constant nFeatures : integer := {};\n".format(self.n_features)
        newline += "  constant nClasses : integer := {};\n\n".format(n_classes)
        newline += "  subtype tx is signed({} downto 0);\n".format(self._fp_converter.width - 1)
        newline += "  subtype ty is signed({} downto 0);\n".format(self._fp_converter.width - 1)
        fout.write(newline)
      else:
        fout.write(line)
    f.close()
    fout.close()

  @copydocstring(ModelBase.compile)
  def compile(self):
    self.write()
    from conifer.backends.vhdl import simulator
    return simulator.compile(self.config.output_dir)

  @copydocstring(ModelBase.decision_function)
  def decision_function(self, X, trees=False):
      from conifer.backends.vhdl import simulator

      config = copy.deepcopy(self.config)

      Xint = np.array([self._fp_converter.to_int(x) for x in X.ravel()]).reshape(X.shape)
      np.savetxt('{}/SimulationInput.txt'.format(config.output_dir),
                Xint, delimiter=' ', fmt='%d')
      success = simulator.run_sim(config.output_dir)
      if not success:
        return 
      y = np.loadtxt('{}/SimulationOutput.txt'.format(config.output_dir)).astype(np.int32)
      y = np.array([self._fp_converter.from_int(yi) for yi in y.ravel()]).reshape(y.shape)
      if np.ndim(y) == 1:
        y = np.expand_dims(y, 1)

      if trees:
          logger.warn("Individual tree output (trees=True) not yet implemented for this backend")
      return y

  @copydocstring(ModelBase.build)
  def build(self, **kwargs):
      cmd = 'vivado -mode batch -source synth.tcl > build.log'
      cwd = os.getcwd()
      os.chdir(self.config.output_dir)
      start = datetime.datetime.now()
      logger.info(f'build starting {start:%H:%M:%S}')
      logger.debug(f'Running build with command "{cmd}"')
      success = os.system(cmd)
      os.chdir(cwd)
      stop = datetime.datetime.now()
      logger.info(f'build finished {stop:%H:%M:%S} - took {str(stop-start)}')
      if(success > 0):
          logger.error("build failed, check build.log")
          return False
      return True

  def read_report(self) -> dict:
    '''
    Read the Synthesis report
    Returns
    ----------
    dictionary of extracted report contents
    '''
    vsynth_report = read_vsynth_report(f'{self.config.output_dir}/util.rpt')
    report = {}
    for key in ['lut', 'ff']:
      report[key] = vsynth_report[key]

    # VHDL latency and II are fixed by implementation
    # use math ceil to return int
    report['latency'] = 1 + self.max_depth + math.ceil(math.log2(self.n_trees))
    report['interval'] = 1
    return report

def make_model(ensembleDict, config):
    return VHDLModel(ensembleDict, config)
    
def auto_config():
    config = {'Backend' : 'vhdl',
              'ProjectName' : 'my_prj',
              'OutputDir'   : 'my-conifer-prj',
              'Precision'   : 'ap_fixed<18,8>',
              'XilinxPart' : 'xcvu9p-flgb2104-2L-e',
             }
    return config