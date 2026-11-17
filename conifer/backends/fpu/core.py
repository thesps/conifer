'''
The FPU core: the VHDL decision function FPU_df, written for every FPU whatever its interface (see targets.py)
'''
import math
import os
import shutil
import numpy as np
from conifer.utils import address_bits, FixedPointType

def _resolve_type(t):
  if isinstance(t, int):
    return f'ap_fixed<{t},{t}>'
  elif isinstance(t, str):
    return t

def _fill_template(src, dst, inserts):
  '''Copy src to dst, replacing each line containing the marker 'conifer insert {key}' with inserts[key]'''
  with open(src) as fin, open(dst, 'w') as fout:
    for line in fin:
      for key, text in inserts.items():
        if f'conifer insert {key}' in line:
          line = text + '\n'
          break
      fout.write(line)

class _VHDLWriter:
  '''
  Write the VHDL decision function FPU_df from the templates in src/vhdl: the package fpu_pkg.vhd with the FPU
  configuration constants written, the other sources, the testbench, and the script to simulate it.
  '''
  vhdl_files = ['fpu_pkg.vhd', 'TreeEngine.vhd', 'FPU_df.vhd']
  vhdl_tb_files = ['FPU_df_tb.vhd']
  # xsim of FPU_df_tb on the project testbench data
  vhdl_scripts = ['xsim_tb.sh']

  def __init__(self, cfg):
    '''cfg: FPUConfig'''
    self.cfg = cfg
    self.nte, self.nf = cfg.tree_engines, cfg.features
    self.tbits, self.ubits = (FixedPointType(_resolve_type(t)).width for t in (cfg.threshold_type, cfg.score_type))
    # as ADDRBITS and FEATBITS in parameters.h
    self.addrbits = math.ceil(np.log2(cfg.nodes)) + 1
    self.featbits = math.ceil(np.log2(cfg.features)) + 1
    self.classbits = 1
    self.nawidth, self.rawidth = address_bits(cfg.nodes), address_bits(cfg.roots + 1)
    self.nodebits = self.tbits + self.ubits + self.featbits + 2 * self.addrbits + self.classbits + 1

  def constants(self):
    return {'NTE' : self.nte, 'NFEATURES' : self.nf, 'NNODES' : self.cfg.nodes, 'NROOTS' : self.cfg.roots, 'TBITS' : self.tbits, 'UBITS' : self.ubits,
            'FEATBITS' : self.featbits, 'ADDRBITS' : self.addrbits, 'CLASSBITS' : self.classbits,
            'NAWIDTH' : self.nawidth, 'RAWIDTH' : self.rawidth}

  @staticmethod
  def copy_sources(odir, files):
    '''Copy the named files from src/vhdl to odir'''
    filedir = os.path.dirname(os.path.abspath(__file__))
    for v in files:
      shutil.copyfile(f'{filedir}/src/vhdl/{v}', f'{odir}/{v}')

  def write_vhdl(self, odir, testbench=True):
    '''Write the VHDL to odir: the package with the configuration, the other sources, and the script to simulate it'''
    filedir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(odir, exist_ok=True)
    constants = '\n'.join(f'  constant {k} : integer := {v};' for k, v in self.constants().items())
    _fill_template(f'{filedir}/src/vhdl/fpu_pkg.vhd', f'{odir}/fpu_pkg.vhd', {'constants' : constants})
    self.copy_sources(odir, self.vhdl_files[1:] + (self.vhdl_tb_files if testbench else []) + self.vhdl_scripts)

  def write(self, odir):
    '''Write the VHDL to {odir}/vhdl'''
    self.write_vhdl(f'{odir}/vhdl')
