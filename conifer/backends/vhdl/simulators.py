import os
import subprocess
import logging
logger = logging.getLogger(__name__)


def _compile_sim(simulator, odir):
  logger.info(f'Compiling simulation for {simulator.__name__.lower()} simulator')
  logger.debug(f'Compiling simulation with command "{simulator._compile_cmd}"')
  cwd = os.getcwd()
  os.chdir(odir)
  success = os.system(simulator._compile_cmd)
  os.chdir(cwd)
  if(success > 0):
      logger.error(f"'sim_compile' failed, check {simulator.__name__.lower()}_compile.log")
  return success == 0

def _run_sim(simulator, odir):
    logger.info(f'Running simulation for {simulator.__name__.lower()} simulator')
    logger.debug(f'Running simulation with command "{simulator._run_cmd}"')
    cwd = os.getcwd()
    os.chdir(odir)
    success = os.system(simulator._run_cmd)
    os.chdir(cwd)
    if(success > 0):
      logger.error(f"'sim_compile' failed, check {simulator.__name__.lower()}.log")
    return success == 0

def _touch(simulator):
  cmd = simulator._touch_cmd
  try:
    success = subprocess.call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
  except:
    success = 1
  return success == 0

class Modelsim:
  _touch_cmd = ['vsim', '-c', '-do', 'quit -f']
  _compile_cmd = 'sh modelsim_compile.sh > modelsim_compile.log'
  _run_cmd = 'vsim -c -do "vsim -L BDT -L xil_defaultlib xil_defaultlib.testbench; run -all; quit -f" > vsim.log'
  
  def write_scripts(outputdir, filedir, n_classes):
    f = open(os.path.join(filedir,'./scripts/modelsim_compile.sh'),'r')
    fout = open(f'{outputdir}/modelsim_compile.sh','w')
    for line in f.readlines():
      if 'insert arrays' in line:
        for i in range(n_classes):
          newline = f'vcom -2008 -work BDT ./firmware/Arrays{i}.vhd\n'
          fout.write(newline)
      else:
        fout.write(line)
    f.close()
    fout.close()

    f = open(f'{outputdir}/test.tcl', 'w')
    f.write('vsim -L BDT -L xil_defaultlib xil_defaultlib.testbench\n')
    f.write('run 100 ns\n')
    f.write('quit -f\n')
    f.close()

  def compile(odir):
    return _compile_sim(Modelsim, odir)

  def run_sim(odir):
    return _run_sim(Modelsim, odir)

class GHDL:
  _touch_cmd = ['ghdl', '--version']
  _compile_cmd = 'sh ghdl_compile.sh > ghdl_compile.log'
  _run_cmd = 'ghdl -r --std=08 --work=xil_defaultlib testbench > ghdl.log'
  def write_scripts(outputdir, filedir, n_classes):
    f = open(os.path.join(filedir, './scripts/ghdl_compile.sh'), 'r')
    fout = open(f'{outputdir}/ghdl_compile.sh', 'w')
    for line in f.readlines():
      if 'insert arrays' in line:
        for i in range(n_classes):
          newline = f'ghdl -a --std=08 --work=BDT ./firmware/Arrays{i}.vhd\n'
          fout.write(newline)
      else:
        fout.write(line)
    f.close()
    fout.close()

  def compile(odir):
    return _compile_sim(GHDL, odir)

  def run_sim(odir):
    return _run_sim(GHDL, odir)

class Xsim:
  _touch_cmd = ['xsim', '--version']
  _compile_cmd = 'sh xsim_compile.sh > xsim_compile.log'
  _run_cmd = 'xsim -R bdt_tb > xsim.log'
  def write_scripts(outputdir, filedir, n_classes):
    f = open(os.path.join(filedir, './scripts/xsim_compile.sh'), 'r')
    fout = open(f'{outputdir}/xsim_compile.sh', 'w')
    for line in f.readlines():
      if 'insert arrays' in line:
        for i in range(n_classes):
          newline = f'xvhdl -2008 -work BDT ./firmware/Arrays{i}.vhd\n'
          fout.write(newline)
      else:
        fout.write(line)
    f.close()
    fout.close()

  def compile(odir):
    return _compile_sim(Xsim, odir)

  def run_sim(odir):
    return _run_sim(Xsim, odir)
