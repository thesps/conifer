import copy
import json
import shutil
import os
import zipfile
import time
from conifer.model import ConfigBase
import logging
logger = logging.getLogger(__name__)

class BoardConfig(ConfigBase):
  backend = 'accelerator_builder'
  _config_fields = ConfigBase._config_fields + ['xilinx_part', 'name']
  _new_alts = {'xilinx_part' : ['XilinxPart'], 'name' : ['Name']}
  _alternates = {**ConfigBase._alternates, **_new_alts}
  _new_defaults = {'xilinx_part' : 'xc7z020clg400-1', 'name' : 'pynq-z2'}
  _defaults = {**ConfigBase._defaults, **_new_defaults}
  _allow_undefined = ConfigBase._allow_undefined + ['output_dir', 'project_name', 'backend']
  def __init__(self, configDict, validate=True):
    super(BoardConfig, self).__init__(configDict, validate=False)
    if validate:
      self._validate()

  def default_config():
    return copy.deepcopy(BoardConfig._defaults)
  
  def to_json(self, filename=None):
    js = json.dumps(self._to_dict(), indent=1)
    if filename is not None:
      with open(filename, 'w') as f:
        f.write(js)
    return js

  def load(filename):
    with open(filename, 'r') as json_file:
      js = json.load(json_file)
      return BoardConfig(js)
  
class ZynqConfig(BoardConfig):
  backend = '{pn}_builder'
  _config_fields = BoardConfig._config_fields + ['board_part', 'processing_system_ip', 'processing_system']
  _new_alts = {'board_part' : ['BoardPart'],
                            'processing_system_ip' : ['ProcessingSystemIP'],
                            'processing_system' : ['ProcessingSystem']}
  _alternates = {**BoardConfig._alternates, **_new_alts}
  _new_defaults = {'board_part' : 'tul.com.tw:pynq-z2:part0:1.0',
                                'processing_system_ip' : 'xilinx.com:ip:processing_system7:5.5',
                                'processing_system' : 'processing_system7'}
  _defaults = {**BoardConfig._defaults, **_new_defaults}
  def __init__(self, configDict, validate=True):
    super(ZynqConfig, self).__init__(configDict, validate=False)
    if validate:
      self._validate()

  def default_config():
    return copy.deepcopy(ZynqConfig._defaults)
  
  def load(filename):
    with open(filename, 'r') as json_file:
      js = json.load(json_file)
      return ZynqConfig(js)

class ZynqBuilder:
  def __init__(self, project_cfg, board_cfg, top_name=None, ip_name=None):
    assert isinstance(project_cfg, ConfigBase)
    assert isinstance(board_cfg, ZynqConfig)
    self.project_cfg = project_cfg
    self.board_cfg = board_cfg
    self.top_name = top_name
    self.ip_name = ip_name

    xp0 = self.project_cfg.xilinx_part
    xp1 = self.board_cfg.xilinx_part
    if  xp0 != xp1 :
      logger.warn(f'Project and Board config xilinx_parts do not match ({xp0}, {xp1}), setting to {xp1}')
    self.project_cfg.xilinx_part = xp1

  def default_cfg():
    return ZynqConfig.default_config()

  def get_tcl_params(self):
    params =  f'set flow_target vivado\n'
    params += f'set export_format ip_catalog\n'

  def write(self):
    '''
    Write project files
    '''
    filedir = os.path.dirname(os.path.abspath(__file__))
    # Vivado fails to connect the m_axi port if the ip_name is a variable for an unknown reason
    #shutil.copyfile(f'{filedir}/src/build_bit.tcl',
    #                f'{self.project_cfg.output_dir}/build_bit.tcl')
    f = open(f'{filedir}/src/build_bit.tcl')
    fout = open(f'{self.project_cfg.output_dir}/build_bit.tcl', 'w')
    for line in f.readlines():
      line = line.replace('${ip_name}', self.ip_name)
      fout.write(line)
    f.close()
    fout.close()
    self.write_tcl()

  def write_tcl(self):
    import conifer
    with open(f'{self.project_cfg.output_dir}/accelerator_parameters.tcl', 'w') as f:
      f.write(f'set prj_name {self.project_cfg.project_name}\n')
      f.write(f'set part {self.project_cfg.xilinx_part}\n')
      f.write(f'set board_part {self.board_cfg.board_part}\n')
      f.write(f'set processing_system_ip {self.board_cfg.processing_system_ip}\n')
      f.write(f'set processing_system {self.board_cfg.processing_system}\n')
      f.write(f'set top {self.top_name}\n')
      f.write(f'set ip_name {self.ip_name}\n')
      f.write(f'set version {conifer.__version__.major}.{conifer.__version__.minor}\n')

  def get_flow_target(self):
    return 'vivado'
  
  def get_export_format(self):
    return 'ip_catalog'
  
  def get_maxi64(self):
    return False

  def build(self):
    '''
    Build Zynq project
    '''
    self.write()
    cwd = os.getcwd()
    os.chdir(self.project_cfg.output_dir)
    success = True
    cmd = 'vivado -mode batch -source build_bit.tcl > vivado_build.log'
    logger.info(f'Building Zynq bitfile with command "{cmd}"')
    success = success and os.system(cmd)==0
    os.chdir(cwd)
    return success
  
  def package(self, retry: int = 6, retries: int = 10):
    '''
    Collect build products and compress to a zip file
    Parameters
    ----------
    retry: int (optional)
      wait time in seconds before retrying
    retries: int (optional)
      number of retries before exiting
    '''
    od = self.project_cfg.output_dir
    pn = self.project_cfg.project_name
    logger.info(f'Packaging bitfile to {od}/{pn}.zip')

    for attempt in range(retries):
      if os.path.exists(f'{od}/{pn}_vivado/{pn}.xsa'):
        with zipfile.ZipFile(f'{od}/{pn}_vivado/{pn}.xsa', 'r') as zip:
          zip.extractall(f'{od}/{pn}_vivado/package')
        os.makedirs(f'{od}/package', exist_ok=True)
        shutil.copyfile(f'{od}/{pn}_vivado/package/{pn}.bit', f'{od}/package/{pn}.bit')
        shutil.copyfile(f'{od}/{pn}_vivado/package/design_1.hwh', f'{od}/package/{pn}.hwh')
        filedir = os.path.dirname(os.path.abspath(__file__))
        with zipfile.ZipFile(f'{od}/{pn}.zip', 'w') as zip:
          zip.write(f'{od}/package/{pn}.bit', f'{pn}.bit')
          zip.write(f'{od}/package/{pn}.hwh', f'{pn}.hwh')
          #zip.write(f'{filedir}/src/LICENSE', 'LICENSE')
        break
      else:
        logger.info(f'Bitfile not found, waiting {retry} seconds for retry')
        time.sleep(retry)   

class AlveoConfig(BoardConfig):
  backend = '{pn}_builder'
  _config_fields = BoardConfig._config_fields + ['platform']
  _new_alts = {'platform' : ['Platform']}
  _alternates = {**BoardConfig._alternates, **_new_alts}
  _new_defaults = {'platform' : 'xilinx_u200_gen3x16_xdma_2_202110_1'}
  _defaults = {**BoardConfig._defaults, **_new_defaults}
  _defaults['clock_period'] = 3
  def __init__(self, configDict, validate=True):
    super(AlveoConfig, self).__init__(configDict, validate=False)
    if validate:
      self._validate()

  def default_config():
    return copy.deepcopy(AlveoConfig._defaults)
  
  def load(filename):
    with open(filename, 'r') as json_file:
      js = json.load(json_file)
      return AlveoConfig(js)
    
class AlveoBuilder:
  def __init__(self, cfg): 
    self.cfg = AlveoConfig(cfg)

  def default_cfg():
    return AlveoConfig.default_config()

  def get_flow_target(self):
    return 'vitis'
  
  def get_export_format(self):
    return 'xo'
  
  def get_maxi64(self):
    return True
   
  def write(self):
    return

  def build(self, target='hw'):
    '''
    Build Alveo project
    Parameters
    ----------
    target: string (optional)
      v++ target
    '''
    self.write()
    cwd = os.getcwd()
    os.chdir(od)
    success = True
    pn = pn
    vitis_cmd = f'v++ -t {target} --platform {self.platform} --link {pn}/solution1/impl/export.xo -o {pn}.xclbin'
    logger.info(f'Building Alveo bitfile with command "{vitis_cmd}"')
    success = success and os.system(vitis_cmd)==0
    os.chdir(cwd)
    return success
  