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
  _config_fields = BoardConfig._config_fields + ['board_part', 'processing_system_ip', 'processing_system', 'ps_config', 'ps_s_axi_port', 'ps_m_axi_port']
  _new_alts = {'board_part' : ['BoardPart'],
                'processing_system_ip' : ['ProcessingSystemIP'],
                'processing_system' : ['ProcessingSystem'],
                'ps_s_axi_port' : [],
                'ps_m_axi_port' : [],
                'ps_config' : []}
  _alternates = {**BoardConfig._alternates, **_new_alts}
  _new_defaults = {'board_part' : 'tul.com.tw:pynq-z2:part0:1.0',
                   'processing_system_ip' : 'xilinx.com:ip:processing_system7:5.5',
                   'processing_system' : 'processing_system7',
                   'ps_config' : 'CONFIG.PCW_USE_S_AXI_HP0 {1}',
                   'ps_s_axi_port' : 'S_AXI_HP0',
                   'ps_m_axi_port' : 'M_AXI_GP0'}
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

class Builder:
  def __init__(self, project_cfg, board_cfg, top_name=None, ip_name=None):
    assert isinstance(project_cfg, ConfigBase)
    assert isinstance(board_cfg, BoardConfig)
    self.project_cfg = project_cfg
    self.board_cfg = board_cfg
    self.top_name = top_name
    self.ip_name = ip_name

    xp0 = getattr(self.project_cfg, 'xilinx_part', None)
    xp1 = self.board_cfg.xilinx_part
    if  xp0 is not None and xp0 != xp1 :
      logger.warn(f'Project and Board config xilinx_parts do not match ({xp0}, {xp1}), setting to {xp1}')
      self.project_cfg.xilinx_part = xp1

  def default_cfg():
    return BoardConfig.default_config()
  
  def write(self):
    raise NotImplementedError
  
  def build(self):
    raise NotImplementedError
  
class ZynqBuilder(Builder):
  
  def __init__(self, project_cfg, board_cfg, top_name=None, ip_name=None):
    super(ZynqBuilder, self).__init__(project_cfg, board_cfg, top_name, ip_name)
    assert isinstance(project_cfg, ConfigBase)
    assert isinstance(board_cfg, ZynqConfig)

  def default_cfg():
    return ZynqConfig.default_config()

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
      line = line.replace('${ps_config}', self.board_cfg.ps_config)
      line = line.replace('${ps_s_axi_port}', self.board_cfg.ps_s_axi_port)
      line = line.replace('${ps_m_axi_port}', self.board_cfg.ps_m_axi_port)
      fout.write(line)
    f.close()
    fout.close()
    self.write_tcl()

  def write_tcl(self):
    import conifer
    with open(f'{self.project_cfg.output_dir}/accelerator_parameters.tcl', 'w') as f:
      f.write(f'set prj_name {self.project_cfg.project_name}\n')
      f.write(f'set part {self.board_cfg.xilinx_part}\n')
      f.write(f'set board_part {self.board_cfg.board_part}\n')
      f.write(f'set processing_system_ip {self.board_cfg.processing_system_ip}\n')
      f.write(f'set processing_system {self.board_cfg.processing_system}\n')
      #f.write(f'set ps_config {self.board_cfg.ps_config}\n')
      f.write(f'set ps_s_axi_port {self.board_cfg.ps_s_axi_port}\n')
      f.write(f'set ps_m_axi_port {self.board_cfg.ps_m_axi_port}\n')
      f.write(f'set top {self.top_name}\n')
      f.write(f'set ip_name {self.ip_name}\n')
      f.write(f'set version {conifer.__version__.major}.{conifer.__version__.minor}\n')

  def get_flow_target(self):
    return 'vivado'
  
  def get_export_format(self):
    return 'ip_catalog'
  
  def get_maxi64(self):
    return False

  def build(self, vivado_opts=None):
    '''
    Build Zynq project

    Parameters
    ----------
    vivado_opts: string (optional)
      additional options to pass to vivado command
    '''
    self.write()
    cwd = os.getcwd()
    os.chdir(self.project_cfg.output_dir)
    success = True
    vivado_opts = '' if vivado_opts is None else vivado_opts
    cmd = f'vivado -mode batch -source build_bit.tcl {vivado_opts} > vivado_build.log'
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
  _new_defaults = {'platform'    : 'xilinx_u200_gen3x16_xdma_2_202110_1',
                   'xilinx_part' : 'xcu200-fsgd2104-2-e'}
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
    
class AlveoBuilder(Builder):
  
  def __init__(self, project_cfg, board_cfg, top_name=None, ip_name=None):
    super(AlveoBuilder, self).__init__(project_cfg, board_cfg, top_name, ip_name)
    assert isinstance(project_cfg, ConfigBase)
    assert isinstance(board_cfg, AlveoConfig)

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

  def build(self, target = 'hw', vitis_opts : str = None):
    '''
    Build Alveo project

    Parameters
    ----------
    target: string (optional)
      v++ target
    
    vitis_opts: string (optional)
      additional options to pass to v++ command
    '''
    self.write()
    cwd = os.getcwd()
    od = self.project_cfg.output_dir
    pn = self.project_cfg.project_name
    os.chdir(od)
    success = True
    pn = pn
    vitis_opts = '' if vitis_opts is None else vitis_opts
    vitis_cmd = f'v++ -t {target} --platform {self.board_cfg.platform} --link {pn}/solution1/impl/export.xo -o {pn}.xclbin {vitis_opts} > vitis_build.log'
    logger.info(f'Building Alveo bitfile with command "{vitis_cmd}"')
    success = success and os.system(vitis_cmd)==0
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
      if os.path.exists(f'{od}/{pn}.xclbin'):
        filedir = os.path.dirname(os.path.abspath(__file__))
        with zipfile.ZipFile(f'{od}/{pn}.zip', 'w') as zip:
          zip.write(f'{od}/{pn}.xclbin', f'{pn}.xclbin')
          zip.write(f'{od}/{pn}.json', f'{pn}.json')
          zip.write(f'{filedir}/src/LICENSE', 'LICENSE')
        break
      else:
        logger.info(f'Bitfile not found, waiting {retry} seconds for retry')
        time.sleep(retry)
  