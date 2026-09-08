import os
import re
import numpy as np
import datetime
from dataclasses import dataclass
from conifer.utils.misc import _ap_include, _gcc_opts
import logging
logger = logging.getLogger(__name__)

# ap_fixed<W,I,Q,O,N>: Q/O default per Xilinx's ap_fixed_base.h when omitted from the type string
_DEFAULT_ROUNDING_MODE = 'AP_TRN'
_DEFAULT_OVERFLOW_MODE = 'AP_WRAP'
_DEFAULT_SATURATION_BITS = 0

_FIXED_RE = re.compile(
  r'ap_(u?)fixed\s*<\s*(\d+)\s*,\s*(-?\d+)'   # width, integer_bits
  r'(?:\s*,\s*(AP_\w+))?'                     # quantization/rounding mode
  r'(?:\s*,\s*(AP_\w+))?'                     # overflow/saturation mode
  r'(?:\s*,\s*(\d+))?'                        # saturation bits, only used by AP_WRAP_SM
)
_INT_RE = re.compile(r'ap_(u?)int\s*<\s*(\d+)')

@dataclass(frozen=True)
class ApType:
  '''Parsed fields of an ap_[u]fixed<W,I,Q,O,N>/ap_[u]int<W> type string'''
  width: int
  integer_bits: int
  signed: bool
  rounding_mode: str = None      # ap_int has no rounding/overflow modes: exact, always wraps
  overflow_mode: str = None
  saturation_bits: int = None

  @property
  def fractional_bits(self):
    return self.width - self.integer_bits

def parse_ap_type(type_string):
  '''
  Parse an ap_[u]fixed<W,I,Q,O,N> or ap_[u]int<W> type string into an ApType
  Returns None if type_string isn't a string or doesn't match either pattern
  '''
  if not isinstance(type_string, str):
    return None
  m = _FIXED_RE.search(type_string)
  if m:
    unsigned, width, integer_bits, rounding_mode, overflow_mode, saturation_bits = m.groups()
    return ApType(int(width), int(integer_bits), not unsigned,
                 rounding_mode=rounding_mode or _DEFAULT_ROUNDING_MODE,
                 overflow_mode=overflow_mode or _DEFAULT_OVERFLOW_MODE,
                 saturation_bits=int(saturation_bits) if saturation_bits else _DEFAULT_SATURATION_BITS)
  m = _INT_RE.search(type_string)
  if m:
    unsigned, width = m.groups()
    return ApType(int(width), int(width), not unsigned)
  return None

class FixedPointConverter:
  '''
  A Python wrapper around ap_fixed types to easily emulate the correct number representations
  '''

  def __init__(self, type_string):
    '''
    Construct the FixedPointConverter. Compiles the c++ library to use for conversions
    args:
      type_string : string for the ap_ type, e.g. ap_fixed<16,6,AP_RND,AP_SAT>
    '''
    self._parse(type_string)
    logger.info(f'Constructing converter for {type_string}')
    self.type_string = type_string
    self.sani_type = type_string.replace('<','_').replace('>','').replace(',','_')
    self.sani_type += f'_{int(datetime.datetime.now().timestamp()) + np.random.randint(0, 2**32)}'
    filedir = os.path.dirname(os.path.abspath(__file__))
    cpp_filedir = f"./.fp_converter_{self.sani_type}"
    cpp_filename = cpp_filedir + f'/{self.sani_type}.cpp'
    os.makedirs(cpp_filedir, exist_ok=True)

    fin = open(f'{filedir}/fixed_point_conversions.cpp', 'r')
    fout = open(cpp_filename, 'w')
    for line in fin.readlines():
      newline = line
      if '// conifer insert typedef' in line:
        newline =  f"typedef {type_string} T;\n"
      fout.write(newline)
    fin.close()
    fout.close()

    curr_dir = os.getcwd()
    os.chdir(cpp_filedir)
    cmd = f"g++ -O3 -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) {_ap_include()} {_gcc_opts()} {self.sani_type}.cpp -o {self.sani_type}.so"
    logger.debug(f'Compiling with command {cmd}')
    try:
      ret_val = os.system(cmd)
      if ret_val != 0:
        raise Exception(f'Failed to compile FixedPointConverter {self.sani_type}.cpp')
    finally:
      os.chdir(curr_dir)

    os.chdir(cpp_filedir)
    logger.debug(f'Importing compiled module {self.sani_type}.so')
    try:
      import importlib.util
      spec = importlib.util.spec_from_file_location('fixed_point', f'{self.sani_type}.so')
      self.lib = importlib.util.module_from_spec(spec)
      spec.loader.exec_module(self.lib)
    except ImportError:
      os.chdir(curr_dir)
      raise Exception("Can't import pybind11 bridge, is it compiled?")
    os.chdir(curr_dir)

  def _parse(self, type_string):
    parsed = parse_ap_type(type_string)
    if parsed is None:
      logger.error(f'Could not parse {type_string}')
      return
    self.width = parsed.width
    self.integer_bits = parsed.integer_bits
    self.fractional_bits = parsed.fractional_bits
    self.signed = parsed.signed
    self.rounding_mode = parsed.rounding_mode
    self.overflow_mode = parsed.overflow_mode
    self.saturation_bits = parsed.saturation_bits

  def to_int(self, x):
    return self.lib.to_int(x)

  def to_double(self, x):
    return self.lib.to_double(x)

  def from_int(self, x):
    return self.lib.from_int(x)
