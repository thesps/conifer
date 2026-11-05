import re
import numpy as np
from quantizers import fixed_quantize
import logging
logger = logging.getLogger(__name__)

# conifer precision strings use ap_ types, quantizers names the same modes without the AP_ prefix
_ROUND_MODES = ('TRN', 'TRN_ZERO', 'RND', 'RND_ZERO', 'RND_INF', 'RND_MIN_INF', 'RND_CONV')
_OVERFLOW_MODES = ('WRAP', 'SAT', 'SAT_ZERO', 'SAT_SYM')
_MAX_EXACT_WIDTH = 53 # conversions go through double precision floats

class FixedPointConverter:
  '''
  Emulate ap_fixed and ap_int number representations in numpy, using the quantizers package
  Conversions accept scalars or arrays, and return the same shape
  '''

  def __init__(self, type_string):
    '''
    Construct the FixedPointConverter
    args:
      type_string : string for the ap_ type, e.g. ap_fixed<16,6,AP_RND,AP_SAT>
    '''
    self.type_string = type_string
    self._parse(type_string)
    self._k = int(self.signed)
    self._i = self.integer_bits - self._k
    self._f = self.fractional_bits
    logger.debug(f'Constructed converter for {type_string}')

  def _parse(self, type_string):
    m = re.fullmatch(r'\s*ap_(fixed|ufixed|int|uint)\s*<(.*)>\s*', type_string)
    if m is None:
      raise ValueError(f'Could not parse fixed point type {type_string}')
    kind = m.group(1)
    args = [arg.strip() for arg in m.group(2).split(',')]
    self.signed = kind in ('fixed', 'int')
    n_args = (1, 1) if kind in ('int', 'uint') else (2, 5)
    if not n_args[0] <= len(args) <= n_args[1]:
      raise ValueError(f'Could not parse fixed point type {type_string}')
    self.width = int(args[0])
    self.integer_bits = self.width if kind in ('int', 'uint') else int(args[1])
    self.fractional_bits = self.width - self.integer_bits
    # casting a float to ap_int truncates towards zero, while ap_fixed defaults to AP_TRN
    default_round = 'AP_TRN_ZERO' if kind in ('int', 'uint') else 'AP_TRN'
    self.round_mode = self._parse_mode(args[2] if len(args) > 2 else default_round, _ROUND_MODES, type_string)
    self.overflow_mode = self._parse_mode(args[3] if len(args) > 3 else 'AP_WRAP', _OVERFLOW_MODES, type_string)
    if len(args) > 4 and int(args[4]) != 0:
      raise ValueError(f'Non-zero saturation bits are not supported: {type_string}')
    if self.signed and self.width == 1 and self.overflow_mode == 'SAT_SYM':
      raise ValueError(f'1-bit signed AP_SAT_SYM is not supported: {type_string}') # quantizers gives 0, ap_fixed gives -1
    if self.width > _MAX_EXACT_WIDTH:
      logger.warning(f'{type_string} is too wide to convert exactly using double precision')

  @staticmethod
  def _parse_mode(mode, modes, type_string):
    if not mode.startswith('AP_') or mode[3:] not in modes:
      raise ValueError(f'Unsupported mode {mode} in {type_string}, expected one of {", ".join("AP_" + m for m in modes)}')
    return mode[3:]

  def _quantize(self, x, round_mode, overflow_mode):
    x = np.asarray(x, dtype=np.float64)
    # match the ap_ types for non-finite values: NaN behaves as +inf, and inf only survives saturation
    x = np.where(np.isnan(x), np.inf, x)
    if overflow_mode in ('WRAP', 'SAT_ZERO'):
      x = np.where(np.isinf(x), 0., x)
    if overflow_mode == 'WRAP':
      # exact, unlike the wrap in quantizers which loses the sign bias for large x
      x = np.fmod(x, 2.0 ** self.integer_bits)
    return fixed_quantize(x, self._k, self._i, self._f, round_mode, overflow_mode)

  @staticmethod
  def _unwrap(x):
    return x.item() if np.ndim(x) == 0 else x

  def to_int(self, x):
    '''Get the integer representation of the fixed point value of x'''
    y = self._quantize(x, self.round_mode, self.overflow_mode)
    return self._unwrap(np.ldexp(y, self._f).astype(np.int64))

  def to_double(self, x):
    '''Get the double representation of the fixed point value of x'''
    return self._unwrap(self._quantize(x, self.round_mode, self.overflow_mode))

  def from_int(self, x):
    '''Set the underlying bits of the fixed point value from integer x, return the double'''
    y = np.ldexp(np.asarray(x, dtype=np.float64), -self._f)
    return self._unwrap(self._quantize(y, 'TRN', 'WRAP'))
