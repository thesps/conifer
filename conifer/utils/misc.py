import os
import sys
import logging
logger = logging.getLogger(__name__)

def _ap_include():
  '''
  Get the -I include option for the g++ compile for the ap_ headers.
  If set use XILINX_AP_INCLUDE, then check for Vivado/Vitis HLS
  '''
  externals_dir = os.path.join(os.path.dirname(__file__),"../externals")
  if "XILINX_AP_INCLUDE" not in os.environ:
    os.environ["XILINX_AP_INCLUDE"] = os.path.join(externals_dir, "Vitis_HLS/simulation_headers/include")
  ret = None
  variables = [('XILINX_AP_INCLUDE', ''),
               ('XILINX_HLS', '/include'),
               ('XILINX_VIVADO', '/include')]
  for var, include in variables:
    if os.environ.get(var) is not None and os.path.isdir(f'{os.environ.get(var)}/{include}'):
      ret = f'-I{os.environ.get(var)}/{include} -I{externals_dir}'
      logger.debug(f'Including ap_ headers from {var}: {ret}')
      break
  if ret is None:
    logger.warn(f'Could not find ap_ headers (e.g., ap_fixed.h). None of {", ".join([var[0] for var in variables])} are defined')
  return ret

def _json_include():
  '''
  Get the -I include option for the g++ compile for JSON headers.
  '''
  os.environ["JSON_ROOT"] = os.path.join(os.path.dirname(__file__),"../externals/nlohmann/include")
  if os.environ.get("JSON_ROOT") is not None:
    ret = f'-I{os.environ.get("JSON_ROOT")}'
    logger.debug(f'Include JSON headers from JSON_ROOT: {ret}')
  else:
    ret = None
    logger.warn('Could not find JSON headers. JSON_ROOT not defined')
  return ret

def _gcc_opts():
  '''
  Get extra platform specific g++ compile options
  '''
  if sys.platform == 'darwin':
     return '-undefined dynamic_lookup'
  else:
     return ''

def _py_executable():
  '''
  Get the python executable
  '''
  return sys.executable

def copydocstring(fromfunc, sep="\n"):
  """
  Decorator: Copy the docstring of `fromfunc`
  """
  def _decorator(func):
      sourcedoc = fromfunc.__doc__
      if func.__doc__ == None:
          func.__doc__ = sourcedoc
      else:
          func.__doc__ = sep.join([sourcedoc, func.__doc__])
      return func
  return _decorator