import os
import sys
import logging
import conifer
logger = logging.getLogger(__name__)

def _ap_include():
  '''
  Get the -I include option for the g++ compile for the ap_ and hls_stream headers.
  Default to the conifer bundled headers, but check for XILINX environment variables
  to override.
  '''
  ap_types_path = f'{conifer.__path__[0]}/external/ap_types/include'
  hls_stream_path = f'{conifer.__path__[0]}/external/hls/'
  variables = [('XILINX_AP_INCLUDE', ''),
               ('XILINX_HLS', '/include'),
               ('XILINX_VIVADO', '/include')]
  for var, include in variables:
    if os.environ.get(var) is not None and os.path.isdir(f'{os.environ.get(var)}/{include}'):
      ap_types_path = f'{os.environ.get(var)}/{include}'
      if os.path.exists(f'{os.environ.get(var)}/include/hls_stream.h'):
        hls_stream_path = '' # no need to add hls_stream path again with Vitis/Vivado install
      break
  if not os.path.isdir(ap_types_path):
    logger.warning(f'Could not find ap_ headers (e.g., ap_fixed.h) at {ap_types_path}')
  logger.debug(f'Including ap_ headers from {ap_types_path}')

  if not os.path.isdir(hls_stream_path):
    logger.warning(f'Could not find hls_stream headers at {hls_stream_path}')
  logger.debug(f'Including hls_stream headers from: {hls_stream_path}')

  ret = f'-I{ap_types_path}' + (f' -I{hls_stream_path}' if hls_stream_path != '' else '')
  return ret

def _json_include():
  '''
  Get the -I include option for the g++ compile for JSON headers.
  '''
  json_path = f'{conifer.__path__[0]}/external/json/'
  if not os.path.isdir(json_path):
    logger.warning(f'Could not find JSON headers at expected location: {json_path}')
  ret = f'-I{json_path}'
  logger.debug(f'Including JSON headers from: {json_path}')
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