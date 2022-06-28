import os
import logging
logger = logging.getLogger(__name__)

def _ap_include():
  '''
  Get the -I include option for the g++ compile for the ap_ headers.
  If set use XILINX_AP_INCLUDE, then check for Vivado/Vitis HLS
  '''
  ret = None
  variables = [('XILINX_AP_INCLUDE', ''),
               ('XILINX_HLS', '/include'),
               ('XILINX_VIVADO', '/include')]
  for var, include in variables:
    if os.environ.get(var) is not None and os.path.isdir(f'{os.environ.get(var)}/{include}'):
      ret = f'-I{os.environ.get(var)}/{include}'
      logger.debug(f'Including ap_ headers from {var}: {ret}')
      break
  if ret is None:
    logger.warn(f'Could not find ap_ headers. None of {", ".join([var[0] for var in variables])} are defined')
  return ret

def _json_include():
  '''
  Get the -I include option for the g++ compile for JSON headers.
  '''
  if os.environ.get("JSON_ROOT") is not None:
    ret = f'-I{os.environ.get("JSON_ROOT")}'
    logger.debug(f'Include JSON headers from JSON_ROOT: {ret}')
  else:
    ret = None
    logger.warn('Could not find JSON headers. JSON_ROOT not defined')
  return ret