import os

def _ap_include():
    '''
    Get the -I include option for the g++ compile for the ap_ headers.
    If set use XILINX_AP_INCLUDE, then check for Vivado/Vitis HLS
    '''
    if os.environ.get("XILINX_AP_INCLUDE") is not None:
        return f'-I{os.environ.get("XILINX_AP_INCLUDE")}'
    elif os.environ.get("XILINX_HLS") is not None and os.path.isdir(f'{os.environ.get("XILINX_HLS")}/include'):
        return f'-I{os.environ.get("XILINX_HLS")}/include'
    elif os.environ.get("XILINX_VIVADO") is not None and os.path.isdir(f'{os.environ.get("XILINX_VIVADO")}/include'):
        return f'-I{os.environ.get("XILINX_VIVADO")}/include'
    else:
        return None

def _json_include():
  '''
  Get the -I include option for the g++ compile for JSON headers.
  '''
  if os.environ.get("JSON_ROOT") is not None:
    return f'-I{os.environ.get("JSON_ROOT")}/include/nlohmann'
  else:
    return None