import sys
import importlib.util

from conifer.backends import xilinxhls
from conifer.backends import vhdl
from conifer.backends import cpp
from conifer.backends import cpp_qs
from conifer.backends import fpu
from conifer.backends import python
from conifer.backends import boards

_backend_map = {'xilinxhls' : xilinxhls,
                'vhdl'      : vhdl,
                'cpp'       : cpp,
                'cpp_qs'    : cpp_qs,
                'fpu'       : fpu,
                'python'    : python,
                'py'        : python,
                }

def get_backend(backend):
  '''Get backend object from string'''
  backend_obj = _backend_map.get(backend)
  if backend_obj is None:
    raise RuntimeError(f'No backend "{backend}" found. Options are {[k for k in _backend_map.keys()]}')
  return backend_obj

def get_available_backends():
  return [k for k in _backend_map.keys()]
