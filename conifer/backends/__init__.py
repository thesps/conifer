import importlib
import sys

class python_backend:
  '''
  Simple backend to make a ModelBase object
  '''
  def make_model(ensembleDict, config):
    from conifer.model import ModelBase
    return ModelBase(ensembleDict, config)

# Backend string -> attribute name, resolved lazily (import / getattr) on first use.
_backend_map = {'xilinxhls' : 'xilinxhls',
                'vhdl'      : 'vhdl',
                'cpp'       : 'cpp',
                'fpu'       : 'fpu',
                'python'    : 'python_backend',
                'py'        : 'python_backend',
                }

# Submodules reachable as conifer.backends.<name> ('boards' is a helper, not a backend).
_backend_submodules = ('xilinxhls', 'vhdl', 'cpp', 'fpu', 'boards')
_submodule_cache = {}

def _load_backend(target):
  '''Resolve a _backend_map value: lazily import the backend submodule, or getattr the named object from this module.'''
  if target in _backend_submodules:
    if target not in _submodule_cache:
      _submodule_cache[target] = importlib.import_module(f'conifer.backends.{target}')
    return _submodule_cache[target]
  return getattr(sys.modules[__name__], target)

def __getattr__(name):
  '''Lazily expose backend submodules as attributes, e.g. conifer.backends.xilinxhls.'''
  if name in _backend_submodules:
    return _load_backend(name)
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

def get_backend(backend):
  '''Get backend object from string'''
  target = _backend_map.get(backend)
  if target is None:
    raise RuntimeError(f'No backend "{backend}" found. Options are {get_available_backends()}')
  return _load_backend(target)

def get_available_backends():
  return [k for k in _backend_map.keys()]
