import importlib

class python_backend:
  '''
  Simple backend to make a ModelBase object
  '''
  def make_model(ensembleDict, config):
    from conifer.model import ModelBase
    return ModelBase(ensembleDict, config)

# Submodules that can be reached as attributes (conifer.backends.xilinxhls, ...).
# Imported lazily so `import conifer` doesn't pull in every backend (and their
# tool discovery / heavy dependencies).
_BACKEND_SUBMODULES = ['xilinxhls', 'vhdl', 'cpp', 'fpu', 'boards']
# String keys accepted by get_backend(); 'boards' is a helper module, not a backend.
_BACKEND_NAMES = ['xilinxhls', 'vhdl', 'cpp', 'fpu', 'python', 'py']
_submodule_cache = {}

def _load_backend(name):
  '''Return the backend object for `name`, importing its submodule lazily and caching it.'''
  if name in ('python', 'py'):
    return python_backend
  if name not in _BACKEND_SUBMODULES:
    return None
  if name not in _submodule_cache:
    _submodule_cache[name] = importlib.import_module(f'conifer.backends.{name}')
  return _submodule_cache[name]

def __getattr__(name):
  '''Lazily expose backend submodules as attributes, e.g. conifer.backends.xilinxhls.'''
  if name in _BACKEND_SUBMODULES:
    return _load_backend(name)
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

def get_backend(backend):
  '''Get backend object from string'''
  backend_obj = _load_backend(backend) if backend in _BACKEND_NAMES else None
  if backend_obj is None:
    raise RuntimeError(f'No backend "{backend}" found. Options are {get_available_backends()}')
  return backend_obj

def get_available_backends():
  return list(_BACKEND_NAMES)
