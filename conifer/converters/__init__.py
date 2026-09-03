import logging
logger = logging.getLogger(__name__)

#the splitting convention for onnx models is defined in the model itself. https://github.com/onnx/onnx/blob/a6b828cdabfb5c0f8795d82e6a3851224acecd10/onnx/defs/traditionalml/defs.cc#L1051-L1057
# User can impose a different splitting convention by setting it in the dictionary splitting_conventions['onnx']='<splitting convention>'
splitting_conventions = {
  "xgboost": "<",  #https://github.com/dmlc/xgboost/blob/9715661c09e61fad15c58ffd059fc0db87fa5d44/plugin/sycl/predictor/predictor.cc#L131C1-L147C2
  "sklearn": "<=", #https://github.com/scikit-learn/scikit-learn/blob/d8932866b6f4b2dee508a54b79f1122ff5f5459d/sklearn/ensemble/_gradient_boosting.pyx#L68-L73
  "tmva": "<=",    #https://github.com/root-project/root/blob/87f151c3a55a33380937a31be65e1f102796770f/tmva/tmva/src/BinarySearchTreeNode.cxx#L120-L133
  "ydf": "<"       #https://github.com/google/yggdrasil-decision-forests/blob/12a83b84859089c508eb4c53b210f49e7bd44c49/yggdrasil_decision_forests/port/python/ydf/model/tree/condition.py#L81-L94
}

import importlib
import importlib.util

_CONVERTERS = ['sklearn', 'tmva', 'xgboost', 'onnx', 'ydf']
_converter_map = {}

def _load_converter(name):
  '''
  Import and cache the conifer converter submodule for `name`.
  Returns the module, or None if it (or its third-party dependency) is unavailable.
  Done lazily so `import conifer` doesn't pull in xgboost/sklearn/ydf etc.
  '''
  if name not in _CONVERTERS:
    return None
  if name not in _converter_map:
    try:
      _converter_map[name] = importlib.import_module(f'conifer.converters.{name}')
    except ImportError:
      logger.warning(f'Could not import conifer {name} converter')
      _converter_map[name] = None
  return _converter_map[name]

def __getattr__(name):
  '''Lazily expose converter submodules as attributes, e.g. conifer.converters.xgboost.'''
  if name in _CONVERTERS:
    module = _load_converter(name)
    if module is None:
      raise AttributeError(f'conifer {name} converter is not available')
    return module
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

from conifer.model import make_model

def get_converter(converter):
  '''Get converter object from string'''
  converter_obj = _load_converter(converter)
  if converter_obj is None:
    raise RuntimeError(f'No converter "{converter}" found. Options are {get_available_converters()}')
  return converter_obj

def get_available_converters():
  return [name for name in _CONVERTERS if _load_converter(name) is not None]

def convert_from_sklearn(model, config=None):
  '''Convert a BDT from a scikit-learn model and configuration'''
  ensembleDict = get_converter('sklearn').convert(model)
  return make_model(ensembleDict, config)

def convert_from_tmva(model, config=None):
  '''Convert a BDT from a TMVA model and configuration'''
  ensembleDict = get_converter('tmva').convert(model)
  return make_model(ensembleDict, config)

def convert_from_xgboost(model, config=None):
  '''Convert a BDT from an xgboost model and configuration'''
  ensembleDict = get_converter('xgboost').convert(model)
  return make_model(ensembleDict, config)

def convert_from_onnx(model, config=None):
  '''Convert a BDT from an ONNX model and configuration'''
  ensembleDict = get_converter('onnx').convert(model)
  return make_model(ensembleDict, config)

def convert_from_ydf(model, config=None):
  '''Convert a BDT from an Yggdrasil Decision Forests model and configuration'''
  ensembleDict = get_converter('ydf').convert(model)
  return make_model(ensembleDict, config)
