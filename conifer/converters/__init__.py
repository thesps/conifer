import logging
logger = logging.getLogger(__name__)

_converter_map = {}
import importlib
for module in ['sklearn', 'tmva', 'xgboost', 'onnx', 'tf_df']:
  try:
    the_module = importlib.import_module(f'conifer.converters.{module}')
    _converter_map[module] = the_module
  except ImportError:
    logger.warn(f'Could not import conifer {module} converter')

from conifer.model import make_model

def get_converter(converter):
  '''Get converter object from string'''
  converter_obj = _converter_map.get(converter)
  if converter_obj is None:
    raise RuntimeError(f'No converter "{converter}" found. Options are {[k for k in _converter_map.keys()]}')
  return converter_obj

def get_available_converters():
  return [k for k in _converter_map.keys()]

def convert_from_sklearn(model, config=None):
  '''Convert a BDT from a scikit-learn model and configuration'''
  ensembleDict = sklearn.convert(model)
  return make_model(ensembleDict, config)

def convert_from_tmva(model, config=None):
  '''Convert a BDT from a TMVA model and configuration'''
  ensembleDict = tmva.convert(model)
  return make_model(ensembleDict, config)

def convert_from_xgboost(model, config=None):
  '''Convert a BDT from an xgboost model and configuration'''
  ensembleDict = xgboost.convert(model)
  return make_model(ensembleDict, config)

def convert_from_onnx(model, config=None):
  '''Convert a BDT from an ONNX model and configuration'''
  ensembleDict = onnx.convert(model)
  return make_model(ensembleDict, config)

def convert_from_tf_df(model, config=None):
  '''Convert a BDT from an TF-DF model and configuration'''
  ensembleDict = tf_df.convert(model)
  return make_model(ensembleDict, config)
