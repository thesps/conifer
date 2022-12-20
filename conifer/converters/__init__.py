from conifer.converters import sklearn
from conifer.converters import tmva
from conifer.converters import xgboost
from conifer.converters import onnx
from conifer.model import make_model

try:
    from conifer.converters import tf_df
except ImportError:
    tf_df = None
    print("Warning: The python package tensorflow_decision_forests is not available. Conversions "
          "from TensorFlow Decision Forests models is not possible.")

import logging
logger = logging.getLogger(__name__)

_converter_map = {'sklearn' : sklearn,
                  'tmva'    : tmva,
                  'xgboost' : xgboost,
                  'onnx'    : onnx,
                  'tf_df':  tf_df}

def get_converter(converter):
  '''Get converter object from string'''
  converter_obj = _converter_map.get(converter)
  if converter_obj is None:
    raise RuntimeError(f'No converter "{converter}" found. Options are {[k for k in _converter_map.keys()]}')
  return converter_obj

def get_available_converters():
  return [k for k in _converter_map.keys()]

def convert_from_sklearn(model, config):
  '''Convert a BDT from a scikit-learn model and configuration'''
  ensembleDict = sklearn.convert(model)
  return make_model(ensembleDict, config)

def convert_from_tmva(model, config):
  '''Convert a BDT from a TMVA model and configuration'''
  ensembleDict = tmva.convert(model)
  return make_model(ensembleDict, config)

def convert_from_xgboost(model, config):
  '''Convert a BDT from an xgboost model and configuration'''
  ensembleDict = xgboost.convert(model)
  return make_model(ensembleDict, config)

def convert_from_onnx(model, config):
  '''Convert a BDT from an ONNX model and configuration'''
  ensembleDict = onnx.convert(model)
  return make_model(ensembleDict, config)

def convert_from_tf_df(model, config):
  '''Convert a BDT from an TF-DF model and configuration'''
  ensembleDict = tf_df.convert(model)
  return make_model(ensembleDict, config)
