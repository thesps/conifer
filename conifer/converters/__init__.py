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

_converter_map = {}
import importlib
for module in ['sklearn', 'tmva', 'xgboost', 'onnx', 'ydf']:
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

def convert_from_ydf(model, config=None):
  '''Convert a BDT from an Yggdrasil Decision Forests model and configuration'''
  ensembleDict = ydf.convert(model)
  return make_model(ensembleDict, config)
