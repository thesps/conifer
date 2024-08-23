import logging
import numpy as np

import conifer

logger = logging.getLogger(__name__)

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


def compute_float_lsb(config):
    """Given a fixed point precision, compute the least significant bit.
    If the precision is float, return None and let the other method compute
    the smallest delta for each float.

    Args:
        config (Dict): backend configuration

    Returns:
        float|None: offset
    """
    if config is None or "Precision" not in config or config["Precision"] == "float":
        # In floating point, the smallest difference depends on the magnitude of the number
        # Use the nextafter numpy method
        return None
    precision = config["Precision"]
    fpc = conifer.utils.FixedPointConverter(precision)
    return fpc.from_int(1)

def subtract_offset_to_thrs(trees, offset):
    """Subtract the least significant bit to the thresholds of the trees.

    Args:
        trees (List): ensembleDict["trees"]
        offset (float|None): offset to subtract from the thresholds.
        If None, the smallest float difference is computed for each threshold float.

    Returns:
        List: ensembleDict["trees"]
    """
    assert isinstance(trees, list)
    if isinstance(trees[0], list):  # needed for multiclass
        for idx, tree in enumerate(trees):
            trees[idx] = subtract_offset_to_thrs(tree, offset)
        return trees
    for idx, tree in enumerate(trees):
        if offset is not None:
            trees[idx]["threshold"] = (np.array(tree["threshold"]) - offset).tolist()
        else:
            trees[idx]["threshold"] = np.nextafter(
                np.array(tree["threshold"], dtype=np.float32), -np.inf
            ).tolist()  # Handling floats
    return trees


def subtract_lsb_to_thrs(trees, config):
    """Compute and subtract the least significant bit to the thresholds of the trees.
    Needed for the libraries with the opposite split convention to conifer.

    Conifer splitting convention: <= or >
    Others (e.g. xgboost): < or >=

    Args:
        trees (List): ensembleDict["trees"]
        config (Dict): backend configuration

    Returns:
        List: ensembleDict["trees"]
    """
    float_lsb = compute_float_lsb(config)
    return subtract_offset_to_thrs(trees, float_lsb)


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
  ensembleDict["trees"]=subtract_lsb_to_thrs(ensembleDict["trees"], config)
  return make_model(ensembleDict, config)

def convert_from_onnx(model, config=None):
  '''Convert a BDT from an ONNX model and configuration'''
  ensembleDict = onnx.convert(model)
  return make_model(ensembleDict, config)

def convert_from_ydf(model, config=None):
  '''Convert a BDT from an Yggdrasil Decision Forests model and configuration'''
  ensembleDict = ydf.convert(model)
  return make_model(ensembleDict, config)
