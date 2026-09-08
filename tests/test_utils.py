from sklearn.datasets import load_iris
from sklearn.ensemble import GradientBoostingClassifier
import conifer
import numpy as np
import pytest
from conifer.utils.fixed_point import FixedPointConverter, parse_ap_type

@pytest.fixture
def model():
  # TODO replace this with dummy-model generation
  X, y = load_iris(return_X_y=True)
  # Train a GradientBoostingClassifier
  clf = GradientBoostingClassifier(n_estimators=20, max_depth=3).fit(X, y)
  return clf, X, y

def test_profiling(model):
  clf, X, y = model
  conifer_model = conifer.converters.convert_from_sklearn(clf)
  fig = conifer_model.profile(return_figure=True)

def test_model_draw(model):
  clf, X, y = model
  conifer_model = conifer.converters.convert_from_sklearn(clf)
  graph = conifer_model.draw()  

def test_tree_draw(model):
  clf, X, y = model
  conifer_model = conifer.converters.convert_from_sklearn(clf)
  tree = conifer_model.trees[0][0]
  graph = tree.draw()

@pytest.fixture
def get_fpc():
  fpc = conifer.utils.FixedPointConverter('ap_fixed<8,4,AP_RND_CONV,AP_SAT>')
  return fpc

@pytest.fixture
def get_fpc_xy():
  x_fps = [1.0, -1.0, 1.5, 2**-4, 1.0+2**-4, -2.**3, 2**3-2**-4]
  x_ints = [16, -16, 24, 1, 17, -128, 127]
  return x_fps, x_ints  

def test_fixed_point_converter_to_int(get_fpc, get_fpc_xy):
  fpc = get_fpc
  x_fps, x_ints = get_fpc_xy
  y_ints = [fpc.to_int(x) for x in x_fps]
  np.testing.assert_equal(x_ints, y_ints)

def test_fixed_point_converter_from_int(get_fpc, get_fpc_xy):
  fpc = get_fpc
  x_fps, x_ints = get_fpc_xy
  y_fps = [fpc.from_int(x) for x in x_ints]
  np.testing.assert_equal(x_fps, y_fps)

@pytest.mark.parametrize('type_string,width,int_bits,signed,rnd,ovf,satbits', [
  ('ap_fixed<18,8>', 18, 8, True, 'AP_TRN', 'AP_WRAP', 0),                       # defaults
  ('ap_ufixed<10, 4>', 10, 4, False, 'AP_TRN', 'AP_WRAP', 0),
  ('ap_fixed<16,6,AP_RND,AP_SAT>', 16, 6, True, 'AP_RND', 'AP_SAT', 0),
  ('ap_fixed<16,6,AP_RND_CONV>', 16, 6, True, 'AP_RND_CONV', 'AP_WRAP', 0),      # overflow mode omitted
  ('ap_fixed<16,6,AP_TRN,AP_WRAP_SM,3>', 16, 6, True, 'AP_TRN', 'AP_WRAP_SM', 3),
  ('ap_int<16>', 16, 16, True, None, None, None),
  ('ap_uint<8>', 8, 8, False, None, None, None),
])
def test_parse_ap_type(type_string, width, int_bits, signed, rnd, ovf, satbits):
  parsed = parse_ap_type(type_string)
  assert (parsed.width, parsed.integer_bits, parsed.signed) == (width, int_bits, signed)
  assert (parsed.rounding_mode, parsed.overflow_mode, parsed.saturation_bits) == (rnd, ovf, satbits)

def test_parse_ap_type_unparseable():
  assert parse_ap_type('float') is None
  assert parse_ap_type(None) is None

def test_fixed_point_converter_parse_uses_shared_parser(monkeypatch):
  # avoid compiling the C++ bridge; only exercise _parse()
  monkeypatch.setattr(FixedPointConverter, '__init__', lambda self, t: self._parse(t))
  fpc = FixedPointConverter('ap_fixed<12,4,AP_RND,AP_SAT>')
  assert (fpc.width, fpc.integer_bits, fpc.fractional_bits) == (12, 4, 8)
  assert (fpc.rounding_mode, fpc.overflow_mode) == ('AP_RND', 'AP_SAT')