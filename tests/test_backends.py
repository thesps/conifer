'''
Test that the conifer backends yield the same output given the same model, data, and config
'''

import conifer
import pytest
import numpy as np
import datetime

def f_train_skl():
    # Example BDT creation from: https://scikit-learn.org/stable/modules/ensemble.html
    from sklearn.datasets import make_hastie_10_2
    from sklearn.ensemble import GradientBoostingClassifier

    # Make a random dataset from sklearn 'hastie'
    X, y = make_hastie_10_2(random_state=0)
    X_train, X_test = X[:2000], X[2000:]
    y_train, y_test = y[:2000], y[2000:]

    # Train a BDT
    clf = GradientBoostingClassifier(n_estimators=20, learning_rate=1.0,
        max_depth=3, random_state=0).fit(X_train, y_train)

    return clf, X_test, y_test.shape

def backend_predict(model, odir, X, y_shape, backend, backend_config):
  the_backend = conifer.backends.get_backend(backend)
  cfg = the_backend.auto_config()
  cfg.update(backend_config)
  cfg['OutputDir'] = odir
  model = conifer.converters.convert_from_sklearn(model, cfg)
  model.compile()
  y = model.decision_function(X).reshape(y_shape)   

class Tester:
   def __init__(self, model, X, y_shape, backend_a, backend_b, config_a={}, config_b={}):
      self.model = model
      self.X = X
      self.y_shape = y_shape
      self.backend_a = backend_a
      self.backend_b = backend_b
      self.config_a = config_a
      self.config_b = config_b

model0 = f_train_skl()

# compare pairs of backends with different precision at different rounding modes
hls_cpp_precisions = ['ap_fixed<16,6>', 'ap_fixed<8,4,AP_TRN,AP_WRAP>', 'ap_fixed<8,4,AP_RND,AP_WRAP>', 'ap_fixed<8,4,AP_RND_ZERO,AP_WRAP>',
                      'ap_fixed<8,4,AP_RND,AP_SAT>', 'ap_fixed<18,8>', 'ap_fixed<18,8,AP_RND_CONV,AP_SAT>']
# compare configs with mixed precision
mixed_precision_cfg = {'InputPrecision' : 'ap_fixed<16,6>', 'ScorePrecision' : 'ap_fixed<12,5>'}

tests = [*[Tester(*model0, 'xilinxhls', 'cpp', {'Precision' : p}, {'Precision' : p}) for p in hls_cpp_precisions],
         Tester(*model0, 'xilinxhls', 'vhdl', mixed_precision_cfg, mixed_precision_cfg),
         Tester(*model0, 'xilinxhls', 'xilinxhls', {'Unroll' : True}, {'Unroll' : False})]

@pytest.mark.parametrize('test', tests)
def test_backend_equality(test):
  stamp = int(datetime.datetime.now().timestamp())
  if test.backend_a == test.backend_b:
     name_a, name_b = test.backend_a + '_a', test.backend_a + '_b'
  else:
     name_a, name_b = test.backend_a, test.backend_b
  name_a, name_b = [f'prj_backends_{stamp}_{n}' for n in [name_a, name_b]]
  y_a = backend_predict(test.model, name_a, test.X, test.y_shape, test.backend_a, test.config_a)
  y_b = backend_predict(test.model, name_b, test.X, test.y_shape, test.backend_b, test.config_b)
  np.testing.assert_array_equal(y_a, y_b)

def test_py_backend():
   clf, X, _ = model0
   model = conifer.converters.convert_from_sklearn(clf)
   assert model.config.backend == 'python'
   y_skl = clf.decision_function(X)
   y_cnf = model.decision_function(X)
   np.testing.assert_allclose(y_skl, y_cnf, rtol=1e-6, atol=1e-6)
