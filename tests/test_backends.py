'''
Test that the conifer backends yield the same output given the same model, data, and config
'''

import conifer
import pytest
import numpy as np

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

def all_backends_predict(model, X, y_shape, precision):

  hls_cfg = conifer.backends.xilinxhls.auto_config()
  hls_cfg['Precision'] = precision
  hls_cfg['OutputDir'] = f'prj_backend_equality_hls_{precision}'
  hls_model = conifer.model(model, conifer.converters.sklearn, conifer.backends.xilinxhls, hls_cfg)
  hls_model.compile()
  y_hls = hls_model.decision_function(X).reshape(y_shape)

  """
  TODO: only include these in the tests when matching is good
  hdl_cfg = conifer.backends.vhdl.auto_config()
  hdl_cfg['Precision'] = precision
  hdl_cfg['OutputDir'] = f'prj_backend_equality_hdl_{precision}'
  hdl_model = conifer.model(model, conifer.converters.sklearn, conifer.backends.vhdl, hdl_cfg)
  hdl_model.compile()
  y_hdl = hdl_model.decision_function(X).reshape(y_shape)
  """

  cpp_cfg = conifer.backends.cpp.auto_config()
  cpp_cfg['Precision'] = precision
  cpp_cfg['OutputDir'] = f'prj_backend_equality_cpp_{precision}'
  cpp_model = conifer.model(model, conifer.converters.sklearn, conifer.backends.cpp, cpp_cfg)
  cpp_model.compile()
  y_cpp = cpp_model.decision_function(X).reshape(y_shape)

  #return {'hls' : y_hls, 'hdl' : y_hdl, 'cpp' : y_cpp}
  return {'hls' : y_hls, 'cpp' : y_cpp}


model0 = f_train_skl()
tests = [(*model0, 'ap_fixed<16,6>'),
         (*model0, 'ap_fixed<8,4,AP_TRN,AP_WRAP>'),
         (*model0, 'ap_fixed<8,4,AP_RND,AP_WRAP>'),
         (*model0, 'ap_fixed<8,4,AP_RND_ZERO,AP_WRAP>'),
         (*model0, 'ap_fixed<8,4,AP_RND,AP_SAT>'),
         (*model0, 'ap_fixed<18,8>'),
         (*model0, 'ap_fixed<18,8,AP_RND_CONV,AP_SAT>'),
        ]

results = [all_backends_predict(*test) for test in tests]

@pytest.mark.parametrize('predictions', results)
def test_cpp_hls(predictions):
  np.testing.assert_array_equal(predictions['cpp'], predictions['hls'])

"""
  TODO: only include these tests when matching is good
@pytest.mark.parametrize('predictions', results)
def test_cpp_hdl(predictions):
  np.testing.assert_array_equal(predictions['cpp'], predictions['hdl'])

@pytest.mark.parametrize('predictions', results)
def test_hls_hdl(predictions):
  np.testing.assert_array_equal(predictions['hls'], predictions['hdl']) 
"""