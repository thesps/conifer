from sklearn.datasets import make_hastie_10_2, load_diabetes, load_iris
import xgboost as xgb
import conifer
import datetime
from scipy.special import expit, logit, softmax
import numpy as np
import pytest
import logging
import sys

#logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

# Make a random dataset from sklearn 'hastie'
# binary classification
def hastie():
  X, y = make_hastie_10_2(random_state=0)
  y[y == -1] = 0
  return X, y, xgb.DMatrix(X, label=y)

# multiclass classification
def iris():
  X, y = load_iris(return_X_y=True)
  return X, y, xgb.DMatrix(X, label=y)

# regression
def diabetes():
  X, y = load_diabetes(return_X_y=True)
  return X, y, xgb.DMatrix(X, label=y)

def model_0(d, kwarg_params={}):
  # Train a BDT
  param = {'max_depth': 3, 'eta': 1, 'objective': 'binary:logistic'}
  param.update(kwarg_params)
  num_round = 20  # num_round is equivalent to number of trees
  bst = xgb.train(param, d, num_round)
  return bst

def model_1(d, kwarg_params={}):
  # Train a BDT
  param = {'max_depth': 3, 'eta': 1, 'objective': 'binary:logistic', 'updater':'grow_histmaker'}
  param.update(kwarg_params)
  num_round = 20  # num_round is equivalent to number of trees
  bst = xgb.train(param, d, num_round)
  return bst

def model_2(X, y, kwarg_params={}):
  bdt = xgb.XGBClassifier(n_estimators=10, max_depth=3, **kwarg_params)
  bdt.fit(X, y)  
  return bdt

def model_3(X, y, kwarg_params={}):
  bdt = xgb.XGBRegressor(n_estimators=10, max_depth=3, **kwarg_params)
  bdt.fit(X, y)  
  return bdt

# parameter format: (test index, model function, data function, data fmt ['np', 'xgb'], predictor function, prediction transform)
@pytest.mark.parametrize('params', [(0, model_0, {}, hastie, 'xgb', 'predict', expit, {}), 
                                    (1, model_1, {}, hastie, 'xgb', 'predict', expit, {}),
                                    (2, model_2, {}, hastie, 'np', 'predict_proba', expit, {}),
                                    (3, model_2, {}, iris, 'np', 'predict_proba', softmax, {'axis':1}),
                                    (4, model_0, {'objective': 'multi:softprob', 'num_class': 3}, iris, 'xgb', 'predict', softmax, {'axis':1}),
                                    (5, model_3, {}, diabetes, 'np', 'predict', lambda x: x, {}), # not yet possible
                                    ])
def test_xgb(params):
  test_idx = params[0]
  get_model_function = params[1]
  get_model_kwargs = params[2]
  get_data_function = params[3]
  data_fmt = params[4]
  predictor = params[5]
  transform = params[6]
  transform_kwargs = params[7]
  X, y, d = get_data_function()
  if data_fmt == 'xgb':
    model = get_model_function(d, get_model_kwargs)
  else:
    model = get_model_function(X, y, get_model_kwargs)
  cfg = conifer.backends.cpp.auto_config()
  cfg['Precision'] = 'double'
  # Set the output directory to something unique
  cfg['OutputDir'] = f'prj_xgb_converter_{test_idx}_{int(datetime.datetime.now().timestamp())}'
  cnf_model = conifer.converters.convert_from_xgboost(model, cfg)
  cnf_model.compile()
  y_cnf = np.squeeze(transform(cnf_model.decision_function(X), **transform_kwargs))
  X_xgb = X if data_fmt == 'np' else d
  y_xgb = getattr(model, predictor)(X_xgb)
  if len(y_xgb.shape) == 2 and y_xgb.shape[1] == 2:
    y_xgb = y_xgb[:,-1]
  np.testing.assert_array_almost_equal(y_cnf, y_xgb)
