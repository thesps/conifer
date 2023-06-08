from sklearn.datasets import make_hastie_10_2
import xgboost as xgb
import conifer
import datetime
from scipy.special import expit, logit
import numpy as np
import pytest
import logging
import sys

#logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

# Make a random dataset from sklearn 'hastie'
X, y = make_hastie_10_2(random_state=0)
y[y == -1] = 0
X_train, X_test = X[:2000], X[2000:]
y_train, y_test = y[:2000], y[2000:]

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

def get_dmatrix():
  return dtest

def get_np():
  return X_test

def model_0():
  # Train a BDT
  param = {'max_depth': 3, 'eta': 1, 'objective': 'binary:logistic'}
  num_round = 20  # num_round is equivalent to number of trees
  bst = xgb.train(param, dtrain, num_round)
  return bst

def model_1():
  # Train a BDT
  param = {'max_depth': 3, 'eta': 1, 'objective': 'binary:logistic', 'updater':'grow_histmaker'}
  num_round = 20  # num_round is equivalent to number of trees
  bst = xgb.train(param, dtrain, num_round)
  return bst

def model_2():
  bdt = xgb.XGBClassifier(n_estimators=10, max_depth=3)
  bdt.fit(X_train, y_train)  
  return bdt

def model_3():
  bdt = xgb.XGBRegressor(n_estimators=10, max_depth=3)
  bdt.fit(X_train, y_train)  
  return bdt

def predict_logit(model, x):
  return logit(model.predict(x))

def predict_proba(model, x):
  return logit(model.predict_proba(x)[:,1])

def predict(model, x):
  return logit(model.predict(x))

@pytest.mark.parametrize('params', [(0, model_0, get_dmatrix, get_np, 'predict', expit), 
                                    (1, model_1, get_dmatrix, get_np, 'predict', expit),
                                    (2, model_2, get_np, get_np, 'predict_proba', expit),
                                    #(3, model_3, get_np, get_np, 'predict', lambda x: x), # not yet possible
                                    ])
def test_xgb(params):
  test_idx = params[0]
  get_model_function = params[1]
  get_data_function_xgb = params[2]
  get_data_function_cnf = params[3]
  predictor = params[4]
  transform = params[5]
  model = get_model_function()
  x_xgb = get_data_function_xgb()
  x_cnf = get_data_function_cnf()
  cfg = conifer.backends.cpp.auto_config()
  cfg['Precision'] = 'float'
  # Set the output directory to something unique
  cfg['OutputDir'] = f'prj_xgb_converter_{test_idx}_{int(datetime.datetime.now().timestamp())}'
  cnf_model = conifer.converters.convert_from_xgboost(model, cfg)
  cnf_model.compile()
  y_cnf = np.squeeze(transform(cnf_model.decision_function(x_cnf)))
  y_xgb = getattr(model, predictor)(x_xgb)
  if len(y_xgb.shape) == 2:
    y_xgb = y_xgb[:,-1]
  np.testing.assert_array_almost_equal(y_cnf, y_xgb)
