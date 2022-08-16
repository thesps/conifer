import pytest
import util

@pytest.fixture
def train_xgb():
    # Example BDT creation from: https://scikit-learn.org/stable/modules/ensemble.html
    from sklearn.datasets import make_hastie_10_2
    import xgboost as xgb

    # Make a random dataset from sklearn 'hastie'
    X, y = make_hastie_10_2(random_state=0)
    # Convert y from -1,1 to 0,1 to suppress XGBoost warnings
    y=((y+1)/2).astype(int) 
    X_train, X_val, X_test = X[:1800], X[1800:2000], X[2000:]
    y_train, y_val, y_test = y[:1800], y[1800:2000], y[2000:]

    # Train XGBoost model
    clf = xgb.XGBClassifier(max_depth = 3, n_estimators = 20, use_label_encoder=False)
    clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=10)
    return clf.get_booster(), X_test, y_test
    
@pytest.fixture
def hls_convert(train_xgb):
    import conifer
    import datetime

    clf, X, y = train_xgb

    # Create a conifer config
    cfg = conifer.backends.xilinxhls.auto_config()
    cfg['Precision'] = 'ap_fixed<32,16,AP_RND,AP_SAT>'
    # Set the output directory to something unique
    cfg['OutputDir'] = 'prj_{}'.format(int(datetime.datetime.now().timestamp()))
    cfg['XilinxPart'] = 'xcu250-figd2104-2L-e'

    # Create and compile the model
    model = conifer.converters.convert_from_xgboost(clf, cfg)
    model.compile()
    return model

@pytest.fixture
def vhdl_convert(train_xgb):
    import conifer
    import datetime

    clf, X, y = train_xgb

    # Create a conifer config
    cfg = conifer.backends.vhdl.auto_config()
    cfg['Precision'] = 'ap_fixed<32,16>'
    # Set the output directory to something unique
    cfg['OutputDir'] = 'prj_{}'.format(int(datetime.datetime.now().timestamp()))
    cfg['XilinxPart'] = 'xcu250-figd2104-2L-e'

    # Create and compile the model
    model = conifer.converters.convert_from_xgboost(clf, cfg)
    model.compile()
    return model

@pytest.fixture
def predict(train_xgb, hls_convert):
    clf, X, y = train_xgb
    model = hls_convert
    return util.predict_xgb(clf, X, y, model)

def test_xgb_hls_predict(predict):
    import numpy as np
    y_hls, y_xgb = predict
    assert np.all(np.isclose(y_hls, y_xgb, rtol=1e-2, atol=1e-2))

def test_xgb_build(hls_convert):
    model = hls_convert
    model.build()
    assert True

def test_hdl_predict(train_xgb, vhdl_convert):
    clf, X, y = train_xgb
    model = vhdl_convert
    return util.predict_xgb(clf, X, y, model)

def test_hdl_build(vhdl_convert):
    model = vhdl_convert
    model.build()
    assert True


