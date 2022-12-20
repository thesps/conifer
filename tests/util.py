import pytest
import onnxruntime as rt
import numpy as np
from scipy.special import expit

def predict_skl(clf, X, y, model):
    # Run HLS C Simulation and get the output
    y_hls = model.decision_function(X)
    y_skl = clf.decision_function(X)
    return y_hls, y_skl

def predict_onnx(name, X, y, model):
    sess = rt.InferenceSession(name)
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name
    y_onnx = sess.run([label_name], {input_name: X.astype(np.float32)})[0]
    y_hls_df = model.decision_function(X)
    if len(y_hls_df.shape) == 1 or (len(y_hls_df.shape) == 2 and y_hls_df.shape[1]==1):
        y_hls = np.where(expit(y_hls_df) > 0.5, 1, -1)
    else:
        y_hls = np.argmax(expit(y_hls_df), axis=1)
    return y_hls, y_onnx

@pytest.fixture
def train_skl():
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

    return clf, X_test, y_test

@pytest.fixture
def hls_convert(train_skl):
    import conifer
    import datetime

    clf, X, y = train_skl

    # Create a conifer config
    cfg = conifer.backends.xilinxhls.auto_config()
    cfg['Precision'] = 'ap_fixed<32,16,AP_RND,AP_SAT>'
    # Set the output directory to something unique
    cfg['OutputDir'] = 'prj_{}'.format(int(datetime.datetime.now().timestamp()))
    cfg['XilinxPart'] = 'xcu250-figd2104-2L-e'

    # Create and compile the model
    model = conifer.converters.convert_from_sklearn(clf, cfg)
    model.compile()
    return model

@pytest.fixture
def vhdl_convert(train_skl):
    import conifer
    import datetime

    clf, X, y = train_skl

    # Create a conifer config
    cfg = conifer.backends.vhdl.auto_config()
    cfg['Precision'] = 'ap_fixed<32,16>'
    # Set the output directory to something unique
    cfg['OutputDir'] = 'prj_{}'.format(int(datetime.datetime.now().timestamp()))
    cfg['XilinxPart'] = 'xcu250-figd2104-2L-e'

    # Create and compile the model
    model = conifer.converters.convert_from_sklearn(clf, cfg)
    model.compile()
    return model

@pytest.fixture
def predict(train_skl, hls_convert):
    clf, X, y = train_skl
    model = hls_convert
    return predict_skl(clf, X, y, model)