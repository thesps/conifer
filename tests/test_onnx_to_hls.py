import pytest
import util
import onnxmltools
import onnx

@pytest.fixture
def train_onnx():
    # Example BDT creation from: https://scikit-learn.org/stable/modules/ensemble.html
    from sklearn.datasets import make_hastie_10_2
    from sklearn.ensemble import GradientBoostingClassifier

    # Make a random dataset from onnxearn 'hastie'
    X, y = make_hastie_10_2(random_state=0)
    X_train, X_test = X[:2000], X[2000:]
    y_train, y_test = y[:2000], y[2000:]

    # Train a BDT
    clf = GradientBoostingClassifier(n_estimators=20, learning_rate=1.0,
        max_depth=3, random_state=0).fit(X_train, y_train)
    

    from skl2onnx.common.data_types import FloatTensorType
    initial_type = [('float_input', FloatTensorType([None, X_train.shape[1]]))]
    clf = onnxmltools.convert_sklearn(clf, 'Hastie model', initial_types=initial_type)
    onnx.save(clf, 'hastie_bdt.onnx')

    return clf, X_test, y_test, 'hastie_bdt.onnx'
    
@pytest.fixture
def hls_convert(train_onnx):
    import conifer
    import datetime

    clf, X, y, name = train_onnx

    # Create a conifer config
    cfg = conifer.backends.xilinxhls.auto_config()
    cfg['Precision'] = 'ap_fixed<32,16,AP_RND,AP_SAT>'
    # Set the output directory to something unique
    cfg['OutputDir'] = 'prj_{}'.format(int(datetime.datetime.now().timestamp()))
    cfg['XilinxPart'] = 'xcu250-figd2104-2L-e'

    # Create and compile the model
    model = conifer.converters.convert_from_onnx(clf, cfg)
    model.compile()
    return model

@pytest.fixture
def vhdl_convert(train_onnx):
    import conifer
    import datetime

    clf, X, y, name = train_onnx

    # Create a conifer config
    cfg = conifer.backends.vhdl.auto_config()
    cfg['Precision'] = 'ap_fixed<32,16>'
    # Set the output directory to something unique
    cfg['OutputDir'] = 'prj_{}'.format(int(datetime.datetime.now().timestamp()))
    cfg['XilinxPart'] = 'xcu250-figd2104-2L-e'

    # Create and compile the model
    model = conifer.converters.convert_from_onnx(clf, cfg)
    model.compile()
    return model

@pytest.fixture
def predict(train_onnx, hls_convert):
    clf, X, y, name = train_onnx
    model = hls_convert
    return util.predict_onnx(name, X, y, model)

def test_onnx_hls_predict(train_onnx, hls_convert):
    import numpy as np
    clf, X, y, name = train_onnx
    model = hls_convert
    y_hls, y_onnx = util.predict_onnx(name, X, y, model)
    assert np.all(np.isclose(y_hls, y_onnx, rtol=1e-2, atol=1e-2))

def test_onnx_build(hls_convert):
    model = hls_convert
    model.build()
    assert True

def test_hdl_predict(train_onnx, vhdl_convert):
    import numpy as np
    clf, X, y, name = train_onnx
    model = vhdl_convert
    y_hls, y_onnx = util.predict_onnx(name, X, y, model)
    assert np.all(np.isclose(y_hls, y_onnx, rtol=1e-2, atol=1e-2))

def test_hdl_build(vhdl_convert):
    model = vhdl_convert
    model.build()
    assert True


