import pytest
import util
import numpy as np

def test_skl_hls_predict(predict):
    y_hls, y_skl = predict
    assert np.all(np.isclose(y_hls, y_skl, rtol=1e-2, atol=1e-2))

def test_skl_build(hls_convert):
    model = hls_convert
    success = model.build()
    assert success, "Build failed"

def test_hdl_predict(train_skl, vhdl_convert):
    clf, X, y = train_skl
    model = vhdl_convert
    y_hdl, y_skl = util.predict_skl(clf, X, y, model)
    assert np.all(np.isclose(y_hdl, y_skl, rtol=1e-2, atol=1e-2))

def test_hdl_build(vhdl_convert):
    model = vhdl_convert
    success = model.build()
    assert success, "Build failed"


