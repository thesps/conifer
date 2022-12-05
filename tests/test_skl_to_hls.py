import pytest
import util

def test_skl_hls_predict(predict):
    import numpy as np
    y_hls, y_skl = predict
    assert np.all(np.isclose(y_hls, y_skl, rtol=1e-2, atol=1e-2))

def test_skl_build(hls_convert):
    model = hls_convert
    model.build()
    assert True

def test_hdl_predict(train_skl, vhdl_convert):
    clf, X, y = train_skl
    model = vhdl_convert
    return util.predict_skl(clf, X, y, model)

def test_hdl_build(vhdl_convert):
    model = vhdl_convert
    model.build()
    assert True


