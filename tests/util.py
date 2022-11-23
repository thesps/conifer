import pytest
import onnxruntime as rt
import numpy as np
from scipy.special import expit

def predict(clf, X, y, model):
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