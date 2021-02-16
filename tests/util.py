import pytest

def predict(clf, X, y, model):
    # Run HLS C Simulation and get the output
    y_hls = model.decision_function(X)
    y_skl = clf.decision_function(X)
    return y_hls, y_skl
