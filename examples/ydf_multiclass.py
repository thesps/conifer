"""Example of BDT creation with Yggdrasil Decision Forests (https://ydf.readthedocs.io)."""

import datetime
import logging
import sys

import numpy as np
import ydf
from sklearn.datasets import load_iris
import conifer

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

# Create dataset
iris = load_iris()
X, y = iris.data, iris.target

# Train a BDT
model = ydf.GradientBoostedTreesLearner(
    num_trees=100, max_depth=6, apply_link_function=False, label="y"
).train({"x": X, "y": y})

stamp = int(datetime.datetime.now().timestamp())
# Create a conifer config
cpp_cfg = conifer.backends.cpp.auto_config()
# cpp_cfg["Precision"] = "float"
# Set the output directory to something unique
cpp_cfg["OutputDir"] = "prj_cpp_{}".format(stamp)

# Create and compile the model
cpp_model = conifer.converters.convert_from_ydf(model, cpp_cfg)
cpp_model.compile()

# Create a conifer config
hls_cfg = conifer.backends.xilinxhls.auto_config()
# hls_cfg["Precision"] = "float"
# Set the output directory to something unique
hls_cfg["OutputDir"] = "prj_hls_{}".format(stamp)

# Create and compile the model
hls_model = conifer.converters.convert_from_ydf(model, hls_cfg)
hls_model.compile()

# Run the C++ and get the output
y_cpp = cpp_model.decision_function(X)
y_hls = hls_model.decision_function(X)
y_skl = model.predict({"x": X})

if np.array_equal(y_hls, y_cpp):
    print(f"HLS and CPP predictions agree 100% ({len(y_cpp)}/{len(y_cpp)})")
else:
    abs_diff = np.abs(y_hls - y_cpp)
    rel_diff = abs_diff / np.abs(y_hls)
    print(
        f"HLS and CPP predictions disagree. Biggest absolute difference: {abs_diff.max():.4f}, biggest relative difference: {rel_diff.max():.4f}"
    )
