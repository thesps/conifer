"""Example of BDT creation with Yggdrasil Decision Forests (https://ydf.readthedocs.io)."""

import datetime
import logging
import sys

import numpy as np
import ydf
from sklearn.datasets import make_hastie_10_2
import conifer

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

# Create dataset
X, y = make_hastie_10_2(random_state=0)
y = y == 1  # Converts Hastie's labels from {-1, 1} to {False, True}.

# Train a Gradient Boosted Decision Trees model using YDF.
model = ydf.GradientBoostedTreesLearner(
    num_trees=1,
    max_depth=2,
    apply_link_function=False,
    label="y",
).train({"x": X, "y": y})

# Create a Conifer configuration.
stamp = int(datetime.datetime.now().timestamp())
cpp_cfg = conifer.backends.cpp.auto_config()
# cpp_cfg["Precision"] = "float"  # Optional float precision.
cpp_cfg["OutputDir"] = "prj_cpp_{}".format(stamp)

# Convert the YDF model to a C++ Conifer model.
cpp_model = conifer.converters.convert_from_ydf(model, cpp_cfg)
cpp_model.compile()

# Create a conifer config
hls_cfg = conifer.backends.xilinxhls.auto_config()
# hls_cfg["Precision"] = "float"  # Optional float precision.
hls_cfg["OutputDir"] = "prj_hls_{}".format(stamp)

# Convert the YDF model to a HLS Conifer model.
hls_model = conifer.converters.convert_from_ydf(model, hls_cfg)
hls_model.compile()

# Compare the predictions of YDF, C++ Conifer, and HLS Conifer model.
y_cpp = cpp_model.decision_function(X)
y_hls = hls_model.decision_function(X)
y_ydf = model.predict({"x": X})


if np.array_equal(y_hls, y_cpp):
    print(f"HLS and CPP predictions agree 100% ({len(y_cpp)}/{len(y_cpp)})")
else:
    abs_diff = np.abs(y_hls - y_cpp)
    rel_diff = abs_diff / np.abs(y_hls)
    print(
        f"HLS and CPP predictions disagree. Biggest absolute difference: {abs_diff.max():.4f}, biggest relative difference: {rel_diff.max():.4f}"
    )
