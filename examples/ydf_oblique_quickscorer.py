"""QuickScorer on an oblique BDT, checked against the plain tree walk.

QuickScorer (https://doi.org/10.1145/2766462.2767733) groups nodes by the quantity
they compare, sorts each group's thresholds and retires a whole prefix of a group
per comparison. Axis-aligned splits compare a feature, so there are only n_features
groups. Oblique splits compare a projection w.x, so nearly every node ends up in a
group of its own. The traversal is correct, and performance was not optimized.
"""

import datetime
import logging
import sys

import numpy as np
import ydf
from sklearn.datasets import make_hastie_10_2

import conifer
from conifer.backends.python.quickscorer import QuickScorer

logging.basicConfig(stream=sys.stdout, level=logging.WARNING)

# Create dataset.
X, y = make_hastie_10_2(random_state=0)
y = y == 1  # Converts Hastie's labels from {-1, 1} to {False, True}.

# Train a sparse oblique GBT with YDF.
model = ydf.GradientBoostedTreesLearner(
    num_trees=100,
    max_depth=4,
    apply_link_function=False,
    split_axis="SPARSE_OBLIQUE",
    label="y",
).train({"x": X, "y": y})

stamp = int(datetime.datetime.now().timestamp())

# Convert to the python backend (floating point reference).
py_cfg = conifer.backends.python.auto_config()
py_cfg["Tau"], py_cfg["Delta"] = 32, 4096  # blockwise QuickScorer block shape
py_model = conifer.converters.convert_from_ydf(model, py_cfg)

# And to the cpp backend (bit accurate emulation of the hardware arithmetic). The
# compare type also accumulates the projection, so it has to hold the range of w.x
# rather than that of the features.
cpp_cfg = conifer.backends.cpp.auto_config()
cpp_cfg["Precision"] = "ap_fixed<18,8>"
cpp_cfg["Tau"], cpp_cfg["Delta"] = 32, 16
cpp_cfg["OutputDir"] = "prj_cpp_{}".format(stamp)
cpp_model = conifer.converters.convert_from_ydf(model, cpp_cfg)
cpp_model.compile()

# The traversals sum the same leaf values in the same order, oblique splits or not,
# so swapping one for another must not move a single bit.
for name, conifer_model in [("python", py_model), ("cpp", cpp_model)]:
    walk = np.squeeze(conifer_model.decision_function(X, algorithm="treewalk"))
    for algorithm in ("quickscorer", "blockwise-quickscorer"):
        qs = np.squeeze(conifer_model.decision_function(X, algorithm=algorithm))
        if np.array_equal(walk, qs):
            print(f"{name}/{algorithm} and the tree walk agree 100% ({len(qs)}/{len(qs)})")
        else:
            print(f"{name}/{algorithm} and the tree walk disagree. Biggest absolute "
                  f"difference: {np.abs(walk - qs).max():.4g}")

# The grouping is what oblique splits change: compare the number of groups the
# traversal scans against the number of nodes in the ensemble.
sizes = np.diff(QuickScorer(py_model).blocks[0][1].offsets)
print(f"{sizes.sum()} internal nodes in {np.count_nonzero(sizes)} groups, "
      f"longest group {sizes.max()}")
