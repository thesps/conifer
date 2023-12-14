from typing import Tuple
import dataclasses

import conifer
import pytest
import numpy as np
import ydf
from sklearn.datasets import load_iris
from sklearn.datasets import make_hastie_10_2


@dataclasses.dataclass
class Dataset:
    X: np.ndarray
    y: np.ndarray
    X_test: np.ndarray


@pytest.fixture(params=["hastie", "iris"])
def build_dataset(request) -> Dataset:
    if request.param == "hastie":
        X, y = make_hastie_10_2(random_state=0, n_samples=4000)
        return Dataset(X[:2000], y[:2000] == 1, X[2000:])

    elif request.param == "iris":
        iris = load_iris()
        return Dataset(iris.data, iris.target, iris.data)

    else:
        raise ValueError("Unknown dataset")


@pytest.fixture(
    params=[
        "exact",
        # "oblique", # TODO: Enable when supported.
    ]
)
def build_ydf_model(build_dataset, request) -> ydf.GenericModel:
    dataset = build_dataset
    if request.param == "exact":
        extra_kwargs = {}
    elif request.param == "oblique":
        extra_kwargs = {"split_axis": "SPARSE_OBLIQUE"}
    else:
        assert False

    # See https://ydf.readthedocs.io/en/latest/py_api/GradientBoostedTreesLearner/
    learner = ydf.GradientBoostedTreesLearner(
        label="y",
        num_trees=5,
        max_depth=6,
        apply_link_function=False,
        **extra_kwargs,
    )
    model = learner.train({"x": dataset.X, "y": dataset.y})
    return model


@pytest.fixture
def hls_convert(build_ydf_model, tmp_path):
    ydf_model = build_ydf_model

    # Create a conifer config
    cfg = conifer.backends.xilinxhls.auto_config()
    cfg["Precision"] = "ap_fixed<32,16,AP_RND,AP_SAT>"
    # Set the output directory to something unique
    cfg["OutputDir"] = str(tmp_path)
    cfg["XilinxPart"] = "xcu250-figd2104-2L-e"

    # Create and compile the model
    model = conifer.converters.convert_from_ydf(ydf_model, cfg)
    model.compile()
    return model


@pytest.fixture
def vhdl_convert(build_ydf_model, tmp_path):
    ydf_model = build_ydf_model

    # Create a conifer config
    cfg = conifer.backends.vhdl.auto_config()
    cfg["Precision"] = "ap_fixed<32,16>"
    # Set the output directory to something unique
    cfg["OutputDir"] = str(tmp_path)
    cfg["XilinxPart"] = "xcu250-figd2104-2L-e"

    # Create and compile the model
    model = conifer.converters.convert_from_ydf(ydf_model, cfg)
    model.compile()
    return model


def test_build_vhdl(vhdl_convert):
    # TODO: Check equality of predictions.
    _ = vhdl_convert
    assert True


def test_predict_hls(hls_convert, build_ydf_model, build_dataset):
    dataset = build_dataset
    hls_model = hls_convert
    ydf_model = build_ydf_model

    hls_pred = hls_model.decision_function(dataset.X_test)
    tf_df_pred = np.squeeze(ydf_model.predict({"x": dataset.X_test}))

    assert np.all(np.isclose(hls_pred, tf_df_pred, rtol=1e-2, atol=1e-2))


def test_toy_model():
    dataset = {
        "x1": np.array([0, 0, 1, 1]),
        "x2": np.array([0, 1, 0, 1]),
        "y": np.array([0, 0, 0, 1]),
    }
    model = ydf.GradientBoostedTreesLearner(
        label="y",
        num_trees=1,
        max_depth=4,
        apply_link_function=False,
        min_examples=1,
    ).train(dataset)

    assert (
        model.get_tree(0).pretty(model.data_spec())
        == """\
'x2' >= 0.5 [score=0.0625 missing=True]
    ├─(pos)─ 'x1' >= 0.5 [score=0.25 missing=True]
    │        ├─(pos)─ value=0.4 sd=0
    │        └─(neg)─ value=-0.13333 sd=0
    └─(neg)─ value=-0.13333 sd=0
"""
    )
    assert model.initial_predictions() == [-1.0986123085021973]

    cfg = conifer.backends.xilinxhls.auto_config()
    conifer_model = conifer.converters.ydf.convert(model)

    assert conifer_model == {
        "max_depth": 2,
        "n_trees": 1,
        "n_classes": 2,
        "n_features": 2,
        "init_predict": [-1.0986123085021973],
        "norm": 1,
        "trees": [
            [
                {
                    "feature": [1, -2, 0, -2, -2],
                    "threshold": [0.5, -2.0, 0.5, -2.0, -2.0],
                    "children_left": [1, -1, 3, -1, -1],
                    "children_right": [2, -1, 4, -1, -1],
                    "value": [
                        0,
                        -0.13333334028720856,
                        0,
                        -0.13333334028720856,
                        0.4000000059604645,
                    ],
                }
            ]
        ],
    }
