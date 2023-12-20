"""Test conifer import of ydf models."""

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
def toy_dataset(request) -> Dataset:
    """Buids a toy dataset."""

    if request.param == "hastie":
        X, y = make_hastie_10_2(random_state=0, n_samples=4000)
        return Dataset(X[:2000], y[:2000] == 1, X[2000:])

    elif request.param == "iris":
        iris = load_iris()
        return Dataset(iris.data, iris.target, iris.data)

    else:
        raise ValueError("Unknown dataset")


@pytest.fixture(params=["exact"])
def toy_ydf_model(toy_dataset, request) -> ydf.GenericModel:
    """Trains a YDF model on a toy dataset."""

    if request.param == "exact":
        extra_kwargs = {}
    elif request.param == "oblique":
        extra_kwargs = {"split_axis": "SPARSE_OBLIQUE"}  # TODO: Test when supported.
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
    model = learner.train({"x": toy_dataset.X, "y": toy_dataset.y})
    return model


def test_vhdl_toy_model(toy_dataset, toy_ydf_model, tmp_path):
    # Create a conifer config
    cfg = conifer.backends.vhdl.auto_config()
    cfg["Precision"] = "ap_fixed<32,16>"
    cfg["OutputDir"] = str(tmp_path)
    cfg["XilinxPart"] = "xcu250-figd2104-2L-e"

    # Create and compile the model
    conifer_model = conifer.converters.convert_from_ydf(toy_ydf_model, cfg)
    conifer_model.compile()

    # Check predictions
    # TODO: Check equality of predictions.
    # conifer_pred = conifer_model.decision_function(toy_dataset.X_test)
    # ydf_pred = np.squeeze(toy_ydf_model.predict({"x": toy_dataset.X_test}))
    # assert np.all(np.isclose(conifer_pred, ydf_pred, rtol=1e-2, atol=1e-2))


def test_hls_toy_model(toy_dataset, toy_ydf_model, tmp_path):
    # Create a conifer config
    cfg = conifer.backends.xilinxhls.auto_config()
    cfg["Precision"] = "ap_fixed<32,16,AP_RND,AP_SAT>"
    cfg["OutputDir"] = str(tmp_path)
    cfg["XilinxPart"] = "xcu250-figd2104-2L-e"

    # Create and compile the model
    conifer_model = conifer.converters.convert_from_ydf(toy_ydf_model, cfg)
    conifer_model.compile()

    # Check predictions
    conifer_pred = conifer_model.decision_function(toy_dataset.X_test)
    ydf_pred = np.squeeze(toy_ydf_model.predict({"x": toy_dataset.X_test}))
    np.testing.assert_allclose(conifer_pred, ydf_pred, atol=1e-3, rtol=1e-3)


def test_cpp_toy_model(toy_dataset, toy_ydf_model, tmp_path):
    # Create a conifer config
    cfg = conifer.backends.cpp.auto_config()
    cfg["Precision"] = "double"
    cfg["OutputDir"] = str(tmp_path)

    # Create and compile the model
    conifer_model = conifer.converters.convert_from_ydf(toy_ydf_model, cfg)
    conifer_model.compile()

    # Check predictions
    conifer_pred = np.squeeze(conifer_model.decision_function(toy_dataset.X_test))
    ydf_pred = np.squeeze(toy_ydf_model.predict({"x": toy_dataset.X_test}))
    np.testing.assert_allclose(conifer_pred, ydf_pred, atol=1e-3, rtol=1e-3)


def test_four_nodes_model():
    # Train a model with 4 nodes
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

    # Check the model structure.
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

    # Injest the model in Conifer.
    conifer_model = conifer.converters.ydf.convert(model)

    # Check the Conifer model.
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
