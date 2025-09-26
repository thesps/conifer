"""Test conifer import of ydf models."""

import dataclasses

import conifer
import pytest
import numpy as np
import ydf
from sklearn.datasets import load_iris, load_diabetes
from sklearn.datasets import make_hastie_10_2
from scipy.special import expit


@dataclasses.dataclass
class Dataset:
    X: np.ndarray
    y: np.ndarray
    X_test: np.ndarray
    kind: str


@pytest.fixture(params=["hastie", "iris", "diabetes"])
def toy_dataset(request) -> Dataset:
    """Buids a toy dataset."""

    if request.param == "hastie":
        X, y = make_hastie_10_2(random_state=0, n_samples=4000)
        return Dataset(X[:2000], y[:2000] == 1, X[2000:], "classification")

    elif request.param == "iris":
        iris = load_iris()
        return Dataset(iris.data, iris.target, iris.data, "classification")
    
    elif request.param == "diabetes":
        diabetes = load_diabetes()
        return Dataset(diabetes.data, diabetes.target, diabetes.data, "regression")
    
    else:
        raise ValueError("Unknown dataset")


@pytest.fixture(params=[("exact", "GradientBoostedTreesLearner", ydf.Task.CLASSIFICATION),
                        ("oblique", "GradientBoostedTreesLearner", ydf.Task.CLASSIFICATION),
                        ("exact", "GradientBoostedTreesLearner", ydf.Task.REGRESSION),
                        ("oblique", "GradientBoostedTreesLearner", ydf.Task.REGRESSION),
                        ("exact", "IsolationForestLearner", ydf.Task.ANOMALY_DETECTION)])
def toy_ydf_model(toy_dataset, request) -> ydf.GenericModel:
    """Trains a YDF model on a toy dataset."""

    # skip the combinations of regression dataset with [classification|anomaly detection] model
    if toy_dataset.kind == "regression" and request.param[2] == ydf.Task.CLASSIFICATION:
        pytest.skip("Skipping classification task on regression dataset")
    # skip classification dataset with regression model
    if toy_dataset.kind == "classification" and request.param[2] == ydf.Task.REGRESSION:
        pytest.skip("Skipping regression task on classification dataset")
    if toy_dataset.kind == "regression" and request.param[2] == ydf.Task.ANOMALY_DETECTION:
        pytest.skip("Skipping anomaly detection task on regression dataset")

    if request.param[0] == "exact":
        extra_kwargs = {}
    elif request.param[0] == "oblique":
        extra_kwargs = {"split_axis": "SPARSE_OBLIQUE","sparse_oblique_weights": "CONTINUOUS"}
    else:
        assert False

    # See https://ydf.readthedocs.io/en/latest/py_api/GradientBoostedTreesLearner/
    if request.param[1] == "GradientBoostedTreesLearner":
        Learner = ydf.GradientBoostedTreesLearner
        learner_kwargs = {"apply_link_function" : False}
    elif request.param[1] == "IsolationForestLearner":
        Learner = ydf.IsolationForestLearner
        learner_kwargs = {}
    else:
        assert False

    learner = Learner(
        label="y",
        num_trees=5,
        max_depth=6,
        task=request.param[2],
        **learner_kwargs,
        **extra_kwargs,
    )
    model = learner.train({"x": toy_dataset.X, "y": toy_dataset.y})
    return model

def test_vhdl_toy_model(toy_dataset, toy_ydf_model, tmp_path):
    # Skip oblique models, evaluate ydfs node condition as no easy way to access hyperparameters
    if isinstance(toy_ydf_model.get_tree(0).root.condition, ydf.tree.NumericalSparseObliqueCondition):
        pytest.skip("Skipping VHDL backend for oblique splits")
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
    cfg["Precision"] = "ap_fixed<32,16>"
    cfg["OutputDir"] = str(tmp_path)
    cfg["XilinxPart"] = "xcu250-figd2104-2L-e"

    # Create and compile the model
    conifer_model = conifer.converters.convert_from_ydf(toy_ydf_model, cfg)
    conifer_model.compile()

    # Check predictions
    conifer_pred = conifer_model.decision_function(toy_dataset.X_test)
    if isinstance(toy_ydf_model, ydf.IsolationForestModel):
        conifer_pred = 2**conifer_pred
    ydf_pred = np.squeeze(toy_ydf_model.predict({"x": toy_dataset.X_test}))
    # Oblique models need expit (applying link function doesn't do anything) but only for single class classification tasks
    if (conifer_model.is_oblique()) and len(ydf_pred.shape) == 1 and toy_dataset.kind != "regression":
        conifer_pred = expit(conifer_pred)
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
    if isinstance(toy_ydf_model, ydf.IsolationForestModel):
        conifer_pred = 2**conifer_pred
    ydf_pred = np.squeeze(toy_ydf_model.predict({"x": toy_dataset.X_test}))
    # Oblique models need expit (applying link function doesn't do anything) but only for single class classification tasks
    if (conifer_model.is_oblique()) and len(ydf_pred.shape) == 1 and toy_dataset.kind != "regression":
        conifer_pred = expit(conifer_pred)
    np.testing.assert_allclose(conifer_pred, ydf_pred, atol=1e-3, rtol=1e-3)


def test_py_toy_model(toy_dataset, toy_ydf_model, tmp_path):

    # Create the model
    conifer_model = conifer.converters.convert_from_ydf(toy_ydf_model)

    # Check predictions
    conifer_pred = np.squeeze(conifer_model.decision_function(toy_dataset.X_test))
    if isinstance(toy_ydf_model, ydf.IsolationForestModel):
        conifer_pred = 2**conifer_pred
    ydf_pred = np.squeeze(toy_ydf_model.predict({"x": toy_dataset.X_test}))
    # Oblique models need expit (applying link function doesn't do anything) but only for single class classification tasks
    if (isinstance(toy_ydf_model.get_tree(0).root.condition, ydf.tree.NumericalSparseObliqueCondition) 
        and len(ydf_pred.shape) == 1) and toy_dataset.kind != "regression":
        conifer_pred = expit(conifer_pred)
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
    conifer_model_dict = conifer.converters.ydf.convert(model)

    # Check the Conifer model.
    assert conifer_model_dict == {
        "max_depth": 2,
        "n_trees": 1,
        "n_classes": 2,
        "n_features": 2,
        "init_predict": [-1.0986123085021973],
        "norm": 1,
        "library": "ydf",
        "splitting_convention": conifer.converters.splitting_conventions["ydf"],
        "trees": [
            [
                {
                    "feature": [1, -2, 0, -2, -2],
                    "weight": [[0, 1], [0, 0], [1, 0], [0, 0], [0, 0]],
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

    # injest the model again into conifer for inference
    conifer_model = conifer.converters.convert_from_ydf(model)

    # create a test dataset including some examples on the threshold to check the splitting convention
    test_dataset = {
        "x1": np.array([0, 0, 1, 1, 0.5, 0, 0.5, 0.5, 1]),
        "x2": np.array([0, 1, 0, 1, 0.5, 0.5, 0, 1, 0.5]),
    }

    # perform inference with ydf and conifer
    ydf_pred = np.squeeze(model.predict(test_dataset))
    X = np.transpose(np.array([test_dataset["x1"], test_dataset["x2"]]))
    conifer_pred = np.squeeze(conifer_model.decision_function(X))

    # assert numerical closeness
    np.testing.assert_allclose(conifer_pred, ydf_pred, atol=1e-5, rtol=1e-5)