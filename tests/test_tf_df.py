import pytest
import numpy as np
from typing import Tuple
import tensorflow_decision_forests as tfdf


@pytest.fixture(params=["hastie", "iris"])
def build_dataset(request) -> Tuple[np.array, np.array, np.array]:

    if request.param == "hastie":
        from sklearn.datasets import make_hastie_10_2
        X, y = make_hastie_10_2(random_state=0, n_samples=4000)
        return X[:2000], y[:2000], X[2000:]

    elif request.param == "iris":

        from sklearn.datasets import load_iris
        iris = load_iris()
        return iris.data, iris.target, iris.data

    else:
        raise ValueError("Unknown dataset")


@pytest.fixture
def build_tf_df_model(build_dataset) -> tfdf.keras.CoreModel:

    X, y, _ = build_dataset

    tf_df_model = tfdf.keras.GradientBoostedTreesModel(
        num_trees=20, max_depth=6, verbose=0, apply_link_function=False)
    tf_df_model.fit(x=X, y=y)

    return tf_df_model


@pytest.fixture
def hls_convert(build_tf_df_model, tmp_path):
    import conifer

    tf_df_model = build_tf_df_model

    # Create a conifer config
    cfg = conifer.backends.xilinxhls.auto_config()
    cfg["Precision"] = "ap_fixed<32,16,AP_RND,AP_SAT>"
    # Set the output directory to something unique
    cfg["OutputDir"] = str(tmp_path)
    cfg["XilinxPart"] = "xcu250-figd2104-2L-e"

    # Create and compile the model
    model = conifer.converters.convert_from_tf_df(tf_df_model, cfg)
    model.compile()
    return model


@pytest.fixture
def vhdl_convert(build_tf_df_model, tmp_path):
    import conifer

    tf_df_model = build_tf_df_model

    # Create a conifer config
    cfg = conifer.backends.vhdl.auto_config()
    cfg["Precision"] = "ap_fixed<32,16>"
    # Set the output directory to something unique
    cfg["OutputDir"] = str(tmp_path)
    cfg["XilinxPart"] = "xcu250-figd2104-2L-e"

    # Create and compile the model
    model = conifer.converters.convert_from_tf_df(tf_df_model, cfg)
    model.compile()
    return model


def test_toy_model(tmp_path):
    import conifer

    tf_df_path = str(tmp_path / "model")
    builder = tfdf.builder.GradientBoostedTreeBuilder(
        path=tf_df_path,
        bias=1.0,
        model_format=tfdf.builder.ModelFormat.YGGDRASIL_DECISION_FOREST,
        objective=tfdf.py_tree.objective.ClassificationObjective(
            label="color", classes=["red", "blue"]))

    #  f0>=1.5
    #    ├─(pos)─ f1>=2.5
    #    │         ├─(pos)─ value: 0.8
    #    │         └─(neg)─ value: 0.6
    #    └─(neg)─ value: 0.1

    Tree = tfdf.py_tree.tree.Tree
    NonLeafNode = tfdf.py_tree.node.NonLeafNode
    NumericalHigherThanCondition = tfdf.py_tree.condition.NumericalHigherThanCondition
    SimpleColumnSpec = tfdf.py_tree.dataspec.SimpleColumnSpec
    LeafNode = tfdf.py_tree.node.LeafNode
    RegressionValue = tfdf.py_tree.value.RegressionValue
    ColumnType = tfdf.py_tree.dataspec.ColumnType

    builder.add_tree(
        Tree(NonLeafNode(
            condition=NumericalHigherThanCondition(
                feature=SimpleColumnSpec(name="f0", type=ColumnType.NUMERICAL),
                threshold=1.5,
                missing_evaluation=False),
            pos_child=NonLeafNode(
                condition=NumericalHigherThanCondition(
                    feature=SimpleColumnSpec(
                        name="f1", type=ColumnType.NUMERICAL),
                    threshold=2.5,
                    missing_evaluation=False),
                pos_child=LeafNode(value=RegressionValue(value=0.8)),
                neg_child=LeafNode(value=RegressionValue(value=0.6))),
            neg_child=LeafNode(value=RegressionValue(value=0.1)))))
    builder.close()

    inspector = tfdf.inspector.make_inspector(tf_df_path)

    cfg = conifer.backends.xilinxhls.auto_config()
    conifer_model = conifer.converters.tf_df.convert(inspector)

    assert conifer_model == {"max_depth": 2,
                             "n_trees": 1,
                             "n_classes": 2,
                             "n_features": 2,
                             "init_predict": [1.0],
                             "norm": 1,
                             "trees": [[{"feature": [0, -2, 1, -2, -2],
                                         "threshold": pytest.approx([1.5, -2.0, 2.5, -2.0, -2.0]),
                                         "children_left": [1, -1, 3, -1, -1],
                                         "children_right": [2, -1, 4, -1, -1],
                                         "value": pytest.approx([0, 0.1, 0, 0.6, 0.8])}
                                        ]]
                             }


def test_build_vhdl(vhdl_convert):
    # TODO: Check equality of predictions.
    _ = vhdl_convert
    assert True


def test_predict_hls(hls_convert, build_tf_df_model, build_dataset):

    import numpy as np

    _, _, X = build_dataset
    hls_model = hls_convert
    tf_df_model = build_tf_df_model

    hls_pred = hls_model.decision_function(X)
    tf_df_pred = np.squeeze(tf_df_model.predict(X))

    assert np.all(np.isclose(hls_pred, tf_df_pred, rtol=1e-2, atol=1e-2))
