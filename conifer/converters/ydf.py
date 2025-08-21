"""Convertion from Yggdrasil Decision Forests format."""

from typing import List, Any, Dict, Union
import dataclasses
from conifer.converters import splitting_conventions
import math

import ydf

FeatureMapping = Dict[int, int]
ConiferModel = Dict[str, Any]


def convert(model: ydf.GenericModel) -> ConiferModel:
    """Converts a YDF model into a Conifer model."""

    if model.task() not in [ydf.Task.CLASSIFICATION, ydf.Task.REGRESSION, ydf.Task.ANOMALY_DETECTION]:
        raise ValueError(
            f"Classification, Regression, or Anomaly Detection YDF model expected. Instead, found task {model.task()!r}"
        )
    if isinstance(model, (ydf.GradientBoostedTreesModel, ydf.IsolationForestModel)):
        return _convert_forest(model)
    raise ValueError(f"Not supported YDF model type {type(model)}")


def _convert_forest(model: Union[ydf.GradientBoostedTreesModel, ydf.IsolationForestModel]) -> ConiferModel:
    """Converts a YDF model into a Conifer model."""

    is_gbtm = isinstance(model, ydf.GradientBoostedTreesModel)
    is_ifm = isinstance(model, ydf.IsolationForestModel)
    src_trees = model.get_all_trees()
    column_idx_to_feature_idx = _feature_mapping(model)
    max_depth = _get_max_depth(src_trees)

    if is_gbtm:
        initial_predictions = model.initial_predictions().tolist()
        num_trees_per_iter = len(initial_predictions)
        num_iterations = model.num_trees() // num_trees_per_iter
        norm = 1.
        if model.task() == ydf.Task.CLASSIFICATION:
            n_classes = len(model.label_classes())
        elif model.task() == ydf.Task.REGRESSION:
            n_classes = 1
        else:
            assert False
    elif is_ifm:
        n_classes = 1
        initial_predictions = [0]
        num_trees_per_iter = 1
        num_iterations = model.num_trees()
        num_examples_per_trees = model.get_tree(tree_idx=0).root.value.num_examples_without_weight
        norm = - 1. / (num_iterations * preiss_average_path_length(num_examples_per_trees))
    else:
        assert False

    # Conifer dictionary without the "trees" field.
    base_conifer_dict = {
        "max_depth": max_depth,
        "n_trees": num_iterations,
        "n_classes": n_classes,
        "n_features": len(column_idx_to_feature_idx),
        "init_predict": initial_predictions,
        "norm": norm,
        "library": "ydf",
        "splitting_convention": splitting_conventions["ydf"],
    }

    # Converts trees
    dst_trees = []
    for iter_idx in range(num_iterations):
        dst_trees_per_iter = []
        for sub_tree_idx in range(num_trees_per_iter):
            src_tree_idx = sub_tree_idx + iter_idx * num_trees_per_iter
            dst_tree = ConiferTreeBuilder()
            dst_tree.set_tree(src_trees[src_tree_idx], column_idx_to_feature_idx)
            dst_trees_per_iter.append(dst_tree.to_dict())
        dst_trees.append(dst_trees_per_iter)
    return {**base_conifer_dict, "trees": dst_trees}


def _get_max_depth(trees: List[ydf.tree.Tree]) -> int:
    """Gets the maximum depth of a list of trees."""

    def node_max_depth(node: ydf.tree.AbstractNode, depth: int) -> int:
        if node.is_leaf:
            return depth
        else:
            return max(
                node_max_depth(node.neg_child, depth + 1),
                node_max_depth(node.pos_child, depth + 1),
            )

    return max(node_max_depth(tree.root, depth=0) for tree in trees)


def _feature_mapping(model: ydf.GenericModel) -> FeatureMapping:
    """Computes a dense indexing of the YDF model "column index".

    YDF column index is a possibly sparse indexing of the input features of the model.
    This method maps the column indices of the input features to [0, m) where m is the
    number of features.
    """

    mapping = {}
    for input_feature in model.input_features():
        new_index = len(mapping)
        mapping[input_feature.column_idx] = new_index
    return mapping


@dataclasses.dataclass
class ConiferTreeBuilder:
    feature: List[List[int]] = dataclasses.field(default_factory=list)
    weight: List[List[float]] = dataclasses.field(default_factory=list)
    threshold: List[float] = dataclasses.field(default_factory=list)
    children_left: List[int] = dataclasses.field(default_factory=list)
    children_right: List[int] = dataclasses.field(default_factory=list)
    value: List[float] = dataclasses.field(default_factory=list)

    def to_dict(self) -> Dict[str, List[Union[float, int]]]:
        return dataclasses.asdict(self)

    def num_nodes(self) -> int:
        """Number of nodes in the tree."""

        n = len(self.feature)
        assert n == len(self.threshold)
        assert n == len(self.children_left)
        assert n == len(self.children_right)
        assert n == len(self.value)
        return n

    def set_tree(
        self, tree: ydf.tree.Tree, column_idx_to_feature_idx: FeatureMapping
    ) -> None:
        self._add_node(tree.root, column_idx_to_feature_idx, 0)

    def _add_node(
        self,
        node: ydf.tree.AbstractNode,
        column_idx_to_feature_idx: FeatureMapping,
        depth: int,
    ) -> None:
        if node.is_leaf:
            # A leaf node
            assert isinstance(node, ydf.tree.Leaf)

            self.feature.append(-2)
            self.weight.append( [0 for i in range(len(column_idx_to_feature_idx))])
            self.threshold.append(-2.0)
            self.children_left.append(-1)
            self.children_right.append(-1)
            value = 0
            if isinstance(node.value, ydf.tree.RegressionValue):
                value = node.value.value
            elif isinstance(node.value, ydf.tree.AnomalyDetectionValue):
                num_examples = node.value.num_examples_without_weight
                value = depth + preiss_average_path_length(num_examples)
            else:
                raise ValueError(f"No supported leaf value {node.value!r}")

            self.value.append(value)

        else:
            # A non leaf node
            assert isinstance(node, ydf.tree.NonLeaf)
            node_idx = self.num_nodes()

            # Set condition
            if isinstance(node.condition, ydf.tree.NumericalHigherThanCondition):
                feature_idx = column_idx_to_feature_idx.get(node.condition.attribute)
                if feature_idx is None:
                    raise RuntimeError(f"Unknown feature {node.condition.attribute}")
                weights = [0 for i in range(len(column_idx_to_feature_idx))]
                weights[feature_idx] = 1
                self.feature.append(feature_idx)
                self.weight.append(weights)
                self.threshold.append(node.condition.threshold)
            elif isinstance(node.condition, ydf.tree.NumericalSparseObliqueCondition):
                feature_idx = [column_idx_to_feature_idx.get(feature) for feature in node.condition.attributes]
                weights = [0 for i in range(len(column_idx_to_feature_idx))]
                for i,feature in enumerate(feature_idx):
                    weights[feature] = node.condition.weights[i]
                self.feature.append(0)
                self.weight.append(weights)
                self.threshold.append(node.condition.threshold)
                #raise ValueError("Oblique conditions are not yet supported")
            else:
                raise ValueError(f"No supported YDF condition: {node.condition}")

            # Placeholder until the children node indices are computed
            self.children_left.append(-3)
            self.children_right.append(-3)

            # Ignored value
            self.value.append(0)

            # Set children nodes
            self.children_left[node_idx] = self.num_nodes()
            self._add_node(node.neg_child, column_idx_to_feature_idx, depth+1)

            self.children_right[node_idx] = self.num_nodes()
            self._add_node(node.pos_child, column_idx_to_feature_idx, depth+1)

def preiss_average_path_length(num_examples: int) -> float:
    # Implementation from https://github.com/google/yggdrasil-decision-forests/blob/v1.10.0/yggdrasil_decision_forests/model/isolation_forest/isolation_forest.cc#L58
    assert num_examples > 0, "num_examples must be greater than 0"

    # Harmonic number approximation (from "Isolation Forest" by Liu et al.)
    def H(x: float) -> float:
        euler_constant = 0.5772156649
        return math.log(x) + euler_constant

    if num_examples > 2:
        return 2.0 * H(num_examples - 1.0) - 2.0 * (num_examples - 1.0) / num_examples
    elif num_examples == 2:
        return 1.0
    else:
        return 0.0  # To be safe
