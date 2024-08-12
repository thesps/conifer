"""Convertion from Yggdrasil Decision Forests format."""

from typing import List, Any, Dict, Union
import dataclasses

import ydf

FeatureMapping = Dict[int, int]
ConiferModel = Dict[str, Any]


def convert(model: ydf.GenericModel) -> ConiferModel:
    """Converts a YDF model into a Conifer model."""

    if model.task() != ydf.Task.CLASSIFICATION:
        raise ValueError(
            f"Classification YDF model expected. Instead, found task {model.task()!r}"
        )
    if isinstance(model, ydf.GradientBoostedTreesModel):
        return _convert_gbt(model)
    raise ValueError(f"Not supported YDF model type {type(model)}")


def _convert_gbt(model: ydf.GradientBoostedTreesModel) -> ConiferModel:
    """Converts a YDF GBT model into a Conifer model."""

    src_trees = model.get_all_trees()
    column_idx_to_feature_idx = _feature_mapping(model)
    max_depth = _get_max_depth(src_trees)
    initial_predictions = model.initial_predictions().tolist()
    label_classes = model.label_classes()
    num_trees_per_iter = len(initial_predictions)
    num_iterations = model.num_trees() // num_trees_per_iter

    # Conifer dictionary without the "trees" field.
    base_conifer_dict = {
        "max_depth": max_depth,
        "n_trees": num_iterations,
        "n_classes": len(label_classes),
        "n_features": len(column_idx_to_feature_idx),
        "init_predict": initial_predictions,
        "norm": 1,
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
    feature: List[int] = dataclasses.field(default_factory=list)
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
        self._add_node(tree.root, column_idx_to_feature_idx)

    def _add_node(
        self,
        node: ydf.tree.AbstractNode,
        column_idx_to_feature_idx: FeatureMapping,
    ) -> None:
        if node.is_leaf:
            # A leaf node
            assert isinstance(node, ydf.tree.Leaf)

            if not isinstance(node.value, ydf.tree.RegressionValue):
                raise ValueError(f"No supported leaf value {node.value!r}")
            self.feature.append(-2)
            self.threshold.append(-2.0)
            self.children_left.append(-1)
            self.children_right.append(-1)
            self.value.append(node.value.value)

        else:
            # A non leaf node
            assert isinstance(node, ydf.tree.NonLeaf)
            node_idx = self.num_nodes()

            # Set condition
            if isinstance(node.condition, ydf.tree.NumericalHigherThanCondition):
                feature_idx = column_idx_to_feature_idx.get(node.condition.attribute)
                if feature_idx is None:
                    raise RuntimeError(f"Unknown feature {node.condition.attribute}")
                self.feature.append(feature_idx)
                self.threshold.append(node.condition.threshold)
            elif isinstance(node.condition, ydf.tree.NumericalSparseObliqueCondition):
                raise ValueError("Oblique conditions are not yet supported")
            else:
                raise ValueError(f"No supported YDF condition: {node.condition}")

            # Placeholder until the children node indices are computed
            self.children_left.append(-3)
            self.children_right.append(-3)

            # Ignored value
            self.value.append(0)

            # Set children nodes
            self.children_left[node_idx] = self.num_nodes()
            self._add_node(node.neg_child, column_idx_to_feature_idx)

            self.children_right[node_idx] = self.num_nodes()
            self._add_node(node.pos_child, column_idx_to_feature_idx)
