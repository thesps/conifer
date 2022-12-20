"""Convertion from TensorFlow Decision Forests format."""

from typing import List, Any, Dict
import tensorflow_decision_forests as tfdf

FeatureMapping = Dict[int, int]


def _convert_gbt(inspector: tfdf.inspector.AbstractInspector):
    """Specialization of "convert" for GBT models."""

    if inspector.task != tfdf.inspector.Task.CLASSIFICATION:
        raise ValueError("Only classification TF-DF models are supported")

    src_trees = inspector.extract_all_trees()

    # Mapping between the (sparse) column index and a (dense) feature index of the features.
    column_idx_to_feature_idx = _feature_mapping(inspector)

    # Find the maximum depth of the trees
    max_depth = _get_max_depth(src_trees)

    # Conifer dictionary without the "trees" field.

    bias = inspector.bias
    if isinstance(bias, float):
        bias = [bias]
    else:
        bias = bias[:]

    base_conifer_dict = {"max_depth": max_depth,
                         # Total number of iterations
                         "n_trees": inspector.num_trees() // inspector.num_trees_per_iter,
                         # Number of label classes
                         "n_classes": inspector.objective().num_classes,
                         # Total number of input features
                         "n_features": len(inspector. features()),
                         # Predictions of the model without any trees.
                         "init_predict": bias,
                         "norm": 1,
                         }

    # Converts trees
    dst_trees = []
    num_iterations = inspector.num_trees() // inspector.num_trees_per_iter

    for iter_idx in range(num_iterations):
        dst_trees_per_iter = []
        for sub_tree_idx in range(inspector.num_trees_per_iter):

            # Converts trees from TF-DF to Conifer format
            src_tree_idx = sub_tree_idx + iter_idx * inspector.num_trees_per_iter
            dst_tree = ConiferTree()
            dst_tree.set_tree(src_trees[src_tree_idx],
                              column_idx_to_feature_idx)

            dst_trees_per_iter.append(dst_tree.to_dict())

        dst_trees.append(dst_trees_per_iter)

    return {**base_conifer_dict, "trees": dst_trees}


def _get_max_depth(trees: List[tfdf.py_tree.tree.Tree]) -> int:
    """Gets the maximum depth of a list of trees."""

    def node_max_depth(node: tfdf.py_tree.node.AbstractNode, depth: int) -> int:
        if isinstance(node, tfdf.py_tree.node.NonLeafNode):
            return max(
                node_max_depth(node.neg_child, depth+1),
                node_max_depth(node.pos_child, depth+1)
            )

        return depth

    return max(node_max_depth(tree.root, depth=0) for tree in trees)


def _feature_mapping(inspector: tfdf.inspector.AbstractInspector) -> FeatureMapping:
    """Computes a dense indexing of the TF-DF model "column index".

    TF-DF column index is a possibly sparse indexing of the input features of the model.
    This method maps the column indices of the input features to [0, m) where m is the
    number of features.
    """

    mapping: FeatureMapping = {}
    for feature in inspector.features():
        new_index = len(mapping)
        mapping[feature.col_idx] = new_index
    return mapping


class ConiferTree:

    def __init__(self):
        self.feature: List[int] = []
        self.threshold: List[float] = []
        self.children_left: List[int] = []
        self.children_right: List[int] = []
        self.value: List[float] = []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature": self.feature,
            "threshold": self.threshold,
            "children_left": self.children_left,
            "children_right": self.children_right,
            "value": self.value,
        }

    def num_nodes(self) -> int:
        """Number of nodes in the tree."""

        n = len(self.feature)
        assert n == len(self.threshold)
        assert n == len(self.children_left)
        assert n == len(self.children_right)
        assert n == len(self.value)
        return n

    def set_tree(self, tree: tfdf.py_tree.tree.Tree, column_idx_to_feature_idx: FeatureMapping) -> None:
        self._add_node(tree.root, column_idx_to_feature_idx)

    def _add_node(self, node: tfdf.py_tree.node.AbstractNode, column_idx_to_feature_idx: FeatureMapping) -> None:

        if isinstance(node, tfdf.py_tree.node.LeafNode):

            # A leaf node
            self.feature.append(-2)
            self.threshold.append(-2.0)
            self.children_left.append(-1)
            self.children_right.append(-1)

            if not isinstance(node.value, tfdf.py_tree.value.RegressionValue):
                raise ValueError(f"No supported leaf type: {node.value}")

            self.value.append(node.value.value)

        elif isinstance(node, tfdf.py_tree.node.NonLeafNode):

            # A non leaf node
            node_idx = self.num_nodes()

            # Set condition
            if isinstance(node.condition, tfdf.py_tree.condition.NumericalHigherThanCondition):
                feature_idx = column_idx_to_feature_idx.get(
                    node.condition.feature.col_idx)
                if feature_idx is None:
                    raise RuntimeError(
                        f"Unknown feature {node.condition.feature}")
                self.feature.append(feature_idx)
                self.threshold.append(node.condition.threshold)
            else:
                raise ValueError(
                    f"No supported TF-DF condition: {node.condition}")

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

        else:
            raise ValueError("No supported node type")


def convert(model: Any) -> Any:

    if isinstance(model, tfdf.inspector.AbstractInspector):
        # Skip "make_inspector" if the model is already an inspector
        inspector = model
    else:
        inspector = model.make_inspector()

    if inspector.model_type() == "GRADIENT_BOOSTED_TREES":
        return _convert_gbt(inspector)

    raise ValueError(
        f"Not supported TF-DF model type {inspector.model_type()}")
