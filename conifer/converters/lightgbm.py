import tempfile
import lightgbm as lgb


def parse_model(model_file):
    model_dict = {}
    with open(model_file, 'r') as f:
        lines = f.readlines()
    for line in lines:
        if line.startswith('Tree='):
            tree_id = int(line.split('=')[1])
            model_dict[tree_id] = {}
        elif line.startswith('split_feature'):
            model_dict[tree_id]['split_feature'] = list(map(int, line.split('=')[1].strip().split()))
        elif line.startswith('threshold'):
            model_dict[tree_id]['threshold'] = list(map(float, line.split('=')[1].strip().split()))   
        elif line.startswith('left_child'):
            model_dict[tree_id]['left_child'] = list(map(int, line.split('=')[1].strip().split()))
        elif line.startswith('right_child'):
            model_dict[tree_id]['right_child'] = list(map(int, line.split('=')[1].strip().split()))
        elif line.startswith('leaf_value'):
            model_dict[tree_id]['leaf_value'] = list(map(float, line.split('=')[1].strip().split()))
    return model_dict


def convert_tree(lgb_tree_dict):
    I = len(lgb_tree_dict['split_feature'])
    assert len(lgb_tree_dict['threshold']) == I
    assert len(lgb_tree_dict['left_child']) == I
    assert len(lgb_tree_dict['right_child']) == I
    assert len(lgb_tree_dict['leaf_value']) == I + 1

    skl_tree_dict = {
        "feature": [-2] * (2 * I + 1),
        "threshold": [-2.0] * (2 * I + 1),
        "children_left": [-1] * (2 * I + 1),
        "children_right": [-1] * (2 * I + 1),
        "value": [0.0] * (2 * I + 1),
    }
    skl_tree_dict['feature'][:I] = lgb_tree_dict['split_feature']
    skl_tree_dict['threshold'][:I] = lgb_tree_dict['threshold']
    skl_tree_dict['children_left'][:I] = lgb_tree_dict['left_child']
    skl_tree_dict['children_right'][:I] = lgb_tree_dict['right_child']
    skl_tree_dict['value'][I:] = lgb_tree_dict['leaf_value']
    for i in range(I):
        if skl_tree_dict['children_left'][i] < 0:
            skl_tree_dict['children_left'][i] = I - 1 - skl_tree_dict['children_left'][i]
        if skl_tree_dict['children_right'][i] < 0:
            skl_tree_dict['children_right'][i] = I - 1 - skl_tree_dict['children_right'][i]
    return skl_tree_dict


def convert(model: lgb.Booster):
    # only support regression
    # TODO: use model.dump_model() instead
    # TODO: support classification
    with tempfile.NamedTemporaryFile() as f:
        model.save_model(f.name)
        lgb_trees_dict = parse_model(f.name)
    trees_dict = {}
    for tree_id, tree_dict in lgb_trees_dict.items():
        trees_dict[tree_id] = convert_tree(tree_dict)

    ensemble_dict = {
        "max_depth": model.params["max_depth"],
        "n_trees": model.num_trees(),
        "n_features": model.num_feature(),
        "n_classes": model.params["num_class"],
        "trees": [[v] for v in trees_dict.values()],  # only support regression
        "init_predict": [0] * model.params["num_class"],
        "norm": 1,
    }
    return ensemble_dict