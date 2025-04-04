from conifer import __version__ as version
from packaging.version import Version
import numpy as np
import os
import json
import copy
import datetime
import platform
import getpass
from typing import Union, Literal
try:
    import pydot
except ImportError:
    pydot = None
import logging
logger = logging.getLogger(__name__)

def _check_pydot():
    '''Returns True if PyDot and Graphviz are available, otherwise returns False'''
    if pydot is None:
        return False
    try:
        # Attempt to create an image of a blank graph
        # to check the pydot/graphviz installation.
        pydot.Dot.create(pydot.Dot())
        return True
    except OSError:
        return False
    
class DecisionTreeBase:
  '''
  Conifer DecisionTreeBase representation class
  '''
  _tree_fields = ['feature', 'threshold', 'value', 'children_left', 'children_right']
  def __init__(self, treeDict, splitting_convention):
    for key in DecisionTreeBase._tree_fields:
      val = treeDict.get(key, None)
      assert val is not None, f"Missing expected key {key} in treeDict"
      setattr(self, key, val)
    assert splitting_convention in ["<", "<="]
    self.splitting_convention = splitting_convention
    self.split_function = lambda arg1, arg2: arg1 < arg2 if splitting_convention == '<' else arg1 <= arg2

  def n_nodes(self):
    return len(self.feature)

  def n_leaves(self):
    return len([n for n in self.feature if n == -2])
  
  def max_depth(self):
    parents = [0] * self.n_nodes()
    for i in range(self.n_nodes()):
      j = self.children_left[i]
      if j != -1:
        parents[j] = i
      k = self.children_right[i]
      if k != -1:
        parents[k] = i
    parents[0] = -1
    depth = [0] * self.n_nodes()
    max_depth = 0
    for i in range(self.n_nodes()):
      depth = 0
      parent = parents[i]
      while parent != -1:
        depth += 1
        parent = parents[parent]
      max_depth = depth if depth > max_depth else max_depth
    return max_depth

  def sparsity(self):
      return 1 - (self.n_nodes() - self.n_leaves()) / (2 ** self.max_depth() - 1)

  def draw(self, filename : str = None, graph=None, tree_id=None):
    '''
    Draw a pydot graph of the decision tree

    Parameters
    ----------
    filename: string
        filename to save to, with any extension supported by pydot write

    graph:
        existing pydot graph to add to

    tree_id:
        ID of the tree within an ensemble

    Returns
    ----------
    pydot Dot graph object
    '''
    if not _check_pydot():
        raise ImportError('Could not import pydot. Install Graphviz and pydot to draw trees')
    graph = pydot.Dot(graph_type='graph') if graph is None else graph
    tree_id = '' if tree_id is None else tree_id
    sg = pydot.Cluster(tree_id, label=tree_id, peripheries=0 if tree_id=='' else 1)
    graph.add_subgraph(sg)
    for i in range(self.n_nodes()):
      node_id = f'{tree_id}_{i}'
      l = f'{tree_id}_{self.children_left[i]}'
      r = f'{tree_id}_{self.children_right[i]}'
      label = f'x[{self.feature[i]}] {self.splitting_convention} {self.threshold[i]:.2f}' if self.feature[i] != -2 else f'{self.value[i]:.2f}'
      sg.add_node(pydot.Node(node_id, label=label))
      if self.children_left[i] != -1:
        sg.add_edge(pydot.Edge(node_id, l,))
      if self.children_right[i] != -1:
        sg.add_edge(pydot.Edge(node_id, r,))
    if filename is not None:
        _, extension = os.path.splitext(filename)
        if not extension:
            extension = 'png'
        else:
            extension = extension[1:]
        graph.write(filename, format=extension)

    return graph

  def apply(self, X):
    assert len(X.shape) == 2, 'Expected 2D input'
    y = np.zeros(X.shape[0], dtype='int')
    for i, x in enumerate(X):
      n = 0
      while self.feature[n] != -2:
        comp=self.split_function(x[self.feature[n]],self.threshold[n])
        n = self.children_left[n] if comp else self.children_right[n]
      y[i] = n
    return y

  def decision_function(self, X):
    assert len(X.shape) == 2, 'Expected 2D input'
    yi = self.apply(X)
    return np.array(self.value)[yi]

class ConfigBase:
    '''
    Conifer Config representation class
    '''
    _config_fields = ['output_dir', 'project_name', 'backend']
    _alternates = {'output_dir'   : ['OutputDir'],
                   'project_name' : ['ProjectName'],
                   'backend'      : ['Backend']
                   }
    _defaults = {'output_dir'   : 'my-conifer-prj',
                 'project_name' : 'my_prj',
                 'backend'      : 'cpp'
                }
    _allow_undefined = []

    def __init__(self, configDict, validate=True):
        for key in self._config_fields:
            for k in [key, *self._alternates[key]]:
                val = configDict.get(k, None)
                if val is not None or key in self._allow_undefined:
                    setattr(self, key, val)
        if validate:
            self._validate()
        if getattr(self, 'output_dir', None) is not None:
            self.output_dir = os.path.abspath(self.output_dir)

    def _validate(self):
        vals = {}
        for key in self._config_fields:
            if key not in self._allow_undefined:
                vals[key] = getattr(self, key, None)
        assert not (None in vals.values()), f'Missing some required configuration, have: {vals}'


    def _to_dict(self):
        dictionary = {}
        for k in self._config_fields:
            if hasattr(getattr(self, k), '_to_dict'):
                v = getattr(self, k)._to_dict()
            else:
                v = getattr(self, k)
            dictionary[k] = v
        return dictionary

    def _log(self, logger):
        logger.info(f'Configuration: {self._to_dict()}')

    def default_config():
        return copy.deepcopy(ConfigBase._defaults)

class ModelBase:

    '''
    Conifer BDT representation class
    Primary interface to write, compile, execute, and synthesize conifer projects
    '''

    _ensemble_fields = ['n_classes', 'n_features', 'n_trees', 'max_depth', 'init_predict', 'norm', 'library', 'splitting_convention', 'feature_map']

    _optional_fields = []

    def __init__(self, ensembleDict, configDict=None, metadata=None):
        if 'feature_map' not in ensembleDict:
            ensembleDict['feature_map'] = {f"feat_{i}":i for i in range(ensembleDict['n_features'])}
        for key in ModelBase._ensemble_fields:
            val = ensembleDict.get(key, None)
            assert val is not None, f'Missing expected key {key} in ensembleDict'
            setattr(self, key, val)
        for key in ModelBase._optional_fields:
            val = ensembleDict.get(key, None)
            setattr(self, key, val)
        trees = ensembleDict.get('trees', None)
        assert trees is not None, f'Missing expected key trees in ensembleDict'
        self.trees = [[DecisionTreeBase(treeDict, self.splitting_convention) for treeDict in trees_class] for trees_class in trees]

        def _make_stamp():
            import datetime
            return int(datetime.datetime.now().timestamp())
        self._stamp = _make_stamp()

        if configDict is not None:
            self.config = ConfigBase(configDict)
        else:
            self.config = ConfigBase({'output_dir' : '.', 'project_name' : f'conifer_prj_{self._stamp}', 'backend' : 'python'})

        subset_keys = ['max_depth', 'n_trees', 'n_features', 'n_classes']
        subset_dict = {key: getattr(self, key) for key in subset_keys}
        logger.debug(f"Converted BDT with parameters {json.dumps(subset_dict)}")
        if metadata is None:
            self._metadata = [ModelMetaData()]
        else:
            if isinstance(metadata, list):
                self._metadata = metadata
                self._metadata.append(ModelMetaData())
            else:
                self._metadata = [metadata]

    def sparsity(self):
        s = sum([sum([1 - (tree.n_nodes() - tree.n_leaves()) / (2 ** self.max_depth - 1) for tree in tree_c]) for tree_c in self.trees])
        n = sum([len(tree_c) for tree_c in self.trees])
        return s / n

    def n_nodes(self):
        return sum([sum([tree.n_nodes() for tree in tree_c]) for tree_c in self.trees])

    def n_leaves(self):
        return sum([sum([tree.n_leaves() for tree in tree_c]) for tree_c in self.trees])

    def set_config(self, config):
        self.config = config

    def get_config(self):
        return self.config

    def save(self, filename=None):
        '''
        Save Model to file

        Parameters
        ----------
        filename: string
            filename to save to
        '''
        dictionary = {key : getattr(self, key) for key in ModelBase._ensemble_fields}
        dictionary['trees'] = [[{key : getattr(tree, key) for key in DecisionTreeBase._tree_fields} for tree in trees_i] for trees_i in self.trees]
        dictionary['config'] = self.config._to_dict()
        dictionary['metadata'] = [md._to_dict() for md in self._metadata]
        js = json.dumps(dictionary, indent=1)

        cfg = self.config
        if filename is None and cfg is not None:
            filename = f"{cfg.output_dir}/{cfg.project_name}.json"
            directory = cfg.output_dir
        elif filename is not None:
            directory = os.path.dirname(filename)
            directory = directory if directory != '' else './'
        else:
            logger.error('If model has no configuration, filename must be provided')
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(filename, 'w') as f:
            f.write(js)

    def write(self):
        '''
        Write the model files to the output directory specified in configuration
        '''
        raise NotImplementedError

    def compile(self):
        '''
        Compile conifer model ahead of running decision_function.
        Writes the project files first.
        Compilation is carried out by the model backend
        '''
        raise NotImplementedError
    
    def draw(self, filename=None):
        '''
        Draw a pydot graph of the decision tree

        Parameters
        ----------
        filename: string
            filename to save to, with any extension supported by pydot write

        Returns
        ----------
        pydot Dot graph object
        '''
        if not _check_pydot():
            raise ImportError('Could not import pydot. Install Graphviz and pydot to draw trees')
        graph = pydot.Dot(graph_type='graph')
        for i, treesi in enumerate(self.trees):
            for j, tree in enumerate(treesi):
                tree_id = f'Tree {i}, Class {j}'
                tree.draw(filename=None, graph=graph, tree_id=tree_id)
        if filename is not None:
            _, extension = os.path.splitext(filename)
            if not extension:
                extension = 'png'
            else:
                extension = extension[1:]
            graph.write(filename, format=extension)
        return graph

    def decision_function(self, X, trees=False):
        '''
        Compute the decision function of `X`.
        The backend performs the actual computation
        
        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
            Input sample
        
        Returns
        ----------    
        score: ndarray of shape (n_samples, n_classes) or (n_samples,)   

        '''
        assert len(X.shape) == 2, 'Expected 2D input'
        assert X.shape[1] == self.n_features, f'Wrong number of features, expected {self.n_features}, got {X.shape[1]}'

        n_classes = 1 if self.n_classes == 2 else self.n_classes
        n_samples = X.shape[0]
        y = np.zeros((self.n_trees, n_classes, n_samples))
        for it, trees in enumerate(self.trees):
            for ic, tree_c in enumerate(trees):
                y[it, ic] = tree_c.decision_function(X)
        y = (np.transpose(np.sum(y, axis=0)) + self.init_predict) * self.norm
        return np.squeeze(y)

    def build(self, **kwargs):
        '''
        Build the project, running the build function of the backend.

        Parameters
        ----------
        kwargs: keyword arguments of backend build method
        
        Returns
        ----------    
        success: bool
                 True if the build completed successfuly, otherwise False  
        '''
        raise NotImplementedError

    def profile(self, what : Literal["scores", "thresholds", "both"] = "both" , ax=None):
        """Profile the thresholds or scores of the trees in the ensemble for each feature.
        
        For each feature:
        - Orange line: median
        - Box plot: First-third quantile (25%,75%)
        - Gray caps and dashed line: 2.5% and 97.5% percentiles
        - Black caps and line: full range

        Args:
            what ("scores"|"thresholds|both"): choose between profiling the thresholds, the scores or both.
            ax (matplotlib.axes._axes.Axes, optional): Matplotlib axes. Defaults to None.

        Returns:
            matplotlib.axes._axes.Axes: Matplotlib axes.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise Exception("matplotlib not found. Please install matplotlib")

        plt.style.use({
                "font.size": 26,
                "xtick.labelsize": "small",
                "ytick.labelsize": "small",
                "grid.alpha": 0.8,
                "grid.linestyle": ":",
                "axes.linewidth": 2,
                "savefig.transparent": False,
        })
        if ax is None:
            if what == "both":
                _, ax = plt.subplots(1,2, figsize=(20, 10))
            else:
                _, ax = plt.subplots(figsize=(10, 10))
        
        if what == "both":
            assert len(ax) == 2, "Expected 2 axes for 'both' option"
            plt.subplots_adjust(wspace=.0)
            for axis, w in zip(ax,["scores", "thresholds"]):
                axis = self._profile(w, ax=axis)
                axis.set_ylim((-2, 2.1*self.n_features+2))
                if w == "thresholds":
                    axis.set_yticklabels([])
            return ax
        return self._profile(what, ax=ax)
            
            
    def _profile(self, what : Literal["scores", "thresholds"], ax=None):
        if self.feature_map is None:
            feature_names = [f"feat_{i}" for i in range(self.n_features)]
            feature_ids = list(range(self.n_features))
        else:
            feature_names, feature_ids = zip(*self.feature_map.items())
            feature_names = list(feature_names)
            feature_ids = list(feature_ids)

        # Plot the leaves scores if you are profile the scores
        if what == "scores":
            feature_names.append("Leaves")
            feature_ids.append(-2)

        trees = self.trees
        if isinstance(trees[0], list):
            trees = sum(trees, [])  # flatten list of lists

        if what == "thresholds":
            values = np.concatenate([np.array(tree.threshold) for tree in trees])
        elif what == "scores":
            values = np.concatenate([np.array(tree.value) for tree in trees])
        else:
            raise ValueError("Invalid profile type: must be 'thresholds' or 'scores'")

        feat_split = np.concatenate([np.array(tree.feature) for tree in trees])[values != 0]
        values = np.abs(values[values != 0])

        # Initialize base position for the y-axis
        base_position = 0

        # Store y-tick positions and labels
        yticks_positions = []
        yticks_labels = []

        for feature_name, feature_id in zip(feature_names, feature_ids):
            ax.axhline(base_position + 1.05, color="black", alpha=0.2, linestyle="--", lw=1)
            ax.axhline(base_position - 1.05, color="black", alpha=0.2, linestyle="--", lw=1)

            ax.boxplot(
                values[feat_split == feature_id],
                positions=[base_position],
                widths=1.1,
                vert=False,
                patch_artist=True,
                boxprops={"facecolor": "dodgerblue"},
                whiskerprops={
                    "alpha": 1,
                    "color": "black",
                    "linestyle": "-",
                    "linewidth": 4,
                },
                capprops={"color": "black", "linestyle": "-", "linewidth": 3,},
                medianprops={"linewidth":3},
                notch=False,
                whis=[0, 100],
                showfliers=False,
            )

            ax.boxplot(
                values[feat_split == feature_id],
                positions=[base_position],
                widths=0.9,
                vert=False,
                patch_artist=True,
                boxprops={"alpha": 0},
                whiskerprops={"linewidth":2, "color": "white", "linestyle": "--"},
                capprops={"linewidth":3, "color": "grey"},
                medianprops={"alpha":0},
                notch=False,
                whis=[2.5, 97.5],
                showfliers=False,
            )

            # Update y-ticks positions and labels
            yticks_positions.append(base_position)
            yticks_labels.append(feature_name)

            # Increment base position for the next entry
            base_position += 2.1  # Adjust spacing between histograms
            
        ax.set_yticks(yticks_positions)
        ax.set_yticklabels(yticks_labels)

        ax.set_xscale("log", base=2)
        ax.grid(True)
        ax.set_title("Thresholds" if what == "thresholds" else "Scores")

        return ax

class ModelMetaData:
    def __init__(self):
        self.version = version
        self.time = datetime.datetime.now()
        self.host = platform.node()
        self.user = getpass.getuser()

    def _to_dict(self):
        return {'version' : str(self.version),
                'host'    : self.host,
                'user'    : self.user,
                'time'    : self.time.timestamp()}

    def _from_dict(d):
        mmd = ModelMetaData()
        mmd.version = Version(d.get('version', None))
        mmd.host = d.get('host', None)
        mmd.user = d.get('user', None)
        mmd.time = datetime.datetime.fromtimestamp(d.get('time', 0))
        return mmd

def make_model(ensembleDict, config=None):
    from conifer.backends import get_backend
    backend = None
    if config is None:
        backend = 'python'
    else:
        for k in ['backend', 'Backend']:
            b = config.get(k, None)
            if b is not None:
                backend = b
        if backend is None:
            logger.warn('Backend not specified in configuration, loading as ModelBase.')
            return ModelBase(ensembleDict, config)
    backend = get_backend(backend)
    return backend.make_model(ensembleDict, config)

def load_model(filename, new_config=None):
    '''
    Load a Model from JSON file

    Parameters
    ----------
    filename: string
        filename to load from
    new_config: dictionary (optional)
        if provided, override the configuration specified in the JSON file
    '''
    with open(filename, 'r') as json_file:
        js = json.load(json_file)

    if new_config is not None:
        config = new_config
    else:
        config = js.get('config', None)
        if config is None:
            logger.warn('No config found in JSON. The model may be loaded incorrectly')

    metadata = js.get('metadata', None)
    if metadata is not None:
        metadata = [ModelMetaData._from_dict(mmd) for mmd in metadata]
    else:
        metadata = []

    model = make_model(js, config)
    model._metadata = metadata + model._metadata
    return model
