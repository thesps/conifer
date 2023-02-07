from conifer import __version__ as version
import numpy as np
import os
import json
import copy
import datetime
import platform
import getpass
import logging
logger = logging.getLogger(__name__)

class DecisionTreeBase:
  '''
  Conifer DecisionTreeBase representation class
  '''
  _tree_fields = ['feature', 'threshold', 'value', 'children_left', 'children_right']
  def __init__(self, treeDict):
    for key in DecisionTreeBase._tree_fields:
      val = treeDict.get(key, None)
      assert val is not None, f"Missing expected key {key} in treeDict"
      setattr(self, key, val)

  def n_nodes(self):
    return len(self.feature)

  def n_leaves(self):
    return len([n for n in self.feature if n == -2])

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

    def __init__(self, configDict, validate=True):
        for key in self._config_fields:
            for k in [key, *self._alternates[key]]:
                val = configDict.get(k, None)
                if val is not None:
                    setattr(self, key, val)
        if validate:
            self._validate()

    def _validate(self):
        from conifer.backends import get_backend
        vals = {}
        for key in self._config_fields:
            vals[key] = getattr(self, key, None)
        assert not (None in vals.values()), f'Missing some required configuration, have: {vals}'
        assert get_backend(self.backend) is not None, f'Backend {self.backend} not found'

    def _to_dict(self):
        dictionary = {k : getattr(self, k) for k in self._config_fields}
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

    _ensemble_fields = ['n_classes', 'n_features', 'n_trees', 'max_depth', 'init_predict', 'norm']

    def __init__(self, ensembleDict, configDict, metadata=None):
        for key in ModelBase._ensemble_fields:
            val = ensembleDict.get(key, None)
            assert val is not None, f'Missing expected key {key} in ensembleDict'
            setattr(self, key, val)
        trees = ensembleDict.get('trees', None)
        assert trees is not None, f'Missing expected key {key} in ensembleDict'
        self.trees = [[DecisionTreeBase(treeDict) for treeDict in trees_class] for trees_class in trees]
        self.config = ConfigBase(configDict)

        subset_keys = ['max_depth', 'n_trees', 'n_features', 'n_classes']
        subset_dict = {key: getattr(self, key) for key in subset_keys}
        logger.debug(f"Converted BDT with parameters {json.dumps(subset_dict)}")
        def _make_stamp():
            import datetime
            return int(datetime.datetime.now().timestamp())
        self._stamp = _make_stamp()
        if metadata is None:
            self._metadata = [ModelMetaData()]
        else:
            if isinstance(metadata, list):
                self._metadata = metadata
                self._metadata.append(ModelMetaData())
            else:
                self._metadata = [metadata]

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
        if filename is None:
            filename = f"{cfg.output_dir}/{cfg.project_name}.json"
            directory = cfg.output_dir
        else:
            directory = filename.split('/')[:-1]
        os.makedirs(directory, exist_ok=True)
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
        raise NotImplementedError

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

    def profile(self, bins=50, return_data=False, return_figure=True):
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise Exception("matplotlib not found. Please install matplotlib")
        value = np.array([tree['value'] for trees in self._ensembleDict['trees']
                         for tree in trees]).flatten()
        threshold = np.array(
            [tree['threshold'] for trees in self._ensembleDict['trees'] for tree in trees]).flatten()
        hv, bv = np.histogram(value, bins=bins)
        wv = bv[1] - bv[0]
        ht, bt = np.histogram(threshold, bins=bins)
        wt = bt[1] - bt[0]
        figure = plt.figure()
        plt.subplot(1, 2, 1)
        plt.bar(bv[:-1]+wv/2, hv, width=wv)
        plt.xlabel('Distribution of tree values (scores)')
        plt.subplot(1, 2, 2)
        plt.bar(bt[:-1]+wt/2, ht, width=wt)
        plt.xlabel('Distribution of tree thresholds')
        if return_data and return_figure:
            return (value, threshold, figure)
        elif return_data:
            return (value, threshold)
        elif return_figure:
            return (figure)

class ModelMetaData:
    def __init__(self):
        self.version = version
        self.time = datetime.datetime.now()
        self.host = platform.node()
        self.user = getpass.getuser()

    def _to_dict(self):
        return {'version' : self.version,
                'host'    : self.host,
                'user'    : self.user,
                'time'    : self.time.timestamp()}

    def _from_dict(d):
        mmd = ModelMetaData()
        mmd.version = d.get('version', None)
        mmd.host = d.get('host', None)
        mmd.user = d.get('user', None)
        mmd.time = datetime.datetime.fromtimestamp(d.get('time', 0))
        return mmd

def make_model(ensembleDict, config):
    from conifer.backends import get_backend
    backend = None
    for k in ['backend', 'Backend']:
        b = config.get(k, None)
        if b is not None:
            backend = b
    if backend is None:
        logger.warn('Backend not specified in configuration, loading as ModelBase. It will not be possible to write out.')
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