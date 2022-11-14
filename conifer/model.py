from conifer import backends
from conifer import __version__ as version
import numpy as np
import os
import json
import jsonpickle
import copy
import datetime
import platform
import logging
logger = logging.getLogger(__name__)

class Model:

    '''
    Conifer BDT representation class
    Primary interface to write, compile, execute, and synthesize conifer projects
    '''

    _ensemble_fields = ['n_classes', 'n_features', 'n_trees', 'max_depth', 'init_predict', 'norm', 'trees']
    _tree_fields = ['feature', 'value', 'children_left', 'children_right']

    def __init__(self, ensembleDict, config, metadata=None):
        self.backend = backends.get_backend(config.get('Backend', 'cpp'))
        for key in Model._ensemble_fields:
            val = ensembleDict.get(key, None)
            assert val is not None, f'Missing expected key {key} in ensembleDict'
            setattr(self, key, val)
        self._ensembleDict = ensembleDict
        self.config = config
        subset_keys = ['max_depth', 'n_trees', 'n_features', 'n_classes']
        subset_dict = {key: self._ensembleDict[key] for key in subset_keys}
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
        json = jsonpickle.encode(self)
        cfg = self.config
        if filename is None:
            filename = f"{cfg['OutputDir']}/{cfg['ProjectName']}.json"
            directory = cfg['OutputDir']
        else:
            directory = filename.split('/')[:-1]
        os.makedirs(directory, exist_ok=True)
        with open(filename, 'w') as f:
            f.write(json)

    def write(self):
        '''
        Write the model files to the output directory specified in configuration
        '''
        self.backend.write(self)

    def compile(self):
        '''
        Compile conifer model ahead of running decision_function.
        Writes the project files first.
        Compilation is carried out by the model backend
        '''
        self.write()
        self.backend.sim_compile(self)

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
        return self.backend.decision_function(X, self, trees=trees)

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
        return self.backend.build(self.config, **kwargs)

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
        self.user = os.getlogin()

def load_model(filename):
    '''
    Load a Model from JSON file

    Parameters
    ----------
    filename: string
        filename to load from
    '''
    json = open(filename, 'r').read()
    model = jsonpickle.decode(json)
    model._metadata.append(ModelMetaData())
    return model