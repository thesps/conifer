from conifer import backends
import numpy as np
import os
import json
import copy
import logging
logger = logging.getLogger(__name__)

class Model:

    '''
    Conifer BDT representation class
    Primary interface to write, compile, execute, and synthesize conifer projects
    '''

    def __init__(self, ensembleDict, config):
        self.backend = backends.get_backend(config.get('Backend', None))
        self._ensembleDict = ensembleDict
        self.config = config
        subset_keys = ['max_depth', 'n_trees', 'n_features', 'n_classes']
        subset_dict = {key: self._ensembleDict[key] for key in subset_keys}
        logger.debug(f"Converted BDT with parameters {json.dumps(subset_dict)}")
        def _make_stamp():
            import datetime
            return int(datetime.datetime.now().timestamp())
        self._stamp = _make_stamp()

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
        ensembleDict = copy.deepcopy(self._ensembleDict)
        cfg = copy.deepcopy(self.config)
        ensembleDict['Config'] = cfg
        if filename is None:
            filename = f"{cfg['OutputDir']}/{cfg['ProjectName']}.json"
            directory = cfg['OutputDir']
        else:
            directory = filename.split('/')[:-1]
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Saving model to {directory}")
        with open(filename, 'w') as f:
            json.dump(ensembleDict, f)
            f.close()

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
        '''
        self.backend.build(self.config, **kwargs)

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

def load_model(filename):
    '''
    Load a Model from JSON file

    Parameters
    ----------
    filename: string
        filename to load from
    '''
    logger.info(f'Loading Model from {filename}')
    dictionary = json.load(open(filename, 'r'))
    config = dictionary.get('Config')
    if config is not None:
        del dictionary['Config']
    return Model(dictionary, config)