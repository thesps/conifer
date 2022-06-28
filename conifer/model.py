from . import backends
from . import converters
import numpy as np
import os
import sys
import json
import logging
logger = logging.getLogger(__name__)

class model:

    def __init__(self, bdt, converter, backend=backends.xilinxhls, config=None):
        self._ensembleDict = converter.convert(bdt)
        self.backend = backend
        logger.info(f'Converting BDT with {converter.__name__} converter and {backend.__name__} backend')
        if config is not None:
            self.config = config
        else:
            logger.info('No configuration provided, creating the default configuration from the backend')
            self.config = backend.auto_config()
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

    def write(self):
        self.backend.write(self)

    def compile(self):
        self.write()
        self.backend.sim_compile(self)

    def decision_function(self, X, trees=False):
        return self.backend.decision_function(X, self, trees=trees)

    def build(self, **kwargs):
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
