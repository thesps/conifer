from . import backends
from . import converters
import numpy as np
import os
import sys

class model:

    def __init__(self, bdt, converter, backend=backends.vivadohls, config=None):
        self._ensembleDict = converter.convert(bdt)
        self.backend = backend
        if config is not None:
            self.config = config
        else:
            self.config = backend.auto_config()

    def set_config(self, config):
        self.config = config

    def get_config(self):
        return self.config

    def write(self):
        self.backend.write(self._ensembleDict, self.config)

    def compile(self):
        self.write()
        self.backend.sim_compile(self.config)

    def decision_function(self, X, trees=False):
        return self.backend.decision_function(X, self.config, trees=trees)

    def build(self):
        self.backend.build(self.config)

    def profile(self, bins=50, return_data=False, return_figure=True):
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise Exception("matplotlib not found. Please install matplotlib")
        value = np.array([tree['value'] for trees in self._ensembleDict['trees'] for tree in trees]).flatten()
        threshold = np.array([tree['threshold'] for trees in self._ensembleDict['trees'] for tree in trees]).flatten()
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
