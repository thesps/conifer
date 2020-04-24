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
