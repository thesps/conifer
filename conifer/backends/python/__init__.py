"""
Python backend.

Runs the model in floating point with no compilation step, as a reference for
validating the conversion itself (independently of any hardware precision).
Three traversal algorithms are available, selected with the 'Algorithm'
config key or with the `algorithm` argument when calling decision_function.
 - 'treewalk' (default): plain root-to-leaf walk of every tree (ModelBase)
 - 'quickscorer': interleaved bitvector traversal of the whole ensemble
   (https://doi.org/10.1145/2766462.2767733)
 - 'blockwise-quickscorer': BWQS, the cache-friendly block-wise variant of
   quickscorer from the same paper. Blocks of 'Tau' trees are traversed for
   'Delta' documents at a time, so one block of QS data structures and result
   bitvectors fits in last-level CPU cache.
"""

import copy
import datetime
from conifer.model import ModelBase, ConfigBase
from conifer.backends.python.quickscorer import QuickScorer
import logging

logger = logging.getLogger(__name__)

_algorithms = ("treewalk", "quickscorer", "blockwise-quickscorer")


class PythonConfig(ConfigBase):
    backend = "python"
    _config_fields = ConfigBase._config_fields + ["algorithm", "tau", "delta"]
    _alternates = {
        **ConfigBase._alternates,
        "algorithm": ["Algorithm"],
        "tau": ["Tau"],
        "delta": ["Delta"],
    }
    # default block sizes for 'blockwise-quickscorer'
    # NOTE: tau/delta need to be tuned per ensemble and per machine, cf. Table 4 of the QuickScorer paper.
    # NOTE: we keep delta large enough to amortize the numpy overhead for each operation.
    # TODO: add a function for hyperparameter autotuning?
    _defaults = {
        **ConfigBase._defaults,
        "backend": "python",
        "algorithm": "treewalk",
        "tau": 128,
        "delta": 4096,
    }

    def __init__(self, configDict, validate=True):
        configDict = dict(configDict)
        for key in ["algorithm", "tau", "delta"]:
            if configDict.get(key) is None and configDict.get(key.capitalize()) is None:
                configDict[key] = self._defaults[key]
        super(PythonConfig, self).__init__(configDict, validate=validate)
        assert self.algorithm in _algorithms, (
            f'Unknown algorithm "{self.algorithm}", expected one of {_algorithms}'
        )


class PythonModel(ModelBase):
    def __init__(self, ensembleDict, config=None, metadata=None):
        # the python backend writes no project files, so all configuration is optional
        config = {} if config is None else copy.deepcopy(config)
        stamp = int(datetime.datetime.now().timestamp())
        if config.get("output_dir") is None and config.get("OutputDir") is None:
            config["output_dir"] = "."
        if config.get("project_name") is None and config.get("ProjectName") is None:
            config["project_name"] = f"conifer_prj_{stamp}"
        if config.get("backend") is None and config.get("Backend") is None:
            config["backend"] = "python"
        super(PythonModel, self).__init__(ensembleDict, config, metadata)
        self.config = PythonConfig(config)
        # to avoid rebuilding on per-call algorithm overrides
        self._scorers = {}

    def compile(self):
        """
        No compilation step: the python backend runs the model directly.
        Provided so that the python backend can be used interchangeably with
        the compiled backends.
        """
        logger.debug("The python backend requires no compilation")
        return True

    def decision_function(self, X, trees=False, return_leaf=False, algorithm=None):
        """
        Compute the decision function of `X`.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
            Input sample

        trees: bool, optional
            If True, returns the decision function of each tree in the ensemble. Otherwise, returns the sum of the decision function of all trees. Defaults to False.

        return_leaf: bool, optional
            If True, returns the leaf node indices of each tree in the ensemble. Otherwise, returns the decision function. Defaults to False.

        algorithm: string, optional
            Traversal algorithm, 'treewalk', 'quickscorer' or
            'blockwise-quickscorer'. Overrides the 'Algorithm' of the model
            configuration for this call only.

        Returns
        ----------
        score: ndarray of shape (n_samples, n_classes) or (n_samples,)
        """
        algorithm = self.config.algorithm if algorithm is None else algorithm
        assert algorithm in _algorithms, (
            f'Unknown algorithm "{algorithm}", expected one of {_algorithms}'
        )
        if algorithm == "treewalk":
            return super(PythonModel, self).decision_function(
                X, trees=trees, return_leaf=return_leaf
            )
        # (bw)quickscorer
        if algorithm not in self._scorers:
            if algorithm == "quickscorer":
                self._scorers[algorithm] = QuickScorer(self)
            else:
                self._scorers[algorithm] = QuickScorer(
                    self, tau=self.config.tau, delta=self.config.delta
                )
        return self._scorers[algorithm].decision_function(X, return_leaf=return_leaf)


def auto_config():
    config = {
        "Backend": "python",
        "ProjectName": "my_prj",
        "OutputDir": "my-conifer-prj",
        "Algorithm": "treewalk",
    }
    return config


def make_model(ensembleDict, config=None):
    return PythonModel(ensembleDict, config)
