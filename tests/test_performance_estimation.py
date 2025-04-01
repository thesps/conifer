import conifer.utils.performance.prediction
import pytest
import conifer

dummy_model_dict = {'max_depth'    : 3,
                    'n_trees'      : 2,
                    'n_features'   : 4,
                    'n_classes'    : 2,
                    'norm'         : 1,
                    'init_predict' : [1.],
                    'library'      : 'conifer',
                    'splitting_convention' : '<',
                    'trees' : [[{'children_left'  : [1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, -1],
                                  'children_right' : [8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, -1],
                                  'feature'        : [0, 1, 2, -2, -2, 3, -2, -2, 2, 1, -2, -2, -2],
                                  'threshold'      : [1.0, -1.3, -2.1, -2.0, -2.0, 1.2, -2.0, -2.0, 0.9, 1.5, -2.0, -2.0, -2.0],
                                  'value'          : [1.0, -1.3, -2.1, -2.0, -2.0, 1.2, -2.0, -2.0, 0.9, 1.5, -2.0, -2.0, -2.0]
                                  }],
                                [{'children_left'  : [1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, -1],
                                  'children_right' : [8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, -1],
                                  'feature'        : [2, 1, 3, -2, -2, 0, -2, -2, 3, 1, -2, -2, -2],
                                  'threshold'      : [1.0, -1.3, -2.1, -2.0, -2.0, 1.2, -2.0, -2.0, 0.9, 1.5, -2.0, -2.0, -2.0],
                                  'value'          : [1.0, -1.3, -2.1, -2.0, -2.0, 1.2, -2.0, -2.0, 0.9, 1.5, -2.0, -2.0, -2.0]
                                  }]]}

@pytest.fixture
def dummy_conifer_model():
  return conifer.model.ModelBase(dummy_model_dict)

# -------------------------------------------------------------------------------------------
# A few tests just to check that the metrics gathering functions run without error
def test_metrics_sparsity(dummy_conifer_model):
  conifer.utils.performance.metrics.get_sparsity_metrics(dummy_conifer_model)

def test_metrics_feature_frequency(dummy_conifer_model):
  conifer.utils.performance.metrics.get_feature_frequency_metrics(dummy_conifer_model)

def test_metrics_model(dummy_conifer_model):
  conifer.utils.performance.metrics.get_model_metrics(dummy_conifer_model)
# -------------------------------------------------------------------------------------------

@pytest.mark.parametrize('estimator', [conifer.utils.performance.prediction.vhdlLatencyEstimator,
                                       conifer.utils.performance.prediction.vhdlLUTEstimator,
                                       conifer.utils.performance.prediction.vhdlFFEstimator,
                                       conifer.utils.performance.prediction.hlsLatencyEstimator,
                                       conifer.utils.performance.prediction.hlsLUTEstimator,
                                       conifer.utils.performance.prediction.hlsFFEstimator,
                                       ])
def test_performance_estimators(dummy_conifer_model, estimator):
  estimator.predict(dummy_conifer_model)

@pytest.mark.parametrize('backend', ['xilinxhls', 'vhdl'])
def test_performance_estimates(dummy_conifer_model, backend):
  conifer.utils.performance.prediction.performance_estimates(dummy_conifer_model, backend)
