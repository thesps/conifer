# Example BDT creation from: https://xgboost.readthedocs.io/en/latest/get_started.html
# With data import from: https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html

from sklearn.datasets import load_iris
import xgboost as xgb
import conifer
import datetime
from scipy.special import softmax

# Load the iris dataset from sklearn'
iris = load_iris()
X, y = iris.data, iris.target

# Train a BDT using the scikit-learn API
bst = xgb.XGBClassifier(n_estimators=20, max_depth=3,
                        learning_rate=1., objective='multi:softmax')
bst = bst.fit(X, y)

# Create a conifer config
cfg = conifer.backends.xilinxhls.auto_config()
# Set the output directory to something unique
cfg['OutputDir'] = 'prj_{}'.format(int(datetime.datetime.now().timestamp()))

# Create and compile the model
# We need to pass the Booster object to conifer, so from xgboost's scikit-learn API,
# we call bst.get_booster()
model = conifer.converters.convert_from_xgboost(bst.get_booster(), cfg)
model.compile()

# Run HLS C Simulation and get the output
# xgboost 'predict' returns a probability like sklearn 'predict_proba'
# so we need to compute the probability from the decision_function returned
# by the HLS C Simulation
y_hls = softmax(model.decision_function(X), axis=1)
y_xgb = bst.predict_proba(X)

# Synthesize the model
model.build()
