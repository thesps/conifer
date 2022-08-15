# Example BDT creation from: https://scikit-learn.org/stable/modules/ensemble.html

from sklearn.datasets import load_boston
from sklearn.utils import shuffle
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
import conifer
import datetime

# Load Boston regression dataset
boston = load_boston()
X, y = shuffle(boston.data, boston.target, random_state=13)
X = X.astype(np.float32)
offset = int(X.shape[0] * 0.9)
X_train, y_train = X[:offset], y[:offset]
X_test, y_test = X[offset:], y[offset:]

# Train a BDT
clf = GradientBoostingRegressor(n_estimators=100, max_depth=4, min_samples_split=2,
                                learning_rate=0.01, loss='ls')
clf.fit(X_train, y_train)

# Create a conifer config
cfg = conifer.backends.xilinxhls.auto_config()
# Set the output directory to something unique
cfg['OutputDir'] = 'prj_{}'.format(int(datetime.datetime.now().timestamp()))
cfg['Precision'] = 'ap_fixed<64,32>'

# Create and compile the model
model = conifer.converters.convert_from_sklearn(clf, cfg)
model.compile()

# Run HLS C Simulation and get the output
#y_hls = model.decision_function(X)
y_hls, y_trees = model.decision_function(X)
y_skl = clf.predict(X)

# Synthesize the model
# model.build()
