# Example BDT creation from: https://scikit-learn.org/stable/modules/ensemble.html

from sklearn.datasets import make_hastie_10_2
from sklearn.ensemble import GradientBoostingClassifier
import conifer
import datetime

# Make a random dataset from sklearn 'hastie'
X, y = make_hastie_10_2(random_state=0)
X_train, X_test = X[:2000], X[2000:]
y_train, y_test = y[:2000], y[2000:]

# Train a BDT
clf = GradientBoostingClassifier(n_estimators=20, learning_rate=1.0,
    max_depth=3, random_state=0).fit(X_train, y_train)

# Create a conifer config
cfg = conifer.backends.vivadohls.auto_config()
# Set the output directory to something unique
cfg['OutputDir'] = 'prj_{}'.format(int(datetime.datetime.now().timestamp()))

# Create and compile the model
model = conifer.model(clf, conifer.converters.sklearn, conifer.backends.vivadohls, cfg)
model.compile()

# Run HLS C Simulation and get the output
y_hls = model.decision_function(X)[:,0]
y_skl = clf.decision_function(X)

# Synthesize the model
model.build()
