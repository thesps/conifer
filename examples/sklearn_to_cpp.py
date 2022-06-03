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

stamp = int(datetime.datetime.now().timestamp())
# Create a conifer config
cpp_cfg = conifer.backends.cpp.auto_config()
# Set the output directory to something unique
cpp_cfg['OutputDir'] = 'prj_cpp_{}'.format(stamp)

# Create and compile the model
cpp_model = conifer.model(clf, conifer.converters.sklearn,
                          conifer.backends.cpp, cpp_cfg)
cpp_model.compile()

# Create a conifer config
hls_cfg = conifer.backends.xilinxhls.auto_config()
# Set the output directory to something unique
hls_cfg['OutputDir'] = 'prj_hls_{}'.format(stamp)

# Create and compile the model
hls_model = conifer.model(clf, conifer.converters.sklearn,
                          conifer.backends.xilinxhls, hls_cfg)
hls_model.compile()

# Run the C++ and get the output
y_cpp = cpp_model.decision_function(X)
y_hls = hls_model.decision_function(X)
y_skl = clf.decision_function(X)