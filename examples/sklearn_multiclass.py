from sklearn.datasets import load_iris
from sklearn.ensemble import GradientBoostingClassifier
import conifer
import datetime

iris = load_iris()
X, y = iris.data, iris.target

clf = GradientBoostingClassifier(n_estimators=20, learning_rate=1.0,
                                 max_depth=3, random_state=0).fit(X, y)

# Create a conifer config
cfg = conifer.backends.xilinxhls.auto_config()
# Set the output directory to something unique
cfg['OutputDir'] = 'prj_{}'.format(int(datetime.datetime.now().timestamp()))

model = conifer.converters.convert_from_sklearn(clf, cfg)
model.compile()

# Run HLS C Simulation and get the output
y_hls = model.decision_function(X)
y_skl = clf.decision_function(X)

# Synthesize the model
model.build()
