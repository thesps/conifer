'''
Example BDT creation from: https://scikit-learn.org/stable/modules/ensemble.html
Train a BDT and convert it FPU instructions. Run build_fpu.py first!
'''


from sklearn.datasets import make_moons
from sklearn.ensemble import GradientBoostingClassifier
import conifer
import datetime
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

# Make a random dataset from sklearn 'hastie'
X, y = make_moons(random_state=0)
X_train, X_test = X[:2000], X[2000:]
y_train, y_test = y[:2000], y[2000:]

# Train a BDT
clf = GradientBoostingClassifier(n_estimators=2, learning_rate=1.0,
                                 max_depth=3, random_state=0).fit(X_train, y_train)

stamp = int(datetime.datetime.now().timestamp())
# Create a conifer config
cfg = conifer.backends.fpu.auto_config()
# Set the output directory to something unique
cfg['OutputDir'] = 'prj_fpu_{}'.format(stamp)
# set the target FPU config to match the one built in 
cfg['FPU']['tree_engines'] = 2
cfg['FPU']['nodes'] = 16
cfg['FPU']['dynamic_scaler'] = False

# Create and compile the model
model = conifer.converters.convert_from_sklearn(clf, cfg)
# Note: right now the FPU dynamic scaler doesn't work so we have to scale
# parameters here. Remember to scale the features when running predictions!
model.scale(1024, 1024)
model.write()

