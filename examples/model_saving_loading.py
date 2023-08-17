'''
This example demonstrates saving and loading of conifer models
Example BDT creation from: https://scikit-learn.org/stable/modules/ensemble.html
'''

from sklearn.datasets import make_hastie_10_2
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
import conifer
import datetime
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.WARNING)
logger = logging.getLogger('conifer')
logger.setLevel(logging.DEBUG)

##### Part 1 - dataset
# make a random dataset from sklearn 'hastie'
X, y = make_hastie_10_2(random_state=0)
X_train, X_test = X[:2000], X[2000:]
y_train, y_test = y[:2000], y[2000:]

##### Part 2 - model
# train a BDT
clf = GradientBoostingClassifier(n_estimators=20, learning_rate=1.0,
                                 max_depth=3, random_state=0).fit(X_train, y_train)

# create a timestamp for a unique project directory
stamp = int(datetime.datetime.now().timestamp())

##### Part 3 - conversion, saving
# create a conifer config for xilinxhls backend
cfg_0 = conifer.backends.xilinxhls.auto_config()
cfg_0['OutputDir'] = f'prj_{stamp}_0'

# convert the model
model_0 = conifer.converters.convert_from_sklearn(clf, cfg_0)

# compile the model for CPU inference.
# compile, write, build, and save methods save the model as a JSON file to <output dir>/<project name>.json
model_0.compile()

y_conifer_0 = model_0.decision_function(X)

##### Part 4 - loading (same config)
# load the model
# the configuration is also read from the file so the same backend and settings are applied
model_1 = conifer.model.load_model(f'prj_{stamp}_0/my_prj.json')

# write it to a different directory
model_1.output_dir = f'prj_{stamp}_1'
model_1.compile()

y_conifer_1 = model_1.decision_function(X)

# check the predictions match
print(f'Original/Loaded model arrays equal: {np.array_equal(y_conifer_0, y_conifer_1)}')

##### Part 5 - loading (new config)
# a new configuration can be applied when loading the model
# here we change the backend to 'cpp' and use floating point data types
cfg_2 = conifer.backends.cpp.auto_config()
cfg_2['OutputDir'] = f'prj_{stamp}_2'
model_2 = conifer.model.load_model(f'prj_{stamp}_0/my_prj.json', new_config=cfg_2)
print(f'Original model backend: {model_0.config.backend}')
print(f'Loaded model backend:   {model_2.config.backend}')

##### Part 6 - convert then save
# a model can be converted to conifer without a configuration
# we can use the native decision_function to check the conversion
# we can also save the model to JSON for later reloading

model_3 = conifer.converters.convert_from_sklearn(clf)
print(f'Conversion with no config backend: {model_3.config.backend}')
model_3.save(f'prj_{stamp}_3.json')