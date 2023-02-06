'''
Example BDT creation from: https://scikit-learn.org/stable/modules/ensemble.html
Train a BDT and convert it FPU instructions. Run build_fpu.py first!
'''


from sklearn.datasets import make_hastie_10_2
from sklearn.ensemble import GradientBoostingClassifier
import conifer
import datetime
import logging
import sys
import json
import numpy as np

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

# Make a random dataset from sklearn 'hastie'
X, y = make_hastie_10_2(random_state=0)#, n_samples=500_000)

# Train a BDT
clf = GradientBoostingClassifier(n_estimators=50, learning_rate=1.0,
                                 max_depth=3, random_state=0).fit(X, y)

stamp = int(datetime.datetime.now().timestamp())
# Create a conifer config
cfg = conifer.backends.fpu.auto_config()
# Set the output directory to something unique
cfg['OutputDir'] = 'prj_fpu_{}'.format(stamp)
# set the target FPU config to match the one downloaded
with open('fpu.json') as f:
  cfg['FPU'] = json.load(f)

# Create and compile the model
model = conifer.converters.convert_from_sklearn(clf, cfg)
model.write()
y_clf = clf.decision_function(X)
np.save(f'{cfg["OutputDir"]}/X.npy', X.astype('float32'))
np.save(f'{cfg["OutputDir"]}/y.npy', y.astype('float32'))

# to run inference on a pynq-z2, copy the driver conifer/backends/fpu/runtime.py to the pynq,
# as well as the .zip archive containing the FPU binary,
# and the nodes.json, my-prj.json saved by the model.write() above,
# and the X.npy, y.npy reference data
# inference currently works like:
# import json
# from runtime import ZynqDriver
# import numpy as np
# fpu = ZynqDriver('fpu.bit')
# print(fpu.get_info())
# nodes = json.load(open('nodes.json'))
# model = json.load(open('my-prj.json'))
# X = np.load('X.npy')
# fpu.load(nodes['nodes'], nodes['scales'], model['n_features'], model['n_classes'], batch_size=X.shape[0])
# y_fpu = fpu.predict(X)
