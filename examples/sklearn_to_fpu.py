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
import json

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

# Make a random dataset from sklearn 'hastie'
X, y = make_moons(random_state=0)
X_train, X_test = X[:2000], X[2000:]
y_train, y_test = y[:2000], y[2000:]

# Train a BDT
clf = GradientBoostingClassifier(n_estimators=50, learning_rate=1.0,
                                 max_depth=7, random_state=0).fit(X_train, y_train)

stamp = int(datetime.datetime.now().timestamp())
# Create a conifer config
cfg = conifer.backends.fpu.auto_config()
# Set the output directory to something unique
cfg['OutputDir'] = 'prj_fpu_{}'.format(stamp)
# set the target FPU config to match the one built in build_fpu.py
with open('fpu-07_100TE_512N_NDS/my_prj.json') as f:
  cfg['FPU'] = json.load(f)

# Create and compile the model
model = conifer.converters.convert_from_sklearn(clf, cfg)
# Note: right now the FPU dynamic scaler doesn't work so we have to scale
# parameters here. Remember to scale the features when running predictions!
model.scale(1024, 1024)
model.write()

# to run inference on a pynq-z2, copy the driver conifer/backends/fpu/fpu_driver.py to the pynq,
# as well as the .zip archive from build_fpu.py containing the FPU binary,
# and the nodes.json saved by the model.write() above
# inference currently works like:
# import json
# from fpu_driver import ZynqDriver
# import numpy as np
# fpu = ZynqDriver('fpu.bit')
# print(fpu.get_info())
# model = json.load(open('prj_fpu_.../nodes.json'))
# fpu.load(model['nodes'], model['scales'])
# X = np.zeros(16, dtype='int32')
# fpu.predict(X)

# Create a C++ backend model for reference
# 16 bit variables with 1024 (= 2^10) scales equates to ap_fixed<16,6>
with open(f'prj_fpu_{stamp}/my-prj.json') as f:
  js = json.load(f)
cpp_cfg = js['config']
cpp_cfg['backend'] = 'cpp'
cpp_cfg['input_precision'] = 'ap_fixed<16,6>'
cpp_cfg['threshold_precision'] = 'ap_fixed<16,6>'
cpp_cfg['score_precision'] = 'ap_fixed<16,6>'
cpp_cfg['output_dir'] += '_cpp'

model = conifer.model.load_model(f'prj_fpu_{stamp}/my-prj.json', new_config=cpp_cfg)
model.compile()
y_cpp = model.decision_function(X)
y_cpp_scaled = y_cpp * 1024 # <- this should match the values returned by fpu.predict(X) (on the same inputs, also scaled by 1024)

