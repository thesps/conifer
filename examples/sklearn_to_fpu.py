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

# First, either build an FPU yourself (see build_fpu_alveo.py and build_fpu_pynq.py)
# Or download one from the conifer website for your device.
# Source the Xilinx RunTime environment
# Then, load the FPU onto Alveo (change the path to point to your xclbin file)
device = conifer.backends.fpu.runtime.AlveoDriver('fpu-u200_100TE_512N_DS/fpu.xclbin')
print(device.config)

# Make a random dataset from sklearn 'hastie'
X, y = make_hastie_10_2(random_state=0, n_samples=1_000)
X = X.astype('float32') # for FPU compatibility

# Train a BDT
clf = GradientBoostingClassifier(n_estimators=32, learning_rate=1.0,
                                 max_depth=5, random_state=0).fit(X, y)

stamp = int(datetime.datetime.now().timestamp())
# Create a conifer config
cfg = conifer.backends.fpu.auto_config()

# Important! Set the target FPU config to match the one loaded on the board
cfg['FPU'] = device.config

# Create and compile the model for target FPU
model = conifer.converters.convert_from_sklearn(clf, cfg)

# Load this model onto the FPU
# Important! Set the batch size to allocate buffers
# Inference is more efficient with larger batches
model.attach_device(device, batch_size=X.shape[0])

# Execute model predictions on FPU
y_fpu = model.decision_function(X)

# Execute model prediction on sklearn
y_skl = clf.decision_function(X)
