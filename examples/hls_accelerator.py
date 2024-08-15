'''
This example demonstrates creation of an FPGA accelerator with the HLS backend
Conifer can create two types of accelerators: static and dynamic
This example creates a static accelerator - the trained BDT is part of the FPGA binary
Check out the FPU documentation and examples for dynamic accelerators
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

# make a random dataset from sklearn 'hastie'
X, y = make_hastie_10_2(random_state=0)
X_train, X_test = X[:2000], X[2000:]
y_train, y_test = y[:2000], y[2000:]

# train a BDT
clf = GradientBoostingClassifier(n_estimators=20, learning_rate=1.0,
                                 max_depth=3, random_state=0).fit(X_train, y_train)

# create a timestamp for a unique project directory
stamp = int(datetime.datetime.now().timestamp())

# create a conifer config
cfg = conifer.backends.xilinxhls.auto_config()
# Set the output directory to something unique
cfg['OutputDir'] = 'prj_{}'.format(int(datetime.datetime.now().timestamp()))

# we need to specify the target board and interface type (the data type sent between host-accelerator)
# advanced: define a new board by creating a conifer.backends.boards.BoardConfig object
# see all available boards:
print(f'conifer available boards: {", ".join(conifer.backends.boards.get_available_boards())}')

# target pynq-z2 board and float interface (the conversion from float to ap_fixed is performed on the FPGA)
accelerator_config = {'Board' : 'pynq-z2',
                      'InterfaceType': 'float'}
cfg['AcceleratorConfig'] = accelerator_config

# create and compile the model
model = conifer.converters.convert_from_sklearn(clf, cfg)
model.compile()

# run HLS C Simulation and get the output
y_hls = model.decision_function(X)
y_skl = clf.decision_function(X)

# save the data to numpy files to load them again on the pynq
np.save(f'{cfg["OutputDir"]}/X.npy', X)
np.save(f'{cfg["OutputDir"]}/y_hls.npy', y_hls)

# synthesize the model, targeting bitfile creation and packaging
# warning: this may take some time!
model.build(bitfile=True, package=True)

# copy the following files at <output dir>to the pynq-z2:
# - my_prj.zip
# - X.npy
# - y_hls.npy
# install conifer on the device (!pip install conifer in a jupyter notebook)
# the following code shows how to use the accelerator on the device:

# import conifer
# import numpy as np
# X = np.load('X.npy').astype('float32')
# y_hls = np.load('y_hls.npy').astype('float32')
# # load the bitstream onto the PL, provide the batch size to allocate buffers
# accelerator = conifer.backends.xilinxhls.runtime.ZynqDriver('my_prj.bit', batch_size=X.shape[0])
# # execute inference
# y_accelerator = accelerator.decision_function(X)
# print(f'first 10 examples hls:         {y_hls[:10]}')
# print(f'first 10 examples accelerator: {y_accelerator[:10]}')
