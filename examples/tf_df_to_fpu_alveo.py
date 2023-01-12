'''
Example BDT creation from: https://scikit-learn.org/stable/modules/ensemble.html
Train a BDT and convert it FPU instructions. Run build_fpu.py first!
'''


from sklearn.datasets import make_hastie_10_2
import tensorflow_decision_forests as tfdf
import conifer
import datetime
import logging
import sys
import json
import numpy as np
import os

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

# Make a random dataset from sklearn 'hastie'
X, y = make_hastie_10_2(random_state=0, n_samples=100_000)

# Train a BDT
model = tfdf.keras.GradientBoostedTreesModel(
    num_trees=100, max_depth=6, verbose=0)
model.fit(x=X, y=y, verbose=1)

stamp = int(datetime.datetime.now().timestamp())
os.makedirs(f'prj_{stamp}', exist_ok=True)
model.save(f'prj_{stamp}/tf_df_hastie_100_6')

# load the FPU onto the Alveo card and read its info
device = conifer.backends.fpu.fpu_driver.AlveoDriver('fpu-08-u200_100TE_512N_NDS/my_prj.xclbin')
print(device.get_info())

# create a configuration to convert the model
# set the FPU config to the one we read back from the device
cfg = conifer.backends.fpu.auto_config()
cfg['FPU'] = device.config

# convert the model and scale to integers
conifer_model = conifer.converters.convert_from_tf_df(model, cfg)
conifer_model.scale(1024, 1024)

# attach the FPU device to the model (the model is laoded to the FPU)
conifer_model.attach_device(device)

y_alveo = conifer_model.decision_function(X[0] * 1024) / 1024
y_tf_df = model.predict(np.expand_dims(X[0], 0))

print(f'Alveo prediction: {y_alveo}')
print(f'TF DF prediction: {y_tf_df}')
