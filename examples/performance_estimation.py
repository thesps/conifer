# conifer includes modelling of the resource and latency usage for the VHDL and HLS backends
# this can be used for rapid performance estimation and performance-aware training optimisation

from ucimlrepo import fetch_ucirepo  # To download the dataset
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import conifer
import datetime
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.WARNING)
logger = logging.getLogger('conifer')
logger.setLevel(logging.DEBUG)

# Download the cover types dataset
# https://archive.ics.uci.edu/dataset/31/covertype
covertype_repo = fetch_ucirepo(id=31)
dataset = pd.concat([covertype_repo.data.features, covertype_repo.data.targets], axis=1)
features = ["Elevation", "Aspect", "Slope", "Horizontal_Distance_To_Hydrology",
            "Vertical_Distance_To_Hydrology", "Horizontal_Distance_To_Roadways",
            "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm",
            "Horizontal_Distance_To_Fire_Points"]
dataset = dataset[features + ["Cover_Type"]]

# Covert type as text
dataset["Cover_Type"] = dataset["Cover_Type"].map({
    1: "Spruce/Fir",
    2: "Lodgepole Pine",
    3: "Ponderosa Pine",
    4: "Cottonwood/Willow",
    5: "Aspen",
    6: "Douglas-fir",
    7: "Krummholz"
})
filtered_dataset = dataset[dataset["Cover_Type"].isin(["Spruce/Fir", "Cottonwood/Willow"])]

# Split into features/target and train/test sets
X = filtered_dataset[features].to_numpy(dtype=np.float32)
y = (filtered_dataset["Cover_Type"] == "Spruce/Fir").to_numpy(dtype=np.int32)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Standardise the features so the default fixed-point precision doesn't saturate
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train).astype(np.float32)
X_test = scaler.transform(X_test).astype(np.float32)

# Train a BDT
clf = GradientBoostingClassifier(n_estimators=20, learning_rate=1.0,
                                 max_depth=3, random_state=0).fit(X_train, y_train)

# Create a conifer config
cfg = conifer.backends.xilinxhls.auto_config()
# Set the output directory to something unique
cfg['OutputDir'] = 'prj_{}'.format(int(datetime.datetime.now().timestamp()))

# Create and compile the model
model = conifer.converters.convert_from_sklearn(clf, cfg)
model.compile()

# Run HLS C Simulation and get the output
y_hls = model.decision_function(X_test)
y_skl = clf.decision_function(X_test)
print('max abs diff between conifer and sklearn decision_function: {}'.format(
    np.max(np.abs(y_hls.reshape(y_skl.shape) - y_skl))))

# Estimate the resource usage and latency before running synthesis, which can be time consiming
print('conifer performance estimates (pre synthesis):')
print(conifer.utils.performance.prediction.performance_estimates(model))

# Synthesize the model
model.build(synth=True, vsynth=True)
print('conifer HLS synthesis result:')
print(model.read_report())
