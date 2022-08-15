<img src="conifer_v1.png" width="250" alt="conifer">

Conifer translates trained Boosted Decision Trees to FPGA firmware for extreme low latency inference. 

# Installation
Conifer is on the Python Package Index, so install it like:
```
pip install conifer
```

# About
Conifer converts from popular BDT training frameworks, and can emit code projects in different FPGA languages.
Available converters:
- scikit-learn
- xgboost
- ONNX - giving access to other training libraries such as lightGBM and CatBoost with ONNXMLTools
- TMVA

Available backends:
- Xilinx HLS - for best results use latest Vitis HLS, but Vivado HLS is also supported (conifer uses whichever is on your `$PATH`)
- VHDL - a direct-to-VHDL implementation, deeply pipelined for high clock frequencies
- C++ - intended for bit-accurate emulation on CPU with a single include header file

See our paper in JINST: "[Fast inference of Boosted Decision Trees in FPGAs for particle physics](https://iopscience.iop.org/article/10.1088/1748-0221/15/05/P05026)".

Conifer originated as a development for [hls4ml](https://fastmachinelearning.org/hls4ml/), and borrows heavily from the code and ideas developed for it.

# Usage
```
from sklearn.ensemble import GradientBoostingClassifier
# Train a BDT
clf = GradientBoostingClassifier().fit(X_train, y_train)

# Create a conifer config dictionary
cfg = conifer.backends.xilinxhls.auto_config()
# Change the bit precision (print the config to see everything modifiable)
cfg['Precision'] = 'ap_fixed<12,4>' 

# Create the conifer model
model = conifer.model(clf, conifer.converters.sklearn,
                      conifer.backends.xilinxhls, cfg)
# Write the HLS project and compile the C++-Python bridge                      
model.compile()

# Run bit-accurate prediction on the CPU
y_hls = model.decision_function(X)
y_skl = clf.decision_function(X)

# Synthesize the model for the target FPGA
model.build()
```

Check the examples directory for examples to get started with, and the BDT part of the [hls4ml tutorial](https://github.com/fastmachinelearning/hls4ml-tutorial).
