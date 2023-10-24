<img src="https://github.com/thesps/conifer/raw/master/conifer_v1.png" width="250" alt="conifer">

Conifer translates trained Boosted Decision Trees to FPGA firmware for extreme low latency inference. 

# Installation
Conifer is on the Python Package Index, so install it like:
```
pip install conifer
```

Building FPGA firmware requires tools from Xilinx - a High Level Synthesis tool and Vivado, depending on the choice of backend. We recommend the most recent version Vivado ML and Vitis HLS 2022.2.

Using the C++ backends requires JSON header files from [here](https://github.com/nlohmann/json). Clone or download the repository, and set the environment variable `JSON_ROOT` to the JSON include path. Xilinx's arbitrary precision header files (e.g. `ap_fixed.h`) are required to use these type for emulation. They are automatically found if you have source the Xilinx HLS toolchain, but you can also find them [here](https://github.com/Xilinx/HLS_arbitrary_Precision_Types). If using the C++ backend without a Vivado installation, clone or download Xilinx's repository, and set the environment variable `XILINX_HLS` to the include path.

# About
Conifer converts from popular BDT training frameworks, and can emit code projects in different FPGA languages.


Available converters:
- scikit-learn
- xgboost
- ONNX - giving access to other training libraries such as lightGBM and CatBoost with ONNXMLTools
- TMVA
- Tensorflow Decision Forest (tf_df)

Available backends:
- Xilinx HLS - for best results use latest Vitis HLS, but Vivado HLS is also supported (conifer uses whichever is on your `$PATH`)
- VHDL - a direct-to-VHDL implementation, deeply pipelined for high clock frequencies
- FPU - Forest Processing Unit reusable IP core for flexible BDT inference
- C++ - intended for bit-accurate emulation on CPU with a single include header file
- Python - intended for validation of model conversion and to allow inspection of a model without a configuration

See our paper in JINST: "[Fast inference of Boosted Decision Trees in FPGAs for particle physics](https://iopscience.iop.org/article/10.1088/1748-0221/15/05/P05026)".

Conifer originated as a development for [hls4ml](https://fastmachinelearning.org/hls4ml/), and is developed under the [Fast Machine Learning Lab](https://fastmachinelearning.org/).

# Usage

View the API reference at the [conifer homepage](https://ssummers.web.cern.ch/conifer/)

```
from sklearn.ensemble import GradientBoostingClassifier
# Train a BDT
clf = GradientBoostingClassifier().fit(X_train, y_train)

# Create a conifer config dictionary
cfg = conifer.backends.xilinxhls.auto_config()
# Change the bit precision (print the config to see everything modifiable)
cfg['Precision'] = 'ap_fixed<12,4>' 

# Convert the sklearn model to a conifer model
model = conifer.converters.convert_from_sklearn(clf, cfg)
# Write the HLS project and compile the C++-Python bridge                      
model.compile()

# Run bit-accurate prediction on the CPU
y_hls = model.decision_function(X)
y_skl = clf.decision_function(X)

# Synthesize the model for the target FPGA
model.build()
```

Check the examples directory for examples to get started with, and the BDT part of the [hls4ml tutorial](https://github.com/fastmachinelearning/hls4ml-tutorial).

# Forest Processing Unit
The conifer Forest Processing Unit (FPU) is a flexible IP for fast BDT inference on FPGAs. One FPU can be configured to perform inference of different BDTs without rebuilding the IP or bitstream.

<details>

    <summary>More information</summary>

FPUs comprise multiple Tree Engines (TEs) that operate in parallel. Each TE navigates a Decision Tree from root to leaf and outputs the leaf score. A summing network then combines the class scores to make the BDT prediction. TEs are programmed by the conifer compiler, allowing you to map different BDTs - for example with different numbers of nodes and maximum tree depth - onto the same FPU.

## Downloading the FPU
Premade binaries for select boards are available for [download here](https://ssummers.web.cern.ch/ssummers/conifer/). Navigate to the conifer version, board and configuration and download the bitfile.

## Building the FPU
If you would like to build the FPU yourself, for example if you need a custom configuration or to target a different board, you can use the `FPUBuilder`s in `conifer.backends.fpu`. Check the `build_fpu.py` example for ideas. You can change the number of tree engines, the number of nodes per engine, as well as the bitwidth allocated to each variable. All of this configuration is carried out through a configuration dictionary.

## Running the FPU
The conifer `fpu` backend maps your trained BDT onto a specific FPU configuration, and provides the driver to interact with the FPU - to load (and read) a BDT, and to perform inference.

For a pynq-z2 board the first step is to copy the `fpu_driver.py` and bitstream to the pynq-z2 SD card, then load it like this:

```
from fpu_driver import ZynqDriver
fpu = ZynqDriver('fpu.bit')
```

The FPU stores the configuration settings it was built with, which we can query like this:
```
print(fpu.get_info())
```

```
# model = json.load(open('prj_fpu_.../nodes.json'))
# fpu.load(model['nodes'], model['scales'])
# X = np.zeros(16, dtype='int32')
# fpu.predict(X)
```


</details>

# License
Apart from the source code and binaries of the Forest Processing Unit (FPU), `conifer` is licensed under *Apache v2*. The FPU source code and binaries are licensed under the [*CERN-OHL-P v2*](https://cern.ch/cern-ohl) or later.
