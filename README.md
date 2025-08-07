<img src="https://github.com/thesps/conifer/raw/master/conifer_v1.png" width="250" alt="conifer">

Conifer translates trained Boosted Decision Trees to FPGA firmware for extreme low latency inference. 

# Installation
Conifer is on the Python Package Index, so install it like:
```
pip install conifer
```

Building FPGA firmware requires tools from Xilinx - a High Level Synthesis tool and Vivado, depending on the choice of backend. We recommend a recent version Vivado ML and Vitis HLS. The latest validate tool version is 2024.1. Modelsim and GHDL are additionally supported for VHDL backend simulation.

Using the C++ backends requires JSON header files from [here](https://github.com/nlohmann/json). Clone or download the repository, and set the environment variable `JSON_ROOT` to the JSON include path. Xilinx's arbitrary precision header files (e.g. `ap_fixed.h`) are required for the C++, Xilinx HLS, and VHDL backends. They are automatically found if you have sourced the Xilinx HLS toolchain, but you can also find them [here](https://github.com/Xilinx/HLS_arbitrary_Precision_Types). If using the C++ backend without a Vivado installation, clone or download Xilinx's repository, and set the environment variable `XILINX_AP_INCLUDE` to the include path.

# About
Conifer converts from popular BDT training frameworks, and can emit code projects in different FPGA languages.


Available converters:
- `scikit-learn`
- `xgboost`
- `ONNX` - giving access to other training libraries such as lightGBM and CatBoost with ONNXMLTools
- `TMVA`
- `ydf`

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
import conifer

# enable more verbose output from conifer
import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.WARNING)
logger = logging.getLogger('conifer')
logger.setLevel(logging.DEBUG)

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

# Development
1. Clone the github repository: `git clone https://github.com/thesps/conifer`
1. Install the dependencies listed in the *Installation* section. For instance:
    - Clone nlohmann json: `git clone https://github.com/nlohmann/json`
    - Clone Arbitrary Precision: `git clone https://github.com/Xilinx/HLS_arbitrary_Precision_Types`
    - Set the env variables `JSON_ROOT` and `XILINX_AP_INCLUDE` as the `include` directory in them respectively e.g.
    ```shell
    export XILINX_AP_INCLUDE=$(pwd)/HLS_arbitrary_Precision_Types/include
    export JSON_ROOT=$(pwd)/json/include
    ```
1. Go into the Conifer project directory: `cd conifer`
1. Install the python development dependencies: `pip install -r dev_requirements.txt`
1. Run an example: `export PYTHONPATH="$(pwd):${PYTHONPATH}" && python examples/sklearn_to_cpp.py`
1. Run a single unit test: `pytest tests/test_multiclass.py`
1. Run all the unit tests: `pytest`

# Use Cases

The following are some applications using `conifer` either in production or R&D. If you think your own use case should be featured here, please get in touch.

## ATLAS Calorimeter Tau Trigger

The ATLAS experiment at CERN's Large Hadron Collider is currently using `conifer` to better reconstruct events with tau leptons. See [this talk](https://indico.cern.ch/event/1283970/contributions/5554387/) at the Fast Machine Learning 2023 workshop.

- Domain: particle physics
- Frontend: `xgboost`
- Backend: `Vitis HLS`
- Target Device: AMD Virtex 7 FPGA

## Tracking detector frontend data reduction

J. Gonski _et al_, Embedded FPGA Developments in 130nm and 28nm CMOS for Machine Learning in Particle Detector Readout, 2024, [arXiv:2404.17701v1](https://arxiv.org/pdf/2404.17701)

A team from Stanford and SLAC have shown that a tiny Decision Tree model running in an eFPGA at the periphery of a particle tracking detector frontend ASIC can be used to filter data from pile-up events.

- Domain: particle physics
- Frontend: `scikit-learn`
- Backend: `HLS`
- Target Device: eFPGA from [FABulous](https://doi.org/10.1145/3431920.3439302)

## Enhancing blood vessel segmentation

Alsharari M _et al_. Efficient Implementation of AI Algorithms on an FPGA-based System for Enhancing Blood Vessel Segmentation.  [doi:10.21203/rs.3.rs-4351485/v1](https://doi.org/10.21203/rs.3.rs-4351485/v1).

The authors demonstrate methods to perform performing image segmentation to identify blood vessels in a proposed surgical imaging device. A GBDT, deployed to a Kria SoM with `conifer`, achieved the best FPS with segmentation performance competitive with U-Net models.

- Domain: medical devices
- Frontend: `ONNX` (`catboost`)
- Backend: `Vitis HLS`
- Target Device: AMD Kria KV260


# License
Apart from the source code and binaries of the Forest Processing Unit (FPU), `conifer` is licensed under *Apache v2*. The FPU source code and binaries are licensed under the [*CERN-OHL-P v2*](https://cern.ch/cern-ohl) or later.
