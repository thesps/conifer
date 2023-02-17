Forest Processing Unit
########################

The conifer Forest Processing Unit (FPU) is a flexible, fast architecture for BDT inference on FPGAs.
The key difference of the FPU to the other conifer backends like HLS and VHDL, is that one FPU bitfile can perform inference for many BDTs, reconfigurable at runtime.

Quickstart
**********

To get started using FPU quickly, first download an FPU bitfile for your device from the conifer website downloads area.
This example shows how to load and use the FPU for an Alveo, and the process is similar for Zynq devices.
Using Alveo, source the Xilinx RunTime environment first.

.. code-block:: python
  :caption: conifer FPU quickstart

  import conifer
  device = conifer.backends.fpu.runtime.AlveoDriver('fpu.xclbin') # load FPU onto Alveo
  model = ... # load or train a BDT with any packages with a conifer frontend
  config = conifer.backends.fpu.auto_config()
  config['FPU'] = device.config
  conifer_model = conifer.converters.convert_from_<frontend>(model, config)
  conifer_model.attach_device(device, batch_size=1) # load model onto FPU, allocate buffers
  y = conifer_model.decision_function(X) 


Architecture Description
************************

Tree Engine
===========

The building block of the FPU is the 'Tree Engine' (TE).
Each TE has a local memory, implemented in FPGA Block RAM, storing tree nodes. 
Logic to loop over the nodes, performing comparisons between input features and thresholds, is implemented in the TE.
One FPU comprises many (hundreds) of TEs, each of which operate independently and in parallel.
Aggregation logic sums the TE outputs to compute the final ensemble prediction.

.. _Dynamic Scaler:

Dynamic Scaler
==============

The data types for thresholds and scores is configurable at build time, though integer types are preferred.
In order to have flexibility for different BDT models in the same architecture, a scaling unit to convert from floating point inputs to integers is optionally implemented internally.
When converting a BDT targeting an FPU with ``DynamicScaler`` enabled, scale factors for each feature and class are derived from a static analysis of the model.


Interface
=========

The FPU provides AXI Master interfaces for: nodes (one port each for load/read); scales factors (one port each for load/read); input features, output predictions, and the configuration used to build the FPU.
AXI Slave interfaces are provided for configuration registers: instruction (info string length, load, read, predict); batch size; number of features, length of information string.

Runtime
=======

A runtime interface is provided as ``conifer.backends.fpu.runtime`` implementing classes ``AlveoDriver`` and ``ZynqDriver``.
The runtime uses Xilinx's ``pynq`` package, including XRT for Alveo platforms.

Building FPUs
*************

The FPU backend provides builder classes to build FPUs with different configurations or different devices.
``conifer.backends.fpu`` provides ``AlveoFPUBuilder`` and ``ZynqFPUBuilder``.
Configuration is provided through a dictionary, as for other conifer backends. 
Get a starting template with default parameters like ``config = conifer.backends.fpu.AlveoFPUBuilder.default_cfg()``.
Modify the configuration, create a builder instance ``builder = conifer.backends.fpu.AlveoFPUBuilder(config)``, and build with ``builder.build(csynth=True, bitfile=True)``.

Configuration options available for all platforms:
==================================================

- ``output_dir`` : string, project output directory
- ``project_name`` : string, project name
- ``part`` : string, FPGA part to target
- ``tree_engines`` : integer, number of Tree Engines
- ``nodes`` : integer, number of nodes per TE
- ``features`` : integer, maximum number of model features
- ``threshold_type`` : integer, number of bits for thresholds
- ``score_type`` : integer, number of bits for scores
- ``dynamic_scaler`` : boolean, include/exclude float-to-int dynamic scaler (described in :ref:`Dynamic Scaler <Dynamic Scaler>`)

Platform specific options:
==========================
Alveo:

- ``clock_period`` : integer, FPU kernel target clock period
- ``platform`` : string, Alveo platform to target (e.g. ``"xilinx_u200_gen3x16_xdma_2_202110_1"``)

Zynq:

- ``board_part`` : string, Vivado IPI board name (e.g. ``"tul.com.tw:pynq-z2:part0:1.0"``)
- ``processing_system_ip`` : string, Vivado IPI IP name of PS (e.g. ``"xilinx.com:ip:processing_system7:5.5"``)
- ``processing_system`` : string, Vivado IPI PS version (e.g. ``"processing_system7"``)