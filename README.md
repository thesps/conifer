<img src="conifer_v1.png" width="250" alt="conifer">

Conifer translates trained Boosted Decision Trees to FPGA firmware for extreme low latency inference. Check the examples directory for examples to get started with.

Currently models from `sklearn`, `xgboost`, and `TMVA` are supported. FPGA firmware can be produced in Xilinx Vivado HLS or VHDL.

See our paper: [https://arxiv.org/abs/2002.02534](https://arxiv.org/abs/2002.02534)

Conifer originated as a development for [https://hls-fpga-machine-learning.github.io/hls4ml/](hls4ml), and borrows heavily from the code and ideas developed for it.

# Installation
```
git clone https://github.com/thesps/conifer.git
cd conifer
pip install .
```
