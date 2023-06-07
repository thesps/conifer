'''
Build a conifer Forest Processing Unit (FPU)
You can specify the target device (default pynq-z2), and parameters of the FPU
Once the FPU is built, you can use it to run inference for multiple BDTs without rebuilding
Note: this is work in progress and probably buggy
'''

from conifer.backends.fpu import AlveoFPUBuilder
import datetime
import sys
import logging
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

# Create an FPUBuilder config
cfg = AlveoFPUBuilder.default_cfg()
cfg['part'] = 'xcu200-fsgd2104-2-e'
cfg['platform'] = 'xilinx_u200_gen3x16_xdma_2_202110_1'
cfg['tree_engines'] = 100
cfg['nodes'] = 512
cfg['dynamic_scaler'] = True

# Set the output directory to some code name
cfg['output_dir'] = 'fpu-0A-u200_100TE_512N_DS'

builder = AlveoFPUBuilder(cfg)

builder.build(csynth=True, bitfile=True)
builder.package()