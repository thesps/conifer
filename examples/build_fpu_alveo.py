'''
Build a conifer Forest Processing Unit (FPU)
You can specify the target device (default pynq-z2), and parameters of the FPU
Once the FPU is built, you can use it to run inference for multiple BDTs without rebuilding
Note: this is work in progress and probably buggy
'''

from conifer.backends.fpu import FPUBuilder
import sys
import logging
logging.basicConfig(stream=sys.stdout, level=logging.WARNING)
logger = logging.getLogger('conifer')
logger.setLevel(logging.DEBUG)

# Create an FPUBuilder config
cfg = FPUBuilder.default_cfg()
cfg['board'] = 'xilinx_u200_gen3x16_xdma_2_202110_1'
cfg['tree_engines'] = 100
cfg['nodes'] = 512
cfg['dynamic_scaler'] = True

# Set the output directory to some code name
cfg['output_dir'] = 'fpu-0B-u200_100TE_512N_DS'

builder = FPUBuilder(cfg)

builder.build(csynth=True, bitfile=True)
builder.package()