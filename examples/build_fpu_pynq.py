'''
Build a conifer Forest Processing Unit (FPU)
You can specify the target device (default pynq-z2), and parameters of the FPU
Once the FPU is built, you can use it to run inference for multiple BDTs without rebuilding
Note: this is work in progress and probably buggy
'''

from conifer.backends.fpu import FPUConfig, FPUBuilder
from conifer.backends.boards import get_board_config
import datetime
import sys
import logging
logging.basicConfig(stream=sys.stdout, level=logging.WARNING)
logger = logging.getLogger('conifer')
logger.setLevel(logging.DEBUG)

# Create an FPUBuilder config
cfg = FPUBuilder.default_cfg()
cfg['tree_engines'] = 100
cfg['nodes'] = 512
cfg['dynamic_scaler'] = True
cfg['board'] = 'pynq-z2'

stamp = int(datetime.datetime.now().timestamp())
# Set the output directory to some code name
cfg['output_dir'] = 'fpu-0B-pynq-z2_100TE_512N_DS'

builder = FPUBuilder(cfg)

builder.build()
builder.package()