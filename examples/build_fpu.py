from conifer.backends.fpu import FPUBuilder
import datetime
import sys
import logging
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

# Create an FPUBuilder config
cfg = FPUBuilder.default_cfg()
cfg['tree_engines'] = 2
cfg['nodes'] = 16
cfg['dynamic_scaler'] = False

stamp = int(datetime.datetime.now().timestamp())
# Set the output directory to some code name
cfg['output_dir'] = 'fpu-07_2TE16N_NDS'

builder = FPUBuilder(cfg)

builder.build()