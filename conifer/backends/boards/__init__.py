from conifer.backends.boards.boards import BoardConfig, AlveoConfig, ZynqConfig, AlveoBuilder, ZynqBuilder
import logging
logger = logging.getLogger(__name__)
import os
filedir = os.path.dirname(os.path.abspath(__file__))

zynq_files = [f'{filedir}/configs/zynq/{f}' for f in os.listdir(f'{filedir}/configs/zynq') if '.json' in f]
alveo_files = [f'{filedir}/configs/alveo/{f}' for f in os.listdir(f'{filedir}/configs/alveo') if '.json' in f]
__loaders = [(zynq_files, ZynqConfig),
             (alveo_files, AlveoConfig),
             ]

__configs = {}

for loader in __loaders:
  for cfg_file in loader[0]:
      builder = loader[1].load(cfg_file)
      name = getattr(builder, 'name', None)
      if name is not None:
        __configs[name] = builder

def get_available_boards():
  return list(__configs.keys())

def get_board_config(name):
  config = __configs.get(name, None)
  if config is None:
    logger.warn(f'Could not find board "{name}", check get_available_boards')
  return config

__builders = {ZynqConfig  : ZynqBuilder,
              AlveoConfig : AlveoBuilder,
              }

def get_builder(project_config, board_config, **kwargs):
  builder = __builders.get(type(board_config), None)
  assert builder is not None, f'Could not find builder for {type(board_config)} from {list(__builders.keys())}'
  return builder(project_config, board_config, **kwargs)