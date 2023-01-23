from conifer.backends.vhdl.writer import make_model, auto_config
from conifer.backends.vhdl.simulators import Modelsim, GHDL, Xsim
simulator = Xsim
import logging
logger = logging.getLogger(__name__)
for sim in [Xsim, GHDL, Modelsim]:
  from conifer.backends.vhdl.simulators import _touch
  if _touch(sim):
    logger.info(f'Found {sim.__name__}, setting VHDL simulator to {sim.__name__}')
    simulator = sim
    break
