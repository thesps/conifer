from conifer.backends.vhdl.writer import make_model, auto_config
from conifer.backends.vhdl.simulators import Modelsim, GHDL, Xsim, get_simulator
import logging
logger = logging.getLogger(__name__)


def __getattr__(name):
  # Detect the VHDL simulator lazily: accessing `conifer.backends.vhdl.simulator`
  # triggers detection (which shells out to EDA tools) the first time it's needed,
  # rather than on `import conifer`. Result is cached in get_simulator().
  if name == 'simulator':
    return get_simulator()
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
