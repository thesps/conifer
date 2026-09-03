from conifer.backends.vhdl.writer import make_model, auto_config
from conifer.backends.vhdl.simulators import Modelsim, GHDL, Xsim, get_simulator, set_simulator
import logging
logger = logging.getLogger(__name__)


def __getattr__(name):
  # Detect the simulator lazily on first access, not on `import conifer`.
  if name == 'simulator':
    return get_simulator()
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
