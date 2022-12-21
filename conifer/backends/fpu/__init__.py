import logging
logger = logging.getLogger(__name__)

try:
    from conifer.backends.fpu.pynq_driver import FPUDriver
except ImportError:
    FPUDriver = None
    logger.warn('FPUDriver could not be imported. Interacting with FPUs will not be possible.')
from conifer.backends.fpu.writer import make_model, FPUModel, auto_config, FPUBuilder