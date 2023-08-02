import logging
logger = logging.getLogger(__name__)

try:
    from conifer.backends.fpu import runtime
except ImportError:
    ZynqDriver = None
    logger.warn('runtime module could not be imported. Interacting with FPUs will not be possible.')
from conifer.backends.fpu.writer import make_model, FPUModel, auto_config, FPUBuilder, FPUConfig