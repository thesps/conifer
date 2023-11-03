import logging
logger = logging.getLogger(__name__)

from conifer.backends.xilinxhls.writer import make_model, auto_config
try:
    from conifer.backends.xilinxhls import runtime
except ImportError:
    ZynqDriver = None
    AlveoDriver = None
    logger.warn('runtime module could not be imported. Interacting with accelerators will not be possible.')
