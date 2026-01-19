# try loading runtime modules for pynq and xrt

import logging
logger = logging.getLogger(__name__)
try:
    from conifer.backends.xilinxhls.runtime.pynq import ZynqDriver, PynqAlveoDriver
except ImportError:
    ZynqDriver = None
    PynqAlveoDriver = None
    logger.warning('pynq runtime module could not be imported. Interacting with PYNQ-based accelerators will not be possible.')

try:
    from conifer.backends.xilinxhls.runtime.xrt import XrtDriver
except ImportError:
    XrtDriver = None
    logger.warning('xrt runtime module could not be imported. Interacting with XRT-based accelerators will not be possible.')