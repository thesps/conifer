import sys
import importlib.util

SPEC_WRITER = importlib.util.find_spec('.writer', __name__)

writer = importlib.util.module_from_spec(SPEC_WRITER)
if '_tool' in locals() != None:
    writer._tool = _tool
SPEC_WRITER.loader.exec_module(writer)

make_model = writer.make_model
auto_config = writer.auto_config

del SPEC_WRITER

try:
    from conifer.backends.xilinxhls import runtime
except ImportError:
    ZynqDriver = None
    logger.warn('runtime module could not be imported. Interacting with accelerators will not be possible.')
