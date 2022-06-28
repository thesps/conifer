import sys
import importlib.util

from . import xilinxhls

SPEC_XILINXHLS = importlib.util.find_spec('.xilinxhls', __name__)

vivadohls = importlib.util.module_from_spec(SPEC_XILINXHLS)
vivadohls._tool = 'vivadohls'
SPEC_XILINXHLS.loader.exec_module(vivadohls)

vitishls = importlib.util.module_from_spec(SPEC_XILINXHLS)
vitishls._tool = 'vitishls'
SPEC_XILINXHLS.loader.exec_module(vitishls)

del SPEC_XILINXHLS

from . import vhdl
from . import cpp
