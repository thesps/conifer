from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension
from glob import glob
import os

XRT = os.environ.get("XILINX_XRT")
if XRT is None:
    raise RuntimeError(
        "XILINX_XRT environment variable must be set to build conifer-xrt"
    )

ext_modules = [
    Pybind11Extension(
        "conifer_xrt_runtime",
        sorted(glob("src/*.cpp")),
        include_dirs=[XRT + "/include", "../external/json/include"],
        extra_compile_args=["-std=c++17"],
        extra_objects=[XRT + "/lib/libxrt_coreutil.so"],
    )
]

setup(ext_modules=ext_modules)
