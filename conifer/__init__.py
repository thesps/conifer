from __future__ import absolute_import
from importlib.metadata import version, PackageNotFoundError
from packaging.version import Version

try:
    __version__ = Version(version("conifer"))
except (PackageNotFoundError, ImportError):
    # Conifer package is not installed
    __version__ = Version("0.0.0")

from conifer import converters
from conifer import backends
from conifer import utils
