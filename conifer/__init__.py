from __future__ import absolute_import
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("conifer")
except PackageNotFoundError:
    # package is not installed
    pass

from conifer import converters
from conifer import backends
from conifer.model import Model