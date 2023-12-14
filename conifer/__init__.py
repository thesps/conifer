from __future__ import absolute_import
from importlib.metadata import version, PackageNotFoundError
from packaging.version import Version

# TODO: Make it possible to run conifer from a git clone without having conifer installed.
try:
    __version__ = Version(version("conifer"))
except PackageNotFoundError:
    # package is not installed
    pass

from conifer import converters
from conifer import backends
from conifer import utils
