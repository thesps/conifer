import glob
import logging
import os

logger = logging.getLogger(__name__)

# Where a Vitis install keeps its base platforms
_PLATFORM_SUBDIRS = ("base_platforms", "platforms")


def _roots():
    """Directories that may hold platform repositories, most specific first"""
    roots = []
    for var in ("PLATFORM_REPO_PATHS", "XILINX_VITIS", "XILINX_HLS"):
        for p in os.environ.get(var, "").split(os.pathsep):
            if p and p not in roots:
                roots.append(p)
    return roots


def platform_search_paths():
    """Every directory a platform could live in, derived from the Vitis environment"""
    paths = []
    for root in _roots():
        if not os.path.isdir(root):
            continue
        for d in (root,) + tuple(os.path.join(root, s) for s in _PLATFORM_SUBDIRS):
            if os.path.isdir(d) and d not in paths:
                paths.append(d)
    return paths


def find_platform(name):
    """Absolute path of a platform's .xpfm, or None
    e.g. 'vek280_base', 'xilinx_vek280_base_202610_1'
    """
    if name and name.endswith(".xpfm"):
        return os.path.abspath(name) if os.path.exists(name) else None

    hits = []
    for base in platform_search_paths():
        hits.extend(glob.glob(os.path.join(base, f"{name}.xpfm")))
        hits.extend(glob.glob(os.path.join(base, name, f"{name}.xpfm")))
        hits.extend(glob.glob(os.path.join(base, f"*{name}*", "*.xpfm")))
    if not hits:
        return None
    exact = [h for h in hits if os.path.basename(h) == f"{name}.xpfm"]
    return max(exact or hits)


def resolve_platform(name):
    """Locate a platform, raising with what was searched if it is not there"""
    found = find_platform(name)
    if found:
        return found
    searched = platform_search_paths()
    if not searched:
        raise RuntimeError(
            f'Cannot locate the "{name}" platform: no Vitis environment found. Source '
            f"settings64.sh, or set PLATFORM_REPO_PATHS to a platform repository, or set "
            f"the Platform config field to an .xpfm path"
        )
    available = sorted(
        {
            os.path.basename(p)[: -len(".xpfm")]
            for d in searched
            for p in glob.glob(os.path.join(d, "*", "*.xpfm"))
        }
    )
    raise RuntimeError(
        f'Cannot locate the "{name}" platform. Searched: {", ".join(searched)}. '
        + (
            f"Available: {', '.join(available)}"
            if available
            else "No platforms found there"
        )
    )
