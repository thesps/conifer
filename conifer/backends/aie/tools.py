import datetime
import logging
import os
import re
import subprocess

logger = logging.getLogger(__name__)

# AIE tools that a project needs
_AIE_TOOLS = {
    "aiecompiler": "aiecompiler",
    "x86simulator": "x86simulator",
    "aiesimulator": "aiesimulator",
}


def get_tool_exe_in_path(tool):

    if tool not in _AIE_TOOLS:
        return None

    tool_exe = _AIE_TOOLS[tool]

    if os.system(f"type {tool_exe} > /dev/null 2>/dev/null") != 0:
        return None

    return tool_exe


def require_tools(*tools):
    missing = [t for t in tools if get_tool_exe_in_path(t) is None]
    if missing:
        raise RuntimeError(
            f"Could not find {', '.join(missing)} on the path. Source the Vitis settings "
            f"script (settings64.sh) for a release with AI Engine support"
        )


# The tools finish with a log like "(WARNING:3, CRITICAL-WARNING:0, ERROR:0)"
_LOG_LINE = re.compile(r"CRITICAL-WARNING|ERROR:\s*0\b")


def _first_error(log_path):
    try:
        with open(log_path, errors="ignore") as f:
            for line in f:
                if ("ERROR" in line or "error:" in line) and not _LOG_LINE.search(line):
                    return line.strip()[:300]
    except OSError:
        pass
    return None


def run_make(output_dir, target, **variables):
    """Run one target of the project Makefile, capturing its output to a log"""
    tools = {
        "x86sim_build": ("aiecompiler",),
        "x86sim": ("aiecompiler", "x86simulator"),
        "hw_build": ("aiecompiler",),
        "aiesim": ("aiecompiler", "aiesimulator"),
    }.get(target, ())
    require_tools(*tools)
    args = " ".join(f"{k}='{v}'" for k, v in variables.items() if v is not None)
    log_path = os.path.join(output_dir, f"{target}.log")
    cmd = f"make -C {output_dir} {target} {args}".strip()
    logger.debug(f'Running build with command "{cmd}"')

    start = datetime.datetime.now()
    logger.info(f"{target} starting {start:%H:%M:%S}")
    with open(log_path, "w") as log:
        rc = subprocess.call(cmd, shell=True, stdout=log, stderr=subprocess.STDOUT)
    stop = datetime.datetime.now()
    logger.info(
        f"{target} finished {stop:%H:%M:%S} - took {str(stop - start)}, "
        f"log in {log_path}"
    )

    if rc != 0:
        error = _first_error(log_path)
        logger.error(
            f"{target} failed, check the log in {log_path}"
            + (f". First error: {error}" if error else "")
        )
        return False
    return True
