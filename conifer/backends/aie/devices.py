import json
import logging
import os

logger = logging.getLogger(__name__)

_DEVICE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "devices")


def get_available_devices():
    return sorted(f[:-5] for f in os.listdir(_DEVICE_DIR) if f.endswith(".json"))


def _load(name):
    with open(os.path.join(_DEVICE_DIR, f"{name}.json")) as f:
        return json.load(f)


def get_device_config(name_or_part):
    """Get an AI Engine device record by device name or Xilinx part"""
    devices = {n: _load(n) for n in get_available_devices()}
    for name, d in devices.items():
        if name_or_part in (name, d["part"]):
            return d
    # A part differing only in speed grade or package still names the same array.
    stem = str(name_or_part).split("-")[0]
    for name, d in devices.items():
        if d["part"].split("-")[0] == stem:
            logger.info(f"part {name_or_part} matched device {name} ({d['part']})")
            return d
    known = ", ".join(f"{n} ({d['part']})" for n, d in devices.items())
    raise ValueError(
        f'No AI Engine device "{name_or_part}" found. Known devices: {known}'
    )
