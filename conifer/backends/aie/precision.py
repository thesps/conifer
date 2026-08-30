import logging
import re

import numpy as np

logger = logging.getLogger(__name__)

# The compare path is int16 throughout the kernels and the accumulator is acc32,
# the only fully tested widths for now
SUPPORTED_WIDTHS = (8, 16, 32)
COMPARE_WIDTH = 16
SCORE_WIDTH = 32

_C_TYPE = {8: "int8_t", 16: "int16_t", 32: "int32_t"}

REQUIRED_ROUNDING = "AP_RND_CONV"
REQUIRED_OVERFLOW = "AP_SAT"

_AP_FIXED = re.compile(r"^\s*ap_(u?)fixed\s*<([^>]*)>\s*$")

# TODO: quantize() add support for other types
class Precision:
    """One conifer ap_fixed type, as the kernels need it: a width and a binary point"""

    # TODO: improve by taking W,I directly instead of ap_fixed?
    def __init__(self, type_string):
        self.type_string = type_string
        m = _AP_FIXED.match(type_string)
        if m is None:
            raise ValueError(
                f'The aie backend needs an ap_fixed precision, got "{type_string}"'
            )
        self.signed = m.group(1) == ""
        args = [a.strip() for a in m.group(2).split(",")]
        if len(args) not in (2, 4):
            raise ValueError(
                f'Could not parse "{type_string}": expected ap_fixed<W,I> or '
                "ap_fixed<W,I,Q,O>"
            )
        try:
            self.width = int(args[0])
            self.integer_bits = int(args[1])
        except ValueError:
            raise ValueError(
                f'Could not parse the width and integer bits of "{type_string}"'
            )
        self.rounding = args[2] if len(args) == 4 else "AP_TRN"
        self.overflow = args[3] if len(args) == 4 else "AP_WRAP"
        self.shift = self.width - self.integer_bits

    def validate(self, role, allowed=SUPPORTED_WIDTHS):
        if not self.signed:
            raise ValueError(
                f"{role} precision {self.type_string}: the aie backend needs a "
                "signed type"
            )
        if self.width not in allowed:
            want = allowed[0] if len(allowed) == 1 else None
            detail = (
                f"needs width {want}"
                if want
                else f"supports widths {allowed}, not {self.width}"
            )
            nearest = min(allowed, key=lambda w: (abs(w - self.width), w))
            ibits = min(self.integer_bits, nearest)
            raise ValueError(
                f"{role} precision {self.type_string}: the aie backend "
                f"{detail}. Use "
                f"ap_fixed<{nearest},{ibits},{REQUIRED_ROUNDING},"
                f"{REQUIRED_OVERFLOW}>. Only the total width is fixed: the "
                f"integer bits are free."
            )
        if self.shift < 0:
            raise ValueError(
                f"{role} precision {self.type_string}: more integer bits than "
                "total width"
            )
        if (self.rounding, self.overflow) != (REQUIRED_ROUNDING, REQUIRED_OVERFLOW):
            raise ValueError(
                f"{role} precision {self.type_string}: the aie backend needs "
                f"ap_fixed<{self.width},{self.integer_bits},{REQUIRED_ROUNDING},"
                f"{REQUIRED_OVERFLOW}>. The tables are quantized round-half-to-even and "
                f"saturating"
            )

    @property
    def n_bytes(self):
        return self.width // 8

    @property
    def c_type(self):
        return _C_TYPE[self.width]

    @property
    def max_representable(self):
        return (2 ** (self.width - 1) - 1) / (1 << self.shift)

    def quantize(self, x):
        """Float to the integer grid, round-to-even then saturate"""
        lo, hi = -(2 ** (self.width - 1)), 2 ** (self.width - 1) - 1
        q = np.rint(np.asarray(x, dtype=np.float64) * (1 << self.shift))
        return np.clip(q, lo, hi).astype(np.int64)

    def dequantize(self, q):
        return np.asarray(q, dtype=np.float64) / (1 << self.shift)

    def saturates(self, x):
        """Whether any value would clip on this grid"""
        v = np.abs(np.asarray(x, dtype=np.float64))
        return bool(v.size and v.max() > self.max_representable)

    def __repr__(self):
        return f"Precision({self.type_string})"
