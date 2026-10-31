"""The low-toggle rounding of the backward: of the 10 mantissa bits of fp16, an operand or a weight of
a frozen matmul keeps 4. Fewer bits that toggle in the multipliers are more MACs per joule at the 70 W
cap of a T4. The forward never sees these values, and no kernel reads them again.

`(bits + 32) & -64` alone would round to nearest, and it would do three more things, found by going
through all 65,536 patterns, the last one measured on the T4:
- it would carry the 64 largest finite numbers into inf. They stay as they are here;
- it would carry 62 of the patterns of NaN into inf and turn 64 into zero, by overflowing the integer.
  A zero would hide from GradScaler a step that base Soup skips. A NaN stays as it is here, and so
  does inf;
- ties going up, the error would average +0.5 of a unit, in operands and weights alike, and the logged
  grad_norm would read +0.81% on hidden-style rows and +0.38% on default rows. Ties go to even here,
  and it reads -0.21% and -0.25%, at the same cosine against autograd.
"""

import triton
import triton.language as tl


@triton.jit
def rough(value):
    """An fp16 tensor rounded to 4 mantissa bits, for one matmul of the backward."""
    bits = value.to(tl.int16, bitcast=True)
    near = (bits + 31 + ((bits >> 6) & 1)) & -64
    return tl.where((bits & 0x7FFF) < 0x7BE0, near, bits).to(tl.float16, bitcast=True)
