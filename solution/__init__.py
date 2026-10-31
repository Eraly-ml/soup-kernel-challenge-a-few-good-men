"""Our solution: a training step with a reverse sweep by hand, on Triton kernels, with the frozen
weights in host memory. See README.md.

The harness imports this package and calls `apply()` once, in the same process as `soup train`,
before Soup builds the model and the trainer.
"""

from .adjoint import patch_kbit_preparation, patch_training_step
from .startup import prepare
from .stream import patch_loading


def apply():
    prepare()
    patch_kbit_preparation()
    patch_loading()
    patch_training_step()
