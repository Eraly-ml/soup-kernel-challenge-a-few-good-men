"""The whole reverse sweep against autograd, where test_kernels.py checks it one kernel at a time.

Runs on a CPU: see sweep_check.py, and why it needs a process of its own. A GPU stays hidden from
that process, so that the libraries it imports do not go looking for the real Triton.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("mode", ["fp32", "fp16"])
def test_sweep_matches_autograd(mode):
    check = Path(__file__).with_name("sweep_check.py")
    done = subprocess.run([sys.executable, str(check), mode], capture_output=True, text=True,
                          timeout=300, env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
    assert done.returncode == 0, done.stdout + done.stderr
