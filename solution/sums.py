"""Frozen matmuls of the backward and the head's gradient, with cuBLAS adding up in fp16.

By default cuBLAS adds up an fp16 matmul in fp32. Under the 70 W cap of a T4 the multiplies are what
the card runs out of, and with `allow_fp16_accumulation` the same shapes cost fewer joules: on a
sustained bench the backward ran about 7% faster and the head about 5%, the forward no faster, so
the forward stays as it is. In alternating full runs these and the head's logits gave about 0.4 s of
a 27 s loop at equal temperature, of which the logits were about 0.19 s. The logits stay as they are
all the same, so that everything Soup logs is computed as before. The price is rounding, in the
gradients alone: 0.15-0.37% rms per matmul on random data, against 0.02-0.04%. And a running sum
above 65,504, the largest fp16, goes to inf where cuBLAS's default would not: over all 64 steps of
hidden-style rows no running sum could pass 14,699.

The setting is global to the process, so it holds for one call and goes back at once: the original
training step, Soup's tests and the LoRA matmuls never see it.
"""

import torch

BLAS = torch.backends.cuda.matmul


def summed_in_fp16(a, b, out=None):
    """a @ b, with this one matmul adding up in fp16."""
    before = BLAS.allow_fp16_accumulation
    BLAS.allow_fp16_accumulation = True
    try:
        return torch.matmul(a, b, out=out)
    finally:
        BLAS.allow_fp16_accumulation = before
