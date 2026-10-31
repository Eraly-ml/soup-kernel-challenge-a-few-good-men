"""Fused RMSNorm for Qwen2, forward and backward in Triton.

Qwen2RMSNorm runs six PyTorch ops per call (pow, mean, add, rsqrt, two multiplies). Here one
launch does all of it, and the backward needs only the input: with u = grad * weight and
r = rsqrt(mean(x^2) + eps), the gradient is r * u - x * r^3 * (u.x) / H. These two kernels do the
norm at the end of the model; the norms inside a layer are fused with their neighbours in fused.py.

Idea of a fused RMSNorm kernel: Liger Kernel. The code here is ours.
"""

import torch
import triton
import triton.language as tl

from .startup import precompile

BLOCK = 2048  # the smallest power of two that holds a hidden vector of 1536


@triton.jit
def rms_norm_fwd_kernel(x_ptr, weight_ptr, y_ptr, eps, HIDDEN: tl.constexpr, BLOCK: tl.constexpr):
    """One program per token: y = weight * x * rsqrt(mean(x^2) + eps)."""
    start = tl.program_id(0) * HIDDEN
    offsets = tl.arange(0, BLOCK)
    mask = offsets < HIDDEN
    x = tl.load(x_ptr + start + offsets, mask=mask, other=0.0)
    weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0)
    scale = tl.rsqrt(tl.sum(x * x, axis=0) / HIDDEN + eps)
    tl.store(y_ptr + start + offsets, weight * x * scale, mask=mask)


@triton.jit
def rms_norm_bwd_kernel(grad_ptr, x_ptr, weight_ptr, dx_ptr, eps, HIDDEN: tl.constexpr,
                        BLOCK: tl.constexpr):
    """One program per token: the vector-Jacobian product of the forward kernel."""
    start = tl.program_id(0) * HIDDEN
    offsets = tl.arange(0, BLOCK)
    mask = offsets < HIDDEN
    grad = tl.load(grad_ptr + start + offsets, mask=mask, other=0.0)
    x = tl.load(x_ptr + start + offsets, mask=mask, other=0.0)
    weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0)
    scale = tl.rsqrt(tl.sum(x * x, axis=0) / HIDDEN + eps)
    pulled = grad * weight
    along = tl.sum(pulled * x, axis=0) * scale * scale / HIDDEN
    tl.store(dx_ptr + start + offsets, scale * (pulled - x * along), mask=mask)


precompile(rms_norm_fwd_kernel, torch.float32, torch.float32, torch.float32, 1e-6, HIDDEN=1536,
           BLOCK=BLOCK)
precompile(rms_norm_bwd_kernel, torch.float32, torch.float32, torch.float32, torch.float32, 1e-6,
           HIDDEN=1536, BLOCK=BLOCK)


def rms_norm(x, weight, eps):
    """The norm of a [tokens, hidden] fp32 tensor."""
    y = torch.empty_like(x)
    rms_norm_fwd_kernel[(x.shape[0],)](x, weight, y, eps, HIDDEN=x.shape[1], BLOCK=BLOCK)
    return y


def rms_norm_adjoint(grad, x, weight, eps):
    """The gradient over x; the weight is frozen, so it gets none."""
    dx = torch.empty_like(x)
    rms_norm_bwd_kernel[(x.shape[0],)](grad, x, weight, dx, eps, HIDDEN=x.shape[1], BLOCK=BLOCK)
    return dx
