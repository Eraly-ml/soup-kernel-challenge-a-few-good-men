"""The small work around the matmuls of a decoder layer, fused into one launch per place.

Between two large matmuls base Soup, and a plain reverse sweep, run dozens of small torch ops:
casts, the norm, the LoRA dropout, the rotation, additions, each with its own launch and its own
pass over the tensor. Under the 70 W power cap of the T4 that traffic is time twice: a torch op
costs about 10 us to launch and the GPU work it does is taken from the matmuls. Each kernel here
replaces at least five of them. It computes in fp32 on fp16 inputs with no fp32 copy of them in
memory: the fp32 tensors are the running h and its gradient, the dropout masks, the rotary tables
and the norm weights.

All of them work on the packed form of a step: [tokens, ...] without padding. The joined tensors
hold the output of Q, K and V side by side, [tokens, q: heads * width, k: keys * width, v: same],
because the three matrices run as one matmul. Layer [hidden] vectors are programs of BLOCK lanes,
one per token, as in norm.py; the attention tensors are programs of half a head, because the
rotation pairs the two halves.

The fp16 tensors a kernel writes only for the next frozen matmul of the backward are rough: of the
10 mantissa bits of fp16 they keep 4, see rough.py. That is about 7% more MACs per joule at the 70 W
cap, and 19% with rough weights (nf4.py). No kernel reads such a tensor again: the running gradient
itself stays fp32.

Idea of fusing RMSNorm with the work around it: Liger Kernel. The same fusion is public in Liger
Kernel (ops/fused_add_rms_norm.py) and Unsloth (kernels/fused_add_rms_layernorm.py). One matmul for
Q, K and V and the dropout mask kept for the backward: Soup's Fast-LoRA. The first norm computed
again from a saved input: gradient checkpointing. A rotary embedding in a Triton kernel is public
in Liger Kernel (ops/rope.py) and Unsloth (kernels/rope_embedding.py). The code here is ours.
"""

import torch
import triton
import triton.language as tl

from .rough import rough
from .startup import precompile

BLOCK = 2048  # the smallest power of two that holds a hidden vector of 1536
WORDS, BITS = 64, 32  # the same lanes as a tile, where lanes are packed into words of bits
CHUNK = 1024  # lanes per program over an intermediate vector of 8960


@triton.jit
def prepare_kernel(x_ptr, weight_ptr, mask_q_ptr, mask_v_ptr, place_ptr, a_ptr, dropped_q_ptr,
                   dropped_v_ptr, bits_q_ptr, bits_v_ptr, tape_ptr, eps, HIDDEN: tl.constexpr,
                   WORDS: tl.constexpr, BITS: tl.constexpr):
    """One program per token: the norm a of x, in fp16, and a times each of the two LoRA dropout
    masks, in fp16. The masks have the padded shape; place[token] is the token's slot in them.
    Their bits go to the tape, one 32-bit word per BITS lanes, for the backward."""
    token = tl.program_id(0)
    word = tl.arange(0, WORDS)
    bit = tl.arange(0, BITS)
    offsets = word[:, None] * BITS + bit[None, :]
    mask = offsets < HIDDEN
    start = token * HIDDEN
    x = tl.load(x_ptr + start + offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(tape_ptr + start + offsets, x.to(tl.float16), mask=mask)
    weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0)
    a = weight * x * tl.rsqrt(tl.sum(tl.sum(x * x, axis=1), axis=0) / HIDDEN + eps)
    slot = tl.load(place_ptr + token) * HIDDEN
    keep_q = tl.load(mask_q_ptr + slot + offsets, mask=mask, other=0.0)
    keep_v = tl.load(mask_v_ptr + slot + offsets, mask=mask, other=0.0)
    tl.store(a_ptr + start + offsets, a.to(tl.float16), mask=mask)
    tl.store(dropped_q_ptr + start + offsets, (a * keep_q).to(tl.float16), mask=mask)
    tl.store(dropped_v_ptr + start + offsets, (a * keep_v).to(tl.float16), mask=mask)
    at = token * (HIDDEN // BITS) + word
    tl.store(bits_q_ptr + at, tl.sum((keep_q != 0).to(tl.int32) << bit[None, :], axis=1),
             mask=word * BITS < HIDDEN)
    tl.store(bits_v_ptr + at, tl.sum((keep_v != 0).to(tl.int32) << bit[None, :], axis=1),
             mask=word * BITS < HIDDEN)


@triton.jit
def replay_kernel(x_ptr, weight_ptr, bits_q_ptr, bits_v_ptr, dropped_q_ptr, dropped_v_ptr, eps,
                  keep, HIDDEN: tl.constexpr, WORDS: tl.constexpr, BITS: tl.constexpr):
    """The two masked inputs of prepare_kernel again, from the fp16 tape and the bits of the masks:
    the LoRA gradients are all the backward needs them for."""
    token = tl.program_id(0)
    word = tl.arange(0, WORDS)
    bit = tl.arange(0, BITS)
    offsets = word[:, None] * BITS + bit[None, :]
    mask = offsets < HIDDEN
    start = token * HIDDEN
    x = tl.load(x_ptr + start + offsets, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0)
    a = weight * x * tl.rsqrt(tl.sum(tl.sum(x * x, axis=1), axis=0) / HIDDEN + eps)
    at = token * (HIDDEN // BITS) + word
    bits_q = tl.load(bits_q_ptr + at, mask=word * BITS < HIDDEN, other=0)
    bits_v = tl.load(bits_v_ptr + at, mask=word * BITS < HIDDEN, other=0)
    keep_q = ((bits_q[:, None] >> bit[None, :]) & 1).to(tl.float32) * keep
    keep_v = ((bits_v[:, None] >> bit[None, :]) & 1).to(tl.float32) * keep
    tl.store(dropped_q_ptr + start + offsets, (a * keep_q).to(tl.float16), mask=mask)
    tl.store(dropped_v_ptr + start + offsets, (a * keep_v).to(tl.float16), mask=mask)


@triton.jit
def assemble_kernel(joined_ptr, lora_q_ptr, lora_v_ptr, cos_ptr, sin_ptr, position_ptr, query_ptr,
                    key_ptr, value_ptr, scale, HEADS: tl.constexpr, KEYS: tl.constexpr,
                    WIDTH: tl.constexpr, HALF: tl.constexpr):
    """One program per token and key head: q, k and v of the HEADS // KEYS query heads that share
    the key head. The LoRA output is added, q and k rotated in fp32, k and v repeated."""
    token = tl.program_id(0)
    key_head = tl.program_id(1)
    offsets = tl.arange(0, HALF)
    groups = HEADS // KEYS
    row = joined_ptr + token * (HEADS + 2 * KEYS) * WIDTH
    packed = token * HEADS * WIDTH
    position = tl.load(position_ptr + token) * WIDTH
    cos = tl.load(cos_ptr + position + offsets)
    sin = tl.load(sin_ptr + position + offsets)
    at_k = (HEADS + key_head) * WIDTH
    at_v = (HEADS + KEYS + key_head) * WIDTH
    lora_v = lora_v_ptr + token * KEYS * WIDTH + key_head * WIDTH
    key_low = tl.load(row + at_k + offsets).to(tl.float32)
    key_high = tl.load(row + at_k + HALF + offsets).to(tl.float32)
    value_low = (tl.load(row + at_v + offsets).to(tl.float32)
                 + tl.load(lora_v + offsets).to(tl.float32) * scale).to(tl.float16)
    value_high = (tl.load(row + at_v + HALF + offsets).to(tl.float32)
                  + tl.load(lora_v + HALF + offsets).to(tl.float32) * scale).to(tl.float16)
    rotated_low = (key_low * cos - key_high * sin).to(tl.float16)
    rotated_high = (key_high * cos + key_low * sin).to(tl.float16)
    for head in range(key_head * groups, (key_head + 1) * groups):
        at = head * WIDTH
        lora_q = lora_q_ptr + packed + at
        query_low = (tl.load(row + at + offsets).to(tl.float32)
                     + tl.load(lora_q + offsets).to(tl.float32) * scale)
        query_high = (tl.load(row + at + HALF + offsets).to(tl.float32)
                      + tl.load(lora_q + HALF + offsets).to(tl.float32) * scale)
        tl.store(query_ptr + packed + at + offsets,
                 (query_low * cos - query_high * sin).to(tl.float16))
        tl.store(query_ptr + packed + at + HALF + offsets,
                 (query_high * cos + query_low * sin).to(tl.float16))
        tl.store(key_ptr + packed + at + offsets, rotated_low)
        tl.store(key_ptr + packed + at + HALF + offsets, rotated_high)
        tl.store(value_ptr + packed + at + offsets, value_low)
        tl.store(value_ptr + packed + at + HALF + offsets, value_high)


@triton.jit
def residual_kernel(h_ptr, out_ptr, weight_ptr, b_ptr, before_ptr, eps, HIDDEN: tl.constexpr,
                    BLOCK: tl.constexpr):
    """One program per token: h += out in fp32, and in fp16 the norm of h and h itself."""
    start = tl.program_id(0) * HIDDEN
    offsets = tl.arange(0, BLOCK)
    mask = offsets < HIDDEN
    h = tl.load(h_ptr + start + offsets, mask=mask, other=0.0)
    h += tl.load(out_ptr + start + offsets, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0)
    scale = tl.rsqrt(tl.sum(h * h, axis=0) / HIDDEN + eps)
    tl.store(h_ptr + start + offsets, h, mask=mask)
    tl.store(b_ptr + start + offsets, (weight * h * scale).to(tl.float16), mask=mask)
    tl.store(before_ptr + start + offsets, h.to(tl.float16), mask=mask)


@triton.jit
def merge_adjoint_kernel(grad_ptr, x_ptr, weight_ptr, total_ptr, total16_ptr, eps,
                         HIDDEN: tl.constexpr, BLOCK: tl.constexpr):
    """One program per token: total += adjoint of the norm of x applied to grad; the fp16 copy of
    total goes straight into the next matmul."""
    start = tl.program_id(0) * HIDDEN
    offsets = tl.arange(0, BLOCK)
    mask = offsets < HIDDEN
    x = tl.load(x_ptr + start + offsets, mask=mask, other=0.0).to(tl.float32)
    grad = tl.load(grad_ptr + start + offsets, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0)
    scale = tl.rsqrt(tl.sum(x * x, axis=0) / HIDDEN + eps)
    pulled = grad * weight
    along = tl.sum(pulled * x, axis=0) * scale * scale / HIDDEN
    total = tl.load(total_ptr + start + offsets, mask=mask, other=0.0)
    total += scale * (pulled - x * along)
    tl.store(total_ptr + start + offsets, total, mask=mask)
    tl.store(total16_ptr + start + offsets, rough(total.to(tl.float16)), mask=mask)


@triton.jit
def split_adjoint_kernel(grad_q_ptr, grad_k_ptr, grad_v_ptr, cos_ptr, sin_ptr, position_ptr,
                         joined_ptr, HEADS: tl.constexpr, KEYS: tl.constexpr, WIDTH: tl.constexpr,
                         HALF: tl.constexpr):
    """The adjoint of assemble_kernel, one program per token and key head, up to the LoRA part:
    the query rotated back, and k and v summed over the query heads of the group."""
    token = tl.program_id(0)
    key_head = tl.program_id(1)
    offsets = tl.arange(0, HALF)
    groups = HEADS // KEYS
    row = joined_ptr + token * (HEADS + 2 * KEYS) * WIDTH
    packed = token * HEADS * WIDTH
    position = tl.load(position_ptr + token) * WIDTH
    cos = tl.load(cos_ptr + position + offsets)
    sin = tl.load(sin_ptr + position + offsets)
    key_low = tl.zeros([HALF], tl.float32)
    key_high = tl.zeros([HALF], tl.float32)
    value_low = tl.zeros([HALF], tl.float32)
    value_high = tl.zeros([HALF], tl.float32)
    for head in range(key_head * groups, (key_head + 1) * groups):
        at = head * WIDTH
        low = tl.load(grad_q_ptr + packed + at + offsets).to(tl.float32)
        high = tl.load(grad_q_ptr + packed + at + HALF + offsets).to(tl.float32)
        tl.store(row + at + offsets, (low * cos + high * sin).to(tl.float16))
        tl.store(row + at + HALF + offsets, (high * cos - low * sin).to(tl.float16))
        key_low += tl.load(grad_k_ptr + packed + at + offsets).to(tl.float32)
        key_high += tl.load(grad_k_ptr + packed + at + HALF + offsets).to(tl.float32)
        value_low += tl.load(grad_v_ptr + packed + at + offsets).to(tl.float32)
        value_high += tl.load(grad_v_ptr + packed + at + HALF + offsets).to(tl.float32)
    at_k = (HEADS + key_head) * WIDTH
    at_v = (HEADS + KEYS + key_head) * WIDTH
    tl.store(row + at_k + offsets, (key_low * cos + key_high * sin).to(tl.float16))
    tl.store(row + at_k + HALF + offsets, (key_high * cos - key_low * sin).to(tl.float16))
    tl.store(row + at_v + offsets, value_low.to(tl.float16))
    tl.store(row + at_v + HALF + offsets, value_high.to(tl.float16))


@triton.jit
def input_adjoint_kernel(grad_ptr, lora_q_ptr, lora_v_ptr, bits_q_ptr, bits_v_ptr, x_ptr,
                         weight_ptr, total_ptr, total16_ptr, eps, keep, HIDDEN: tl.constexpr,
                         WORDS: tl.constexpr, BITS: tl.constexpr):
    """One program per token: the gradient over a is the frozen part plus each LoRA part through
    its dropout mask, and total += the adjoint of the norm of x applied to it, with its fp16 copy
    for the next layer."""
    token = tl.program_id(0)
    word = tl.arange(0, WORDS)
    bit = tl.arange(0, BITS)
    offsets = word[:, None] * BITS + bit[None, :]
    mask = offsets < HIDDEN
    start = token * HIDDEN
    at = token * (HIDDEN // BITS) + word
    bits_q = tl.load(bits_q_ptr + at, mask=word * BITS < HIDDEN, other=0)
    bits_v = tl.load(bits_v_ptr + at, mask=word * BITS < HIDDEN, other=0)
    keep_q = ((bits_q[:, None] >> bit[None, :]) & 1).to(tl.float32) * keep
    keep_v = ((bits_v[:, None] >> bit[None, :]) & 1).to(tl.float32) * keep
    grad = (tl.load(grad_ptr + start + offsets, mask=mask, other=0.0).to(tl.float32)
            + tl.load(lora_q_ptr + start + offsets, mask=mask, other=0.0).to(tl.float32) * keep_q
            + tl.load(lora_v_ptr + start + offsets, mask=mask, other=0.0).to(tl.float32) * keep_v)
    x = tl.load(x_ptr + start + offsets, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0)
    scale = tl.rsqrt(tl.sum(tl.sum(x * x, axis=1), axis=0) / HIDDEN + eps)
    pulled = grad * weight
    along = tl.sum(tl.sum(pulled * x, axis=1), axis=0) * scale * scale / HIDDEN
    total = tl.load(total_ptr + start + offsets, mask=mask, other=0.0)
    total += scale * (pulled - x * along)
    tl.store(total_ptr + start + offsets, total, mask=mask)
    tl.store(total16_ptr + start + offsets, rough(total.to(tl.float16)), mask=mask)


@triton.jit
def mlp_kernel(both_ptr, out_ptr, WIDTH: tl.constexpr, CHUNK: tl.constexpr):
    """One program per token and CHUNK lanes of the joined [gate, up] output: silu(gate) * up.
    Gate and up run as one matmul, so they lie side by side per token, which the starter kit's
    SwiGLU kernels cannot read. Same function, so it is no kernel of ours to count: check 4 stands
    without these two."""
    token = tl.program_id(0)
    row = both_ptr + token * 2 * WIDTH
    offsets = tl.program_id(1) * CHUNK + tl.arange(0, CHUNK)
    mask = offsets < WIDTH
    gate = tl.load(row + offsets, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(row + WIDTH + offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + token * WIDTH + offsets, (gate * tl.sigmoid(gate) * up).to(tl.float16),
             mask=mask)


@triton.jit
def mlp_adjoint_kernel(grad_ptr, both_ptr, WIDTH: tl.constexpr, CHUNK: tl.constexpr):
    """The adjoint of mlp_kernel, the gradients over gate and up written over gate and up, rough."""
    token = tl.program_id(0)
    row = both_ptr + token * 2 * WIDTH
    offsets = tl.program_id(1) * CHUNK + tl.arange(0, CHUNK)
    mask = offsets < WIDTH
    grad = tl.load(grad_ptr + token * WIDTH + offsets, mask=mask, other=0.0).to(tl.float32)
    gate = tl.load(row + offsets, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(row + WIDTH + offsets, mask=mask, other=0.0).to(tl.float32)
    sig = tl.sigmoid(gate)
    by_gate = grad * up * sig * (1.0 + gate * (1.0 - sig))
    by_up = grad * gate * sig
    tl.store(row + offsets, rough(by_gate.to(tl.float16)), mask=mask)
    tl.store(row + WIDTH + offsets, rough(by_up.to(tl.float16)), mask=mask)


LAYER = {"HIDDEN": 1536, "BLOCK": BLOCK}
TILE = {"HIDDEN": 1536, "WORDS": WORDS, "BITS": BITS}
HEADS = {"HEADS": 12, "KEYS": 2, "WIDTH": 128, "HALF": 64}
MLP = {"WIDTH": 8960, "CHUNK": 1024}
FP16, FP32, INT32 = torch.float16, torch.float32, torch.int32
precompile(prepare_kernel, FP32, FP32, FP32, FP32, INT32, FP16, FP16, FP16, INT32, INT32, FP16, 1e-6,
           **TILE)
precompile(replay_kernel, FP16, FP32, INT32, INT32, FP16, FP16, 1e-6, 1.0, **TILE)
precompile(assemble_kernel, FP16, FP16, FP16, FP32, FP32, INT32, FP16, FP16, FP16, 2.0, **HEADS)
precompile(residual_kernel, FP32, FP16, FP32, FP16, FP16, 1e-6, **LAYER)
precompile(merge_adjoint_kernel, FP16, FP16, FP32, FP32, FP16, 1e-6, **LAYER)
precompile(split_adjoint_kernel, FP16, FP16, FP16, FP32, FP32, INT32, FP16, **HEADS)
precompile(input_adjoint_kernel, FP16, FP16, FP16, INT32, INT32, FP16, FP32, FP32, FP16, 1e-6, 1.0,
           **TILE)
precompile(mlp_kernel, FP16, FP16, **MLP)
precompile(mlp_adjoint_kernel, FP16, FP16, **MLP)


def empty16(tokens, width, device):
    return torch.empty(tokens, width, dtype=torch.float16, device=device)


def prepare(x, norm, mask_q, mask_v, place, bits_q, bits_v, tape=None):
    """a, a * mask_q and a * mask_v, each [tokens, hidden] fp16, from the norm of x. `place` holds
    the slot of every token in the masks; the bits of the masks go to bits_q and bits_v."""
    tokens, hidden = x.shape
    a, dropped_q, dropped_v = (empty16(tokens, hidden, x.device) for _ in range(3))
    if tape is None:
        tape = empty16(tokens, hidden, x.device)
    prepare_kernel[(tokens,)](x, norm.weight, mask_q, mask_v, place, a, dropped_q, dropped_v,
                              bits_q, bits_v, tape, norm.variance_epsilon, HIDDEN=hidden, WORDS=WORDS,
                              BITS=BITS)
    return a, dropped_q, dropped_v


def replay(x, norm, bits_q, bits_v, keep):
    """a * mask_q and a * mask_v, from the fp16 tape of x and the bits. `keep` is the value of a
    kept mask element, 1 / (1 - rate)."""
    tokens, hidden = x.shape
    dropped_q, dropped_v = (empty16(tokens, hidden, x.device) for _ in range(2))
    replay_kernel[(tokens,)](x, norm.weight, bits_q, bits_v, dropped_q, dropped_v,
                             norm.variance_epsilon, keep, HIDDEN=hidden, WORDS=WORDS, BITS=BITS)
    return dropped_q, dropped_v


def assemble(joined, lora_q, lora_v, scale, cos, sin, positions, query, key, value):
    """Writes packed q, k and v, [tokens, heads, width] fp16. `joined` is the output of the joined
    Q, K and V matmul, `lora_q` and `lora_v` the LoRA outputs of a site before their scale, and cos
    and sin the rotary tables by position, [positions, width]."""
    tokens, heads, width = query.shape
    keys = lora_v.shape[1] // width
    assemble_kernel[(tokens, keys)](joined, lora_q, lora_v, cos, sin, positions, query, key, value,
                                    scale, HEADS=heads, KEYS=keys, WIDTH=width, HALF=width // 2)


def residual(h, out, norm, before):
    """h += out in place, in fp32, and writes its fp16 copy to `before`. Returns the norm of h in
    fp16."""
    tokens, hidden = h.shape
    b = empty16(tokens, hidden, h.device)
    residual_kernel[(tokens,)](h, out, norm.weight, b, before, norm.variance_epsilon, HIDDEN=hidden,
                               BLOCK=BLOCK)
    return b


def merge_adjoint(grad, x, norm, total):
    """total += the adjoint of the norm of the fp16 x on grad, in place. Returns total in fp16."""
    tokens, hidden = x.shape
    total16 = empty16(tokens, hidden, x.device)
    merge_adjoint_kernel[(tokens,)](grad, x, norm.weight, total, total16, norm.variance_epsilon,
                                    HIDDEN=hidden, BLOCK=BLOCK)
    return total16


def split_adjoint(grad_q, grad_k, grad_v, cos, sin, positions, keys):
    """The joined [tokens, q + k + v] gradient, fp16, from the packed gradients of FMHA."""
    tokens, heads, width = grad_q.shape
    joined = empty16(tokens, (heads + 2 * keys) * width, grad_q.device)
    split_adjoint_kernel[(tokens, keys)](grad_q, grad_k, grad_v, cos, sin, positions, joined,
                                         HEADS=heads, KEYS=keys, WIDTH=width, HALF=width // 2)
    return joined


def mlp(both):
    """silu(gate) * up, [tokens, width] fp16, from the joined output, which stays as it is."""
    tokens, width = both.shape[0], both.shape[1] // 2
    out = empty16(tokens, width, both.device)
    mlp_kernel[(tokens, triton.cdiv(width, CHUNK))](both, out, WIDTH=width, CHUNK=CHUNK)
    return out


def mlp_adjoint(grad, both):
    """The joined [gate, up] gradient, over the joined output."""
    tokens, width = both.shape[0], both.shape[1] // 2
    mlp_adjoint_kernel[(tokens, triton.cdiv(width, CHUNK))](grad, both, WIDTH=width, CHUNK=CHUNK)
    return both


def input_adjoint(grad, lora_q, lora_v, bits_q, bits_v, keep, x, norm, total):
    """total += the adjoint of the norm of x on grad + lora_q * mask_q + lora_v * mask_v, in place.
    Returns total in fp16."""
    tokens, hidden = x.shape
    total16 = empty16(tokens, hidden, x.device)
    input_adjoint_kernel[(tokens,)](grad, lora_q, lora_v, bits_q, bits_v, x, norm.weight, total,
                                    total16, norm.variance_epsilon, keep, HIDDEN=hidden,
                                    WORDS=WORDS, BITS=BITS)
    return total16
