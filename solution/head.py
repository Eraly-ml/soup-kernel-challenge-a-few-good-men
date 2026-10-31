"""The output head and the loss over the 151,936-token vocabulary, one tile of it at a time.

Base Soup keeps a [batch, seq, vocab] tensor of logits, an fp32 copy of it, log-probabilities and
gradients of that shape alive at once: gigabytes on the largest batch, more than the model's
weights. It reads the logits three more times for the entropy, the accuracy and the rewind log.

Here all tokens of a step make one pass over the vocabulary, which comes from host memory tile by
tile (stream.py), so no tensor of vocabulary width ever exists. `head_tile_kernel` does the work of
those reads in one launch over a tile's logits, and leaves in their place what the gradient needs.
The softmax is summed tile by tile and divided by its total only at the end, which is what lets
one pass do it; each tile carries its own scale, the block floating point of codecs.
Idea of running the head in chunks: Soup's own `stream_loss_chunk` option (a switch this task
forbids), and the chunked cross-entropy of Cut Cross-Entropy. A fused loss over the head is public
in Liger Kernel (ops/fused_linear_cross_entropy.py) and Unsloth (kernels/cross_entropy_loss.py).
The code here is ours.
"""

import torch
import triton
import triton.language as tl

from .startup import precompile
from .sums import summed_in_fp16

BLOCK = 1024
SHIFT = 24.0  # logits are shifted before exp(), which then overflows fp32 only above about 112
TILE = 7680  # vocabulary rows per tile: 23 MiB, so that a tile fits the slot of a layer


@triton.jit(do_not_specialize=["n_tile", "first"])
def head_tile_kernel(logits_ptr, targets_ptr, sums_ptr, scale_ptr, n_tile, first,
                     SHIFT: tl.constexpr, BLOCK: tl.constexpr):
    """One program per token, for one tile of the vocabulary. sums: total, weighted, best, best
    index, chosen. Tokens with a target get exp(logit - peak of the tile) in place of their
    logits, so that the fp16 gemm after this cannot underflow, and that peak in scale."""
    token = tl.program_id(0)
    row = logits_ptr + token * n_tile
    base = sums_ptr + 5 * token
    target = tl.load(targets_ptr + token)
    total = tl.load(base)
    weighted = tl.load(base + 1)
    best = tl.load(base + 2)
    best_index = tl.load(base + 3)
    peak = tl.full([], -1e30, tl.float32)
    for start in range(0, n_tile, BLOCK):
        offsets = start + tl.arange(0, BLOCK)
        mask = offsets < n_tile
        logit = tl.load(row + offsets, mask=mask, other=-1e4).to(tl.float32) - SHIFT
        power = tl.where(mask, tl.exp(logit), 0.0)
        total += tl.sum(power, axis=0)
        weighted += tl.sum(power * logit, axis=0)
        top = tl.argmax(logit, axis=0).to(tl.float32)
        value = tl.max(logit, axis=0)
        best_index = tl.where(value > best, first.to(tl.float32) + start + top, best_index)
        best = tl.maximum(best, value)
        peak = tl.maximum(peak, value)
    tl.store(base, total)
    tl.store(base + 1, weighted)
    tl.store(base + 2, best)
    tl.store(base + 3, best_index)
    inside = target - first
    if (inside >= 0) & (inside < n_tile):
        tl.store(base + 4, tl.load(row + inside).to(tl.float32) - SHIFT)
    if target >= 0:
        for start in range(0, n_tile, BLOCK):
            offsets = start + tl.arange(0, BLOCK)
            mask = offsets < n_tile
            logit = tl.load(row + offsets, mask=mask, other=-1e4).to(tl.float32) - SHIFT
            tl.store(row + offsets, tl.exp(logit - peak).to(row.dtype.element_ty), mask=mask)
        tl.store(scale_ptr + token, tl.exp(peak))


precompile(head_tile_kernel, torch.float16, torch.int32, torch.float32, torch.float32, 0, 0,
           SHIFT=SHIFT, BLOCK=BLOCK)


def head_stream(states16, targets, grad_scale, store):
    """The whole head for all tokens of a step in one pass over the vocabulary. Returns the
    [tokens, 3] stats, cross-entropy, entropy and correct, and the gradient of the scaled loss over
    the states. Tokens without a target only need their entropy, so they skip the gradient."""
    device, dtype = states16.device, states16.dtype
    tokens, width = states16.shape
    wanted = (targets >= 0).nonzero()[:, 0]
    rest = (targets < 0).nonzero()[:, 0]
    order = torch.cat((wanted, rest))
    n_wanted = wanted.numel()
    states, sorted_targets = states16[order], targets[order]
    del states16
    sums = torch.zeros(tokens, 5, device=device)
    sums[:, 2] = -1e30
    scale = torch.zeros(tokens, device=device)
    expected = torch.zeros(n_wanted, width, device=device)
    buffer = torch.empty(tokens * TILE, dtype=dtype, device=device)
    first = 0
    for index in range(len(store.tiles[0])):
        weight = store.tile(index)
        logits = buffer[:tokens * weight.shape[0]].view(tokens, -1)
        torch.matmul(states, weight.t(), out=logits)
        head_tile_kernel[(tokens,)](logits, sorted_targets, sums, scale, weight.shape[0], first,
                                    SHIFT=SHIFT, BLOCK=BLOCK)
        part = summed_in_fp16(logits[:n_wanted], weight).float()
        part *= scale[:n_wanted, None]
        expected += part
        first += weight.shape[0]
    del states, buffer, logits, part
    total = sums[:, 0]
    log_total = torch.log(total)
    in_order = torch.zeros(tokens, 3, device=device)
    in_order[:, 1] = log_total - sums[:, 1] / total
    in_order[:n_wanted, 0] = log_total[:n_wanted] - sums[:n_wanted, 4]
    in_order[:n_wanted, 2] = (sums[:n_wanted, 3] == sorted_targets[:n_wanted].float()).float()
    stats = torch.zeros(tokens, 3, device=device)
    stats[order] = in_order
    expected /= total[:n_wanted, None]
    expected -= store.rows(sorted_targets[:n_wanted]).float()
    expected *= grad_scale
    grads = torch.zeros(tokens, width, dtype=dtype, device=device)
    grads[wanted] = expected.to(dtype)
    return stats, grads
