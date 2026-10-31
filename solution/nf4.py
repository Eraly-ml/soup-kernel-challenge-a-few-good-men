"""NF4 double-quant weights decoded in one Triton launch per decoder layer.

The sweep decodes the linear weights of all layers but the first and the last twice a step, so the
decode is the largest kernel of the step. bitsandbytes spends more on the way to it than the GPU
spends on the work: a torch.library dispatch for each decode, and a separate decode of the absmax
before it. Here one launch does both, for all seven weights of a layer at once: on a T4 that takes
half the GPU time of seven launches and a tenth of the CPU time. Only the packed weights stay in
memory, and one buffer the size of a layer's dense values.

A weight is stored as one byte per two values, high nibble first, in blocks of 64 values. The
value is table[nibble] * absmax of the block. The absmax is double-quantized, one byte per block:
scale_table[byte] * coarse[block // 256] + offset. All of it runs in fp32 and only the product is
rounded to fp16, so the result equals bitsandbytes' dequantize_4bit bit for bit. FP fusion stays
off: fused, the absmax is rounded once where bitsandbytes rounds twice.

The forward wants the values the other way round. cuBLAS runs x @ W' on a T4 at 6-18% fewer joules
per multiply when W' lies in memory as [in, out] than when W does as [out, in], and the backward's
y @ W is at its best the way W is stored. Each direction decodes anyway, so each gets its own form:
`nf4_columns_kernel` works in squares of a block by a block and writes one burst per column.

NF4 and double quantization: QLoRA (https://arxiv.org/abs/2305.14314) and bitsandbytes, whose
stored format this reads and whose decode arithmetic it follows. Weights of one input stacked for
one matmul: Soup's Fast-LoRA. A stored decode plan and a reused decode buffer: Unsloth
(kernels/utils.py, _NF4Plan, _scratch) and the write-ups by lweitkamp and numb3r33. NF4 in one
Triton launch is public in Unsloth (kernels/nf4.py). The kernel is ours.
"""

import itertools

import torch
import triton
import triton.language as tl

from .rough import rough
from .startup import precompile

BLOCKS = 32  # blocks per program; the decode is memory-bound, so this hardly matters
SLOTS = 8  # weights in one launch at most; a power of two, for tl.arange
FIELDS = 6  # numbers per weight in a plan: packed, scale, coarse, offset, dense offset, values
COLUMN_FIELDS = 8  # packed, scale, coarse, offset, dense offset, rows of the group, squares, blocks
arena, held, by_columns = None, None, False  # the buffer, its plan, and which way round it lies


@triton.jit
def nf4_decode_kernel(plan_ptr, scale_table_ptr, table_ptr, dense_ptr, SLOTS: tl.constexpr,
                      FIELDS: tl.constexpr, BLOCKS: tl.constexpr, VALUES: tl.constexpr,
                      COARSE: tl.constexpr, ROUGH: tl.constexpr = False):
    """One program per BLOCKS blocks of VALUES values of one weight. The plan starts with the first
    program of every weight, which tells the program its weight, and then holds one row per weight
    with the addresses to read. Scales first, then the values, for the backward rough ones."""
    program = tl.program_id(0)
    weight = tl.sum((program >= tl.load(plan_ptr + tl.arange(0, SLOTS))).to(tl.int32), axis=0) - 1
    row = plan_ptr + SLOTS + FIELDS * weight
    packed_ptr = tl.load(row).to(tl.pointer_type(tl.uint8))
    scale_ptr = tl.load(row + 1).to(tl.pointer_type(tl.uint8))
    coarse_ptr = tl.load(row + 2).to(tl.pointer_type(tl.float32))
    offset = tl.load(tl.load(row + 3).to(tl.pointer_type(tl.float32)))
    out_ptr = dense_ptr + tl.load(row + 4)
    n_values = tl.load(row + 5)
    block = (program - tl.load(plan_ptr + weight)) * BLOCKS + tl.arange(0, BLOCKS)
    scale = tl.load(scale_ptr + block, mask=block * VALUES < n_values, other=0)
    coarse = tl.load(coarse_ptr + block // COARSE, mask=block * VALUES < n_values, other=0.0)
    absmax = tl.load(scale_table_ptr + scale) * coarse + offset
    start = block[:, None] * VALUES
    pair = tl.arange(0, VALUES // 2)[None, :]
    packed = tl.load(packed_ptr + start // 2 + pair, mask=start + 2 * pair < n_values, other=0)
    high = tl.load(table_ptr + (packed >> 4)) * absmax[:, None]
    low = tl.load(table_ptr + (packed & 15)) * absmax[:, None]
    index = start + tl.arange(0, VALUES)[None, :]
    value = tl.interleave(high, low).to(tl.float16)
    if ROUGH:
        value = rough(value)
    tl.store(out_ptr + index, value, mask=index < n_values)


@triton.jit
def nf4_columns_kernel(plan_ptr, scale_table_ptr, table_ptr, dense_ptr, SLOTS: tl.constexpr,
                       FIELDS: tl.constexpr, VALUES: tl.constexpr, COARSE: tl.constexpr):
    """The values of nf4_decode_kernel in the form [in, out]: one program per VALUES rows by one
    block of columns of one weight. The rows of a square hold VALUES blocks, and each column of the
    square lands in one burst. Weights of a group lie side by side in a column."""
    program = tl.program_id(0)
    weight = tl.sum((program >= tl.load(plan_ptr + tl.arange(0, SLOTS))).to(tl.int32), axis=0) - 1
    row = plan_ptr + SLOTS + FIELDS * weight
    packed_ptr = tl.load(row).to(tl.pointer_type(tl.uint8))
    scale_ptr = tl.load(row + 1).to(tl.pointer_type(tl.uint8))
    coarse_ptr = tl.load(row + 2).to(tl.pointer_type(tl.float32))
    offset = tl.load(tl.load(row + 3).to(tl.pointer_type(tl.float32)))
    out_ptr = dense_ptr + tl.load(row + 4)
    stride = tl.load(row + 5).to(tl.int32)
    squares = tl.load(row + 6).to(tl.int32)
    blocks = tl.load(row + 7).to(tl.int32)
    index = program - tl.load(plan_ptr + weight).to(tl.int32)
    across = index // squares
    rows = (index % squares) * VALUES + tl.arange(0, VALUES)
    block = rows * blocks + across
    scale = tl.load(scale_ptr + block)
    coarse = tl.load(coarse_ptr + block // COARSE)
    absmax = tl.load(scale_table_ptr + scale) * coarse + offset
    pair = tl.arange(0, VALUES // 2)[None, :]
    packed = tl.load(packed_ptr + block[:, None] * (VALUES // 2) + pair)
    high = tl.load(table_ptr + (packed >> 4)) * absmax[:, None]
    low = tl.load(table_ptr + (packed & 15)) * absmax[:, None]
    column = across * VALUES + tl.arange(0, VALUES)[None, :]
    tl.store(out_ptr + column * stride + rows[:, None], tl.interleave(high, low).to(tl.float16))


precompile(nf4_decode_kernel, torch.int64, torch.float32, torch.float32, torch.float16, SLOTS=SLOTS,
           FIELDS=FIELDS, BLOCKS=BLOCKS, VALUES=64, COARSE=256, enable_fp_fusion=False)
precompile(nf4_columns_kernel, torch.int64, torch.float32, torch.float32, torch.float16, SLOTS=SLOTS,
           FIELDS=COLUMN_FIELDS, VALUES=64, COARSE=256, enable_fp_fusion=False)


def arena_of(size, device):
    """One buffer for the layers: only one layer is decoded at a time. Allocating it for every
    layer would leave holes in the allocator that smaller tensors fill, and each next allocation
    would then reserve new memory."""
    global arena, held
    if arena is None or arena.numel() < size:
        arena, held = None, None
        arena = torch.empty(size, dtype=torch.float16, device=device)
    return arena


class Plan:
    """Weights decoded together. Their addresses sit in a small table on the GPU, so a launch takes
    the same four pointers for one weight or for seven. `groups` lists the Params4bit of the plan:
    each inner list comes out as one tensor, its weights stacked by rows, because weights that
    read the same input run as one matmul."""

    def __init__(self, groups):
        self.weights = [weight for group in groups for weight in group]
        assert len(self.weights) <= SLOTS
        states = [weight.quant_state for weight in self.weights]
        self.state, self.device = states[0], states[0].code.device
        self.shapes = [(sum(weight.quant_state.shape[0] for weight in group),
                        group[0].quant_state.shape[1]) for group in groups]
        self.sizes = [rows * columns for rows, columns in self.shapes]
        sizes = [state.shape[0] * state.shape[1] for state in states]
        programs = [triton.cdiv(size, BLOCKS * state.blocksize)
                    for size, state in zip(sizes, states)]
        self.programs = sum(programs)
        self.firsts = [*itertools.accumulate(programs[:-1], initial=0)]
        self.places = [*zip(itertools.accumulate(sizes[:-1], initial=0), sizes)]
        self.tables = {}
        self.column_plan(groups, states)

    def column_plan(self, groups, states):
        """Where each weight goes in the form [in, out], when squares of a block cover it."""
        block = states[0].blocksize
        self.squared = all(side % block == 0 for state in states for side in state.shape)
        counts = [state.shape[0] // block * (state.shape[1] // block) for state in states]
        self.square_programs = sum(counts)
        self.square_firsts = [*itertools.accumulate(counts[:-1], initial=0)]
        self.columns = []
        base = 0
        for group, (rows, width) in zip(groups, self.shapes):
            before = 0
            for weight in group:
                self.columns.append((base + before, rows, weight.quant_state.shape[0] // block,
                                     width // block))
                before += weight.quant_state.shape[0]
            base += rows * width

    def table(self):
        """The addresses of the weights where they are now: streamed, they sit in one of two slots
        of stream.Ring, so a plan ends up with a table for each."""
        where = self.weights[0].data_ptr()
        if where not in self.tables:
            states = [weight.quant_state for weight in self.weights]
            rows = [(weight.data_ptr(), state.absmax.data_ptr(), state.state2.absmax.data_ptr(),
                     state.offset.data_ptr(), at, size)
                    for weight, state, (at, size) in zip(self.weights, states, self.places)]
            table = (self.firsts + [2 ** 31 - 1] * (SLOTS - len(self.firsts))
                     + [*itertools.chain(*rows)])
            self.tables[where] = torch.tensor(table, dtype=torch.int64, device=self.device)
            if self.squared:
                rows = [(*row[:4], *place) for row, place in zip(rows, self.columns)]
                table = (self.square_firsts + [2 ** 31 - 1] * (SLOTS - len(self.square_firsts))
                         + [*itertools.chain(*rows)])
                self.tables[where, "columns"] = torch.tensor(table, dtype=torch.int64,
                                                             device=self.device)
        return self.tables[where]

    def decode(self, columns=False):
        """The dense fp16 [out, in] values of every group, as bitsandbytes' dequantize_4bit gives
        them for each weight. They sit in the shared buffer: valid until another plan decodes.
        Weights are frozen, so a plan that is still in the buffer stays right and is not decoded
        again: the last layer around the head, the first layer from one step to the next. With
        `columns` they lie in memory as [in, out], for x @ W', and come out as a transposed view
        of the same shape and values. A plan already in the buffer stays as it lies."""
        global held, by_columns
        state = self.state
        size = sum(self.sizes)
        dense = arena_of(size, self.device)
        if held is not self:
            table = self.table()
            by_columns = columns and self.squared
            if by_columns:
                nf4_columns_kernel[(self.square_programs,)](
                    self.tables[self.weights[0].data_ptr(), "columns"], state.state2.code,
                    state.code, dense, SLOTS=SLOTS, FIELDS=COLUMN_FIELDS, VALUES=state.blocksize,
                    COARSE=state.state2.blocksize, enable_fp_fusion=False)
            else:
                nf4_decode_kernel[(self.programs,)](
                    table, state.state2.code, state.code, dense, SLOTS=SLOTS, FIELDS=FIELDS,
                    BLOCKS=BLOCKS, VALUES=state.blocksize, COARSE=state.state2.blocksize,
                    enable_fp_fusion=False)
            held = self
        parts = dense[:size].split(self.sizes)
        if by_columns:
            return [part.view(shape[::-1]).t() for part, shape in zip(parts, self.shapes)]
        return [part.view(shape) for part, shape in zip(parts, self.shapes)]

    def rough(self):
        """The values of `decode`, rough, see rough.py. A weight moves by at most 1/32, a third of what
        NF4 has already rounded. For the backward alone, and for one use: no later decode takes them
        for the exact values."""
        global held
        state = self.state
        size = sum(self.sizes)
        dense = arena_of(size, self.device)
        nf4_decode_kernel[(self.programs,)](
            self.table(), state.state2.code, state.code, dense, SLOTS=SLOTS, FIELDS=FIELDS,
            BLOCKS=BLOCKS, VALUES=state.blocksize, COARSE=state.state2.blocksize, ROUGH=True,
            enable_fp_fusion=False)
        held = None
        parts = dense[:size].split(self.sizes)
        return [part.view(shape) for part, shape in zip(parts, self.shapes)]


precompile(nf4_decode_kernel, torch.int64, torch.float32, torch.float32, torch.float16, SLOTS=SLOTS,
           FIELDS=FIELDS, BLOCKS=BLOCKS, VALUES=64, COARSE=256, ROUGH=True, enable_fp_fusion=False)
