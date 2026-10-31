"""The frozen weights stay in host memory: the GPU holds two slots of them, not the whole model.

With the weights resident the GPU can never go below 625 MiB of packed NF4 plus 445 MiB of
embedding matrix, whatever the step does. But a step made of sweeps reads them one layer, or one
tile of the vocabulary, at a time and in an order it knows in advance, so:

- all packed weights and their block scales sit in pinned host memory, and so does the embedding
  matrix, which is tied to lm_head. Nothing is computed there;
- `Ring` keeps two GPU slots. The main stream reads one, DMA on a second CUDA stream fills the
  other with what comes next. CUDA events order them both ways: the main stream waits for the copy
  it needs, and a copy waits for the main stream to finish with the slot it overwrites;
- `Store` is what a training step sees: `layer(i, following)` rebinds the seven NF4 weights of
  layer i to the slot, `tile(i)` gives the next piece of the vocabulary, `rows(tokens)` looks tokens
  up on the host.

The bytes are the bytes of the resident model: only where they live changes, and the time of the
copies counts. Loading has to play along, because the weights must never reach the GPU as a whole:
see `patch_loading`. Without it the run peaks at 1,068 MiB during the load whatever the step saves.

Idea of streaming the layers of a frozen model from pinned host memory into a small pool of GPU
buffers: Soup's `stream_layers` (utils/layer_stream.py, a switch this task forbids) and virtual
texturing in game engines. The code here is ours.
"""

import dataclasses
import math

import torch

from . import ahead
from .ahead import aligned
from .fused import BITS
from .head import TILE
from .parking import PIECES
from .startup import own_stream, pinned_memory, read_done

MAX_TOKENS = 2048  # 4 rows of 512 tokens, the largest batch of the task
MODEL = {"model_type": "qwen2", "hidden_size": 1536, "intermediate_size": 8960,
         "num_hidden_layers": 28, "vocab_size": 151936}  # what startup.PINNED is cut for


class Ring:
    """Two GPU slots over parts of pinned host memory, in the order the caller announces."""

    def __init__(self, parts, device):
        self.parts = parts
        self.stream = own_stream(device)
        size = max(part.numel() for part in parts)
        self.slots = [torch.empty(size, dtype=torch.uint8, device=device) for _ in range(2)]
        self.holding = [None, None]  # slot -> part
        self.filled = [None, None]  # slot -> event: its copy is done
        self.drained = [None, None]  # slot -> event: the main stream is done reading it
        self.active = 0

    def fetch(self, slot, part):
        host = self.parts[part]
        with torch.cuda.stream(self.stream):
            if self.drained[slot] is not None:
                self.drained[slot].wait()
            self.slots[slot][:host.numel()].copy_(host, non_blocking=True)
            self.filled[slot] = torch.cuda.Event()
            self.filled[slot].record()
        self.holding[slot] = part

    def enter(self, part, following=None):
        """After this call, work on the main stream reads `part` from `slots[active]`. The work
        already submitted still reads what it read, and `following` starts copying behind it."""
        other = 1 - self.active
        if self.holding[self.active] != part:
            if self.holding[other] != part:
                self.fetch(other, part)
            self.drained[self.active] = torch.cuda.Event()
            self.drained[self.active].record()
            self.active, other = other, self.active
        if self.filled[self.active] is not None:
            self.filled[self.active].wait()
            self.filled[self.active] = None
        if following is not None and self.holding[other] != following:
            self.fetch(other, following)


def layer_sites(layer):
    """The seven NF4 modules of a decoder layer. LoRA wraps q_proj and v_proj."""
    attn, block = layer.self_attn, layer.mlp
    modules = (attn.q_proj, attn.k_proj, attn.v_proj, attn.o_proj, block.gate_proj, block.up_proj,
               block.down_proj)
    return [getattr(module, "base_layer", module) for module in modules]


def layout(layers):
    """Where the packed bytes and the block scales of each weight sit inside the part of its layer.
    Returns the plan of each layer, and the size of each part."""
    plans, sizes = [], []
    for layer in layers:
        plan, total = [], 0
        for module in layer_sites(layer):
            weight, state = module.weight, module.weight.quant_state
            size, size2 = weight.data.numel(), state.absmax.numel()
            start2 = total + aligned(size)
            plan.append((weight, state, total, size, weight.data.shape, start2, size2,
                         state.absmax.shape))
            total = start2 + aligned(size2)
        plans.append(plan)
        sizes.append(total)
    return plans, sizes


def bound(buffer, plan):
    """The tensors of one layer inside `buffer`: (weight, state, packed view, block scale view)."""
    return [(weight, state, buffer[start:start + size].view(shape),
             buffer[start2:start2 + size2].view(shape2))
            for weight, state, start, size, shape, start2, size2, shape2 in plan]


def take(blocks, size):
    """`size` bytes off the first of the pinned blocks with room for them."""
    for index, block in enumerate(blocks):
        if block.numel() >= size:
            blocks[index] = block[size:]
            return block[:size]
    raise MemoryError(f"no pinned block has {size} bytes left")


def cut(blocks, pieces):
    """Tensors of these (dtype, shape), each in one piece, taken from `blocks`."""
    tensors = []
    for dtype, shape in pieces:
        size = math.prod(shape)
        tensors.append(take(blocks, aligned(size * dtype.itemsize)).view(dtype)[:size].view(shape))
    return tensors


class Store:
    """All frozen weights of the model on the host, behind one ring. The pinned blocks hold them
    and, as `parked`, the host tensors of parking.py, each per layer: see startup.pinned_memory.
    The table comes first, the one piece that needs the largest block."""

    def __init__(self, inner, device):
        self.device, self.on_gpu = device, False
        config, self.weight = inner.config, inner.lm_head.weight
        layers = inner.model.layers
        self.plans, sizes = layout(layers)
        blocks = list(pinned_memory())
        waiting, dtype = len(layers) - 1, self.weight.dtype
        hidden = config.hidden_size
        table, *rows = cut(blocks, (
            (dtype, self.weight.shape),
            *[(dtype, (MAX_TOKENS, 2 * config.intermediate_size))] * waiting,
            *[(dtype, (PIECES * MAX_TOKENS, hidden))] * waiting,
            *[(dtype, (MAX_TOKENS, hidden))] * waiting,
            *[(torch.int32, (MAX_TOKENS, hidden // BITS))] * (2 * waiting)))
        both, attention, x, *bits = [rows[start:start + waiting]
                                     for start in range(0, len(rows), waiting)]
        self.parked = both, x, bits, attention
        parts = [take(blocks, size) for size in sizes]
        parts = self.pack_layers(parts) + self.pack_table(table)
        self.ring = Ring(parts, device)
        self.bind_slots()

    def pack_layers(self, parts):
        for part, plan in zip(parts, self.plans):
            for weight, state, packed, absmax in bound(part, plan):
                packed.copy_(weight.data)
                absmax.copy_(state.absmax)
                weight.data, state.absmax = packed, absmax
        return list(parts)

    def pack_table(self, table):
        self.table = table
        if table.data_ptr() != self.weight.data_ptr():  # the read-ahead has put it there
            self.table.copy_(self.weight.data)
        self.weight.data = self.table
        return [piece.view(-1) for piece in self.table.view(torch.uint8).split(TILE)]

    def bind_slots(self):
        count, width = len(self.plans), self.table.shape[1]
        self.views = [[bound(slot, plan) for plan in self.plans] for slot in self.ring.slots]
        self.tiles = [[slot[:part.numel()].view(self.table.dtype).view(-1, width)
                       for part in self.ring.parts[count:]] for slot in self.ring.slots]

    def layer(self, index, following=None):
        """Rebind the weights of layer `index` to the GPU. `following` is the part read next: a
        layer, or `len(layers)` for the first tile."""
        self.ring.enter(index, following)
        for weight, state, packed, absmax in self.views[self.ring.active][index]:
            weight.data, state.absmax = packed, absmax
        if self.on_gpu:
            self.weight.data, self.on_gpu = self.table, False

    def resident(self):
        """Every weight on the GPU, for a step the sweep does not take: Soup's own path reads them
        where they stand. The next sweep rebinds them to the slots, and the table to the host."""
        if self.on_gpu:
            return
        for part, plan in zip(self.ring.parts, self.plans):
            for weight, state, packed, absmax in bound(part, plan):
                weight.data, state.absmax = packed.to(self.device), absmax.to(self.device)
        self.weight.data = self.table.to(self.device)
        self.on_gpu = True

    def tile(self, index):
        """Rows TILE * index onwards of the vocabulary table, on the GPU. The backward sweep
        starts at the last layer, which is still decoded (nf4.Plan), so the last tile announces
        the layer below it."""
        first = len(self.plans)
        part = first + index
        self.ring.enter(part, part + 1 if part + 1 < len(self.ring.parts) else first - 2)
        return self.tiles[self.ring.active][index]

    def rows(self, tokens):
        """The fp16 rows of `tokens`, on the GPU. A gather on the host costs less than a tile."""
        return self.table[tokens.cpu()].to(self.device)


def streamed(model, hf_quantizer):
    """Only the model of this task, in 4 bits on a GPU: anything else loads as it always did. A
    step the sweep refuses later, by adjoint.usable, gets the weights back from `Store.resident`."""
    config = getattr(hf_quantizer, "quantization_config", None)
    model_config = getattr(model, "config", None)
    return isinstance(model, torch.nn.Module) and hasattr(model, "lm_head") \
        and getattr(config, "load_in_4bit", False) and torch.cuda.is_available() \
        and all(getattr(model_config, key, None) == value for key, value in MODEL.items())


def patch_loading():
    """Keep the weights off the GPU from the start. They are still quantized on the GPU, one at a
    time, because that gives the same bytes as base Soup, but each goes to the host right after.
    Every patch looks at the model and leaves any other load alone."""
    import peft
    import transformers.core_model_loading as loading
    import transformers.modeling_utils as modeling
    from transformers.integrations import bitsandbytes as hf_bnb

    warm_up = modeling.caching_allocator_warmup
    load = modeling.PreTrainedModel._load_pretrained_model
    quantize = hf_bnb.Bnb4bitQuantize.convert
    rename = loading.WeightRenaming.convert
    wrap = peft.get_peft_model

    def caching_allocator_warmup(model, device_map, hf_quantizer):
        """Reserves the size of the whole model in one tensor before the first weight loads."""
        if not streamed(model, hf_quantizer):
            warm_up(model, device_map, hf_quantizer)

    def load_pretrained(model, state_dict, checkpoint_files, load_config, *args, **kwargs):
        if streamed(model, load_config.hf_quantizer):
            device_map = {"model.embed_tokens": "cpu", "lm_head": "cpu", **load_config.device_map}
            load_config = dataclasses.replace(load_config, device_map=device_map)
            model.weights_on_host = True
            read_done()
            model.read_ahead = ahead.same_file(checkpoint_files)
        return load(model, state_dict, checkpoint_files, load_config, *args, **kwargs)

    def take_or_convert(self, layer_name, model=None, **kwargs):
        taken = ahead.take(self, layer_name, model) if getattr(model, "read_ahead", False) else None
        return rename(self, layer_name, model=model, **kwargs) if taken is None else taken

    def convert(self, *args, **kwargs):
        result = quantize(self, *args, **kwargs)
        if streamed(kwargs.get("model"), self.hf_quantizer):
            for param in result.values():
                param.data = param.data.cpu()
                param.quant_state.absmax = param.quant_state.absmax.cpu()
        return result

    def get_peft_model(model, *args, **kwargs):
        """LoRA follows the base layers to the host, and with no accelerate map for one device
        `Accelerator.prepare_model` would call model.to(device) and upload the whole model."""
        wrapped = wrap(model, *args, **kwargs)
        if hasattr(model, "weights_on_host"):
            device = torch.device("cuda", torch.cuda.current_device())
            for param in wrapped.parameters():
                if param.requires_grad:
                    param.data = param.data.to(device)
            model.hf_device_map = {"": device.index}
            model.store = Store(model, device)
        return wrapped

    modeling.caching_allocator_warmup = caching_allocator_warmup
    modeling.PreTrainedModel._load_pretrained_model = staticmethod(load_pretrained)
    hf_bnb.Bnb4bitQuantize.convert = convert
    loading.WeightRenaming.convert = take_or_convert
    peft.get_peft_model = get_peft_model
