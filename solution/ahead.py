"""The checkpoint, read and quantized beside the imports.

The load was the longest stretch of the main thread after the imports: 3 s for 196 weights, each
read through an mmap, converted from bf16 on the host, copied unpinned, quantized and brought back.
None of that needs the main thread, or even the model, only the file:

- one preadv per weight into the memory that will be pinned, then DMA and the conversion on the GPU:
  0.78 s against 1.38 s for the same values. That memory is free until Store cuts it up;
- the quantization is bitsandbytes's own, so the bytes are the ones the load would make;
- the load then only takes the results, weight by weight. Whatever is not ready, or is not what the
  load asks for, loads as it always did.

The read and the GPU part both run beside the imports. bitsandbytes reaches for modules, which must
not happen while main is importing those, so the GPU part waits until main has imported them, 0.38 s
better than waiting for the end of all imports: the load then finds everything ready.

The table of NF4 values is made once, for all weights. bitsandbytes makes it with torch.tensor() and
no dtype, and transformers sets the default dtype to fp16 while it builds the model, so a read that
ends in that window would get an fp16 table, and the decode kernel reads the table itself. Nothing
protects the timing, so such a table is refused and the load goes plain. Nothing else in a quantized
weight depends on it.
"""

import json
import os
import struct
import time

import torch

from . import startup

CHECKPOINT = "Qwen/Qwen2.5-1.5B-Instruct"  # the model of the task; any other loads as always
QUANTIZED = dict(blocksize=64, compress_statistics=True, quant_type="nf4",
                 quant_storage=torch.uint8)  # what the config of the task gives each weight
EMBEDDING = "model.embed_tokens.weight"
ALIGN = 64  # a tensor in a slot starts where an allocated one would: Triton keys on alignment
ready, embedding, ready_file = {}, None, None  # weight name -> (packed bytes, quant state)


def aligned(size):
    return -(-size // ALIGN) * ALIGN


def header(path):
    with open(path, "rb") as handle:
        size = struct.unpack("<Q", handle.read(8))[0]
        table = json.loads(handle.read(size))
    table.pop("__metadata__", None)
    return table, 8 + size


def read_exactly(fd, view, offset):
    """A short read would otherwise be quantized as if it were the checkpoint."""
    if os.preadv(fd, [view], offset) != len(view):
        raise OSError("short read")


def weights_of(table):
    """The weights the quantizer takes, in file order: [(name, first byte, last byte, shape)]."""
    found = [(name, *info["data_offsets"], info["shape"]) for name, info in table.items()
             if name.startswith("model.layers.") and name.endswith("_proj.weight")]
    return sorted(found, key=lambda weight: weight[1])


def read_embedding(fd, table, start, cursor):
    """Into the first bytes of the first block, where Store cuts its table: it is fp16 there."""
    global embedding
    info = table[EMBEDDING]
    begin, end = info["data_offsets"]
    raw = torch.empty(end - begin, dtype=torch.uint8)
    read_exactly(fd, memoryview(raw.numpy()), start + begin)
    host = place(aligned(end - begin), cursor)
    embedding = host.view(torch.float16)[:(end - begin) // 2].view(info["shape"])
    embedding.copy_(raw.view(torch.bfloat16).view(info["shape"]))


def place(size, cursor):
    """The next `size` bytes of the memory startup.pin_ahead maps."""
    cursor[0] += size
    return startup.whole[cursor[0] - size:cursor[0]]


def read_and_quantize():
    global ready_file
    from huggingface_hub import try_to_load_from_cache  # transformers has imported it

    path = try_to_load_from_cache(CHECKPOINT, "model.safetensors")
    if not isinstance(path, str):
        return
    table, start = header(path)
    weights = weights_of(table)
    first, last = table[EMBEDDING]["data_offsets"]
    size = aligned(last - first) + sum(weight[2] - weight[1] for weight in weights)
    names = EMBEDDING, *(weight[0] for weight in weights)
    if size > sum(startup.PINNED) or any(table[name]["dtype"] != "BF16" for name in names):
        return
    startup.claim(size)
    while startup.whole is None:
        if not startup.pinning.is_alive():
            return
        time.sleep(0.002)
    cursor, placed = [0], []
    fd = os.open(path, os.O_RDONLY)
    try:
        read_embedding(fd, table, start, cursor)
        for name, begin, end, shape in weights:
            host = place(end - begin, cursor)
            read_exactly(fd, memoryview(host.numpy()), start + begin)
            placed.append((name, host.view(torch.bfloat16).view(shape)))
    finally:
        os.close(fd)
    startup.filled.set()
    startup.quantizable.wait()
    startup.pinning.join()
    import bitsandbytes as bnb

    device = torch.device("cuda", torch.cuda.current_device())
    code = bnb.functional.get_4bit_type("nf4", device=device)
    if code.dtype != torch.float32:
        return
    for name, host in placed:
        value = host.to(device, non_blocking=True).to(torch.float16)
        packed, state = bnb.functional.quantize_4bit(value, **QUANTIZED)
        state.code = code
        state.absmax = state.absmax.cpu()
        ready[name] = packed.cpu(), state
    ready_file = path


startup.beside(read_and_quantize)


def same_file(checkpoint_files):
    return ready_file is not None and checkpoint_files is not None and [
        os.path.realpath(path) for path in checkpoint_files] == [os.path.realpath(ready_file)]


def take(transform, layer_name, model):
    """What transform.convert would return for this weight, if it is ready and asked for."""
    global embedding
    if layer_name == EMBEDDING and embedding is not None:
        transform.collected_tensors.clear()
        value, embedding = embedding, None
        return {layer_name: value}
    if transform.quantization_operation is None or layer_name not in ready:
        return None
    import bitsandbytes as bnb

    old = model.get_parameter_or_buffer(layer_name)
    if old.dtype != torch.float16 or any(getattr(old, name) != value
                                         for name, value in QUANTIZED.items()):
        return None
    transform.collected_tensors.clear()
    packed, state = ready.pop(layer_name)
    param = bnb.nn.Params4bit(packed, requires_grad=False, **old.__dict__)
    param.quant_state, param.bnb_quantized = state, True
    param.module.quant_state = state
    param.module._is_hf_initialized = True
    return {layer_name: param}
