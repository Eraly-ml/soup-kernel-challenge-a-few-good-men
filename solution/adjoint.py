"""One training step with a reverse sweep written by hand: no autograd graph, no groups of rows.

Autograd keeps every activation of 28 layers alive until backward: 1.5 GB of graph for 512 tokens.
The backward of a layer needs much less.
Forward runs under no_grad over all real tokens of the batch and sends to host memory, see
parking.py, what the backward needs of each layer: one fp16 hidden vector per token, the bits of
the dropout masks, the outputs of G and U, and the q, k, v and output of the attention. Backward
walks the layers in reverse and applies the transposed products to what the forward left, so each
weight runs once each way and FMHA backward reads the very q, k and v of the forward. Only the
first norm is computed again, from the hidden vector, which takes no multiplies. The frozen weights
wait on the host as well, see stream.py.

The adjoint of one layer, with x its input and h the running gradient over its output:

    mbar = h D'     dg = mbar * u * (sig + g sig (1 - sig))     du = mbar * g * sig
    bbar = dg G' + du U'
    ybar = (h + N2'(bbar)) O'
    qbar, kbar, vbar = FMHA backward of ybar
    abar = qbar Q' + kbar K' + vbar V' + s (mask_q qbar B_q A_q + mask_v vbar B_v A_v)
    gradient over x = h + N2'(bbar) + N1'(abar)

The attention, forward and backward, is the cuDNN FMHA pair base Soup itself runs, called directly
row by row: without the autograd engine around it a row costs 90 us to launch instead of 800 us
for the same adjoint written out in torch ops (measured on the T4). LoRA gradients go straight
into .grad, scaled like the loss by the real GradScaler scale, so unscaling, clipping and AdamW
run as usual.

This is the exact gradient in exact arithmetic. The forward is base Soup's in what it computes. In
the backward the frozen matmuls read rough operands and rough weights (rough.py), the
inputs of the two norms come from a tape rounded to fp16, and the frozen matmuls add up in fp16
(sums.py), so it matches autograd to the cosine in the README, not bit for bit. The rest is what base
Soup does: the same fp16 and fp32 mix, the same masks from the same draws of the random stream, once
per step, the same FMHA kernels.

Q, K and V as one matmul and the dropout mask kept for the backward: Soup's Fast-LoRA
(utils/fast_lora.py, fast_lora_qkv.py, fast_lora_mlp.py) and the hand-written LoRA backward of
Unsloth (kernels/fast_lora.py). Computing part of a layer again in the backward from a saved input:
gradient checkpointing. Padding removed inside a batch: TRL's and Soup's padding_free, a switch
this task forbids. Calling the FMHA pair directly is public in Unsloth
(utils/hf_packed_attention.py). The code here is ours.
"""

import functools

import torch
import torch.nn.functional as F

from .fused import (assemble, input_adjoint, merge_adjoint, mlp, mlp_adjoint, prepare, replay,
                    residual, split_adjoint)
from .head import head_stream
from .nf4 import Plan, arena_of
from .norm import rms_norm, rms_norm_adjoint
from .parking import PIECES, Parking
from .startup import finish
from .stream import MAX_TOKENS, MODEL, Store
from .sums import summed_in_fp16

FMHA_BACKWARD = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default


class Site:
    """One LoRA site, with its fp16 weights for the step: they cannot change before the next."""

    def __init__(self, module):
        self.a, self.b = module.lora_A["default"].weight, module.lora_B["default"].weight
        self.scale = module.scaling["default"]
        self.rate = module.lora_dropout["default"].p
        self.keep = 1 / (1 - self.rate)

    def start_step(self):
        self.a16, self.b16 = self.a.half(), self.b.half()

    def adjoint(self, bar, dropped, down):
        """Writes the two gradients of the site; returns the gradient over its input, unmasked."""
        self.b.grad = torch.matmul(bar.transpose(0, 1) * self.scale, down).float()
        back = torch.matmul(bar, self.b16) * self.scale
        self.a.grad = torch.matmul(back.transpose(0, 1), dropped).float()
        return torch.matmul(back, self.a16)


class Layer:
    """What the sweep reads of one decoder layer."""

    def __init__(self, layer):
        attn, block = layer.self_attn, layer.mlp
        self.norm1, self.norm2 = layer.input_layernorm, layer.post_attention_layernorm
        self.query, self.value = Site(attn.q_proj), Site(attn.v_proj)
        projections = attn.q_proj.base_layer, attn.k_proj, attn.v_proj.base_layer
        self.bias = torch.cat([module.bias for module in projections]).half()
        self.heads, self.keys = attn.config.num_attention_heads, attn.config.num_key_value_heads
        self.plan = Plan([[module.weight for module in projections], [attn.o_proj.weight],
                          [block.gate_proj.weight, block.up_proj.weight],
                          [block.down_proj.weight]])


@functools.cache
def sweep_of(core, store):
    """What a sweep over this model keeps from step to step. The decode buffer comes first, so that
    it does not end up between the tensors of a step, where a larger step finds a hole too small."""
    layers = [Layer(layer) for layer in core.layers]
    arena_of(sum(layers[0].plan.sizes), layers[0].plan.device)
    return layers, Parking(*store.parked, core.norm.weight.device, keep_first=True)


class Masks:
    """The LoRA dropout masks of the padded batch: each draw has the full [batch, seq, hidden]
    shape of base Soup, so the random stream runs the same. `place` is the slot of each real
    token in them."""

    def __init__(self, shape, place):
        self.ones = torch.ones(shape, dtype=torch.float32, device=place.device)
        self.place = place

    def draw(self, rate):
        return F.dropout(self.ones, rate, True)


def pieces(slot, tokens, heads):
    """q, k, v and y, [tokens, heads, width], and h before D, [tokens, hidden], in their slot."""
    *inputs, before = slot.view(PIECES, tokens, -1)
    return *(piece.view(tokens, heads, -1) for piece in inputs), before


def row_view(x, start, end):
    """One row of a packed [tokens, heads, width] tensor, in FMHA's [1, heads, length, width]."""
    return x[start:end].transpose(0, 1)[None]


def attention_row(query, key, value):
    """base's FMHA on one row, with what its backward needs."""
    return torch._scaled_dot_product_efficient_attention(query, key, value, None, True, 0.0, True)


def attention_row_adjoint(ybar, query, key, value, y, state):
    """base's FMHA backward on one row: the gradients over q, k and v."""
    log_sum, seed, offset = state
    return FMHA_BACKWARD(ybar, query, key, value, None, y, log_sum, seed, offset, 0.0,
                         [True, True, True, False], True)[:3]


def packed(parts, out=None):
    """Rows in FMHA's layout back into one [tokens, heads, width] tensor."""
    return torch.cat([part[0].transpose(0, 1) for part in parts], out=out)


def forward_layer(layer, h, rotary, segments, masks, parked, index):
    """h is fp32 and advances in place. Returns what the backward reads that stays on the GPU: the
    two LoRA middles and the rest of the state of FMHA."""
    tokens = h.shape[0]
    site_q, site_v = layer.query, layer.value
    dense_in, dense_o, dense_middle, dense_d = layer.plan.decode(columns=True)
    mask_q, mask_v = masks.draw(site_q.rate), masks.draw(site_v.rate)
    x, bits, attention = parked.slots(index)
    a, dropped_q, dropped_v = prepare(h, layer.norm1, mask_q, mask_v, masks.place, *bits, tape=x)
    del mask_q, mask_v
    down_q, down_v = F.linear(dropped_q, site_q.a16), F.linear(dropped_v, site_v.a16)
    del dropped_q, dropped_v
    joined = F.linear(a, dense_in, layer.bias)
    lora_q, lora_v = F.linear(down_q, site_q.b16), F.linear(down_v, site_v.b16)
    del a
    query, key, value, y, before = pieces(attention, tokens, layer.heads)
    assemble(joined, lora_q, lora_v, site_q.scale, *rotary, query, key, value)
    del joined, lora_q, lora_v
    states = [attention_row(row_view(query, *segment), row_view(key, *segment),
                            row_view(value, *segment)) for segment in segments]
    packed([state[0] for state in states], y)
    b = residual(h, F.linear(y.view(tokens, -1), dense_o), layer.norm2, before)
    parked.wait()
    torch.matmul(b, dense_middle.t(), out=parked.both)
    del b
    if index < parked.layers:  # the last layer stays in its slots for the backward
        parked.send(index)
    h += F.linear(mlp(parked.both), dense_d)
    return down_q, down_v, [state[1:] for state in states]


def backward_layer(layer, down_q, down_v, states, hbar, hbar16, rotary, segments, parked, index):
    """Writes the LoRA gradients of a layer. hbar is the gradient over the output of the layer in
    fp32, hbar16 its fp16 copy. Returns the new fp16 copy, with hbar moved to the input in place.
    Every tensor goes as soon as it is done: the peak of a step is here."""
    site_q, site_v = layer.query, layer.value
    x, bits, attention = parked.inputs(index)
    rough = 0 < index < parked.layers  # the first and the last layer use their decode twice
    dense_in, dense_o, dense_middle, dense_d = layer.plan.rough() if rough else layer.plan.decode()
    grad = summed_in_fp16(hbar16, dense_d)
    parked.wait()
    bbar = summed_in_fp16(mlp_adjoint(grad, parked.both), dense_middle)
    del grad
    if index:
        parked.fetch(index - 1)
    query, key, value, y, before = pieces(attention, x.shape[0], layer.heads)
    h1bar16 = merge_adjoint(bbar, before, layer.norm2, hbar)
    del bbar
    ybar = summed_in_fp16(h1bar16, dense_o).view(query.shape)
    del h1bar16
    parts = [attention_row_adjoint(row_view(ybar, *segment), row_view(query, *segment),
                                   row_view(key, *segment), row_view(value, *segment),
                                   row_view(y, *segment), state)
             for segment, state in zip(segments, states)]
    del ybar
    grads = split_adjoint(*(packed(part) for part in zip(*parts)), *rotary, layer.keys)
    del parts
    dropped_q, dropped_v = replay(x, layer.norm1, *bits, site_q.keep)
    lora_q = site_q.adjoint(grads[:, :site_q.b.shape[0]], dropped_q, down_q)
    lora_v = site_v.adjoint(grads[:, -site_v.b.shape[0]:], dropped_v, down_v)
    del dropped_q, dropped_v
    if not index:  # no gradient goes on into the frozen embedding
        return None
    return input_adjoint(summed_in_fp16(grads, dense_in), lora_q, lora_v, *bits, site_q.keep, x,
                         layer.norm1, hbar)


@torch.no_grad()
def adjoint_step(trainer, model, inputs):
    """What Trainer.training_step and SFTTrainer.compute_loss do for one batch."""
    finish()
    if not model.training:
        model.train()
    inputs = trainer._prepare_inputs(inputs)
    ids, mask, labels = inputs["input_ids"], inputs["attention_mask"], inputs["labels"]
    device = ids.device
    batch, width = ids.shape
    lengths = mask.sum(dim=1).tolist()
    inner = model.base_model.model
    core, store = inner.model, inner.store
    targets_all = torch.cat((labels[:, 1:], torch.full((batch, 1), -100, device=device)), dim=1)
    items = (targets_all != -100).sum()
    scaler = trainer.accelerator.scaler
    loss_scale = scaler.scale(torch.ones((), device=device)).item()  # also starts the scaler
    grad_scale = loss_scale / items.item()

    positions = [torch.arange(length, device=device) for length in lengths]
    where_row = torch.cat([torch.full_like(p, row) for row, p in enumerate(positions)])
    where_pos = torch.cat(positions)
    segments, start = [], 0
    for length in lengths:
        segments.append((start, start + length))
        start += length
    tokens = ids[where_row, where_pos]
    targets = targets_all[where_row, where_pos].to(torch.int32)
    rotary = (*rotary_table(core, width, device), where_pos.to(torch.int32))

    layers, parked = sweep_of(core, store)
    for layer in layers:
        layer.query.start_step()
        layer.value.start_step()
    parked.start_step(tokens.shape[0])
    h = store.rows(tokens).float()
    masks = Masks((batch, width, core.config.hidden_size),
                  (where_row * width + where_pos).to(torch.int32))
    tape = []
    for index, layer in enumerate(layers):
        store.layer(index, index + 1)
        tape.append(forward_layer(layer, h, rotary, segments, masks, parked, index))
    del masks
    stats, grads = head_stream(rms_norm(h, core.norm.weight, core.norm.variance_epsilon).half(),
                               targets, grad_scale, store)
    hbar = rms_norm_adjoint(grads.float(), h, core.norm.weight, core.norm.variance_epsilon)
    hbar16 = hbar.half()
    del grads, h
    last = len(layers) - 1
    for index in range(last, -1, -1):
        needed = min(index, last - 1)  # the last layer is still decoded, see nf4.Plan
        store.layer(needed, needed - 1 if needed else None)
        hbar16 = backward_layer(layers[index], *tape.pop(), hbar, hbar16, rotary, segments, parked,
                                index)
    parked.end_step()
    ahead = getattr(trainer, "ahead", None)
    if ahead is not None and trainer.state.global_step + 1 < trainer.state.max_steps:
        ahead.read()

    row_loss = torch.zeros(batch, device=device, dtype=stats.dtype)
    row_loss.index_add_(0, where_row, stats[:, 0])
    row_tokens = (labels[:, 1:] != -100).sum(dim=1)
    total_tokens = int(mask.sum())
    loss = row_loss.sum() / items
    metrics = trainer._metrics["train"]
    metrics["entropy"].append((stats[:, 1].sum() / total_tokens).item())
    trainer._total_train_tokens += total_tokens
    metrics["num_tokens"] = [trainer._total_train_tokens]
    metrics["mean_token_accuracy"].append((stats[:, 2].sum() / items).item())
    rewind = getattr(trainer, "_soup_rewind_state", None)
    if rewind is not None and not rewind.failed:
        means = row_loss / row_tokens.clamp(min=1).to(row_loss.dtype)
        rewind.on_micro_batch(step=int(trainer.state.global_step) + 1, row_loss=means.tolist(),
                              row_tokens=row_tokens.tolist())
    return loss.detach()


def rotary_table(core, length, device):
    """cos and sin of the positions 0..length-1, [length, head_dim], as Qwen2RotaryEmbedding."""
    frequencies = core.rotary_emb.inv_freq.float().to(device)
    positions = torch.arange(length, device=device, dtype=torch.float32)
    angles = positions[:, None] * frequencies[None, :]
    angles = torch.cat((angles, angles), dim=-1)
    scaling = core.rotary_emb.attention_scaling
    return angles.cos() * scaling, angles.sin() * scaling


def store_of(model):
    """The Store of a model with its weights on the host, which is what stream.streamed let in."""
    store = getattr(getattr(getattr(model, "base_model", None), "model", None), "store", None)
    return store if isinstance(store, Store) else None


def plain_lora(model):
    """The one LoRA of this task, on the model stream.py took. The sweep holds for nothing else, and
    neither do the memos of patch_walks: with a second adapter, requires_grad moves during a run."""
    if store_of(model) is None or len(model.peft_config) != 1:
        return False
    peft = model.peft_config.get("default")
    return set(getattr(peft, "target_modules", None) or ()) == {"q_proj", "v_proj"} \
        and getattr(peft, "lora_dropout", 0) > 0 and not getattr(peft, "use_dora", True) \
        and not getattr(peft, "lora_bias", True) and not getattr(peft, "modules_to_save", True)


def usable(trainer, model, inputs):
    """Only the run of this task: plain_lora, fp16, one batch per optimizer step, a padded plain
    batch that fits the host memory and has a target: the mean over no targets would divide by
    zero, where base Soup gets a NaN."""
    if not plain_lora(model) or not isinstance(inputs, dict):
        return False
    mask, labels = inputs.get("attention_mask"), inputs.get("labels")
    if mask is None or labels is None or mask.numel() > MAX_TOKENS:
        return False
    return bool(getattr(trainer.args, "fp16", False)) \
        and getattr(trainer.args, "gradient_accumulation_steps", 1) == 1 \
        and getattr(trainer.accelerator, "scaler", None) is not None \
        and bool((labels[:, 1:] != -100).any())


def weights_to_gpu(model):
    """For Soup's own code, which reads the frozen weights on the GPU."""
    store = store_of(model)
    if store is not None:
        store.resident()


@functools.cache
def parameter_count(model):
    """What Trainer.floating_point_ops multiplies by: it does not change during a run."""
    return model.num_parameters(exclude_embeddings=True)


@functools.cache
def trainable(model):
    """The LoRA weights, in the order of model.parameters()."""
    return [param for param in model.parameters() if param.requires_grad]


class Ahead:
    """The batches of an epoch, the next of which can be read before the Trainer asks for it. When
    a step has launched its backward the host is about 120 ms ahead of the GPU, with nothing to do,
    and the batch takes 3 ms of Python. Read after the step it takes those 3 ms on an idle GPU. The
    same batches in the same order: rewind's record of them is a queue."""

    def __init__(self, batches):
        self.batches, self.early = batches, None

    def __iter__(self):
        return self

    def __next__(self):
        batch, self.early = self.early, None
        return next(self.batches) if batch is None else batch

    def read(self):
        self.early = next(self.batches, None)  # the last batch of an epoch has none behind it


def no_waits_on_the_host(trainer):
    """What holds the host back for no result, once at the start of training. A batch leaves the
    pinned memory it is already in without waiting for the GPU, which would be the whole backward.
    Soup commits one row of its tracker per logged step, and SQLite's default waits for an fsync
    each time: 5 ms of every step. With a write-ahead log, as here, NORMAL writes the same rows; it
    only leaves the last ones to the disk's cache if the machine itself goes down."""
    for loader in trainer.accelerator._dataloaders:
        loader._non_blocking = True
    for callback in trainer.callback_handler.callbacks:
        connection = getattr(getattr(callback, "tracker", None), "_get_conn", None)  # Soup's only
        if connection is not None and not connection().in_transaction:
            connection().execute("PRAGMA synchronous=NORMAL")


def patch_walks(trainer):
    """Three calls of every Trainer step walk the 933 modules of this model for what the first walk
    already knows: floating_point_ops counts all parameters, twice over, and clipping and zero_grad
    look through all 450 parameters for the 112 with a gradient. That is milliseconds of Python per
    step on an idle GPU. Here they read the answer of the first walk: the same count, the same
    tensors in the same order, which holds because this task's parameters never change during a
    run. Any model or LoRA but this task's takes the Trainer's own code."""
    import peft

    count_operations, clip = trainer.floating_point_ops, trainer._clip_grad_norm
    samples, zero_all = trainer.get_batch_samples, peft.PeftModel.zero_grad

    def get_batch_samples(self, epoch_iterator, num_batches, device):
        if not plain_lora(self.model):
            return samples(self, epoch_iterator, num_batches, device)
        ahead = getattr(self, "ahead", None)
        if ahead is None or ahead.batches is not epoch_iterator:
            ahead = self.ahead = Ahead(epoch_iterator)
            no_waits_on_the_host(self)
        return samples(self, ahead, num_batches, device)

    def floating_point_ops(self, inputs):
        if not plain_lora(self.model) or "input_ids" not in inputs:
            return count_operations(self, inputs)
        return 6 * inputs["input_ids"].numel() * parameter_count(self.model)

    def clip_grad_norm(self, model):
        if not plain_lora(model):
            return clip(self, model)
        return self.accelerator.clip_grad_norm_(trainable(model), self.args.max_grad_norm)

    def zero_grad(self, set_to_none=True):
        if not plain_lora(self) or not set_to_none:
            return zero_all(self, set_to_none)
        for param in trainable(self):
            param.grad = None

    trainer.floating_point_ops, trainer._clip_grad_norm = floating_point_ops, clip_grad_norm
    trainer.get_batch_samples = get_batch_samples
    peft.PeftModel.zero_grad = zero_grad


def patch_training_step():
    from trl import SFTTrainer

    original, evaluate = SFTTrainer.training_step, SFTTrainer.evaluate

    def training_step(self, model, inputs, *args, **kwargs):
        if usable(self, model, inputs):
            return adjoint_step(self, model, inputs)
        weights_to_gpu(model)
        return original(self, model, inputs, *args, **kwargs)

    def evaluate_on_gpu(self, *args, **kwargs):
        weights_to_gpu(self.model)
        return evaluate(self, *args, **kwargs)

    SFTTrainer.training_step, SFTTrainer.evaluate = training_step, evaluate_on_gpu
    patch_walks(SFTTrainer)


def patch_kbit_preparation():
    """The checkpoint's embedding matrix loads as fp16 and peft upcasts it to fp32 for good: 891
    MiB instead of 445 for the same values, and 1.3 GiB while both exist. It is frozen and tied
    to lm_head, so it can simply stay fp16; the casts of base Soup around it go away too. Only for
    the model of this task: any other keeps its fp32 embedding."""
    import peft

    original = peft.prepare_model_for_kbit_training

    def prepare_kbit(model, *args, **kwargs):
        config = getattr(model, "config", None)
        if not isinstance(model, torch.nn.Module) or \
                any(getattr(config, key, None) != value for key, value in MODEL.items()):
            return original(model, *args, **kwargs)
        weight = model.get_input_embeddings().weight
        data, weight.data = weight.data, weight.data.new_empty(0)
        try:
            return original(model, *args, **kwargs)
        finally:
            weight.data = data

    peft.prepare_model_for_kbit_training = prepare_kbit
