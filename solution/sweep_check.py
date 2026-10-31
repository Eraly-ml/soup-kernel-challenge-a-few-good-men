"""CPU check of the reverse sweep against autograd on a tiny Qwen2 with LoRA, without a GPU or
Triton: the kernels are replaced with PyTorch. Prints the gap of every case and asserts the targets.
test_sweep.py runs it under `pytest solution`, in a process of its own, because it puts a fake
triton in sys.modules and in fp32 changes the casts of every tensor. Nothing else imports it.

    python solution/sweep_check.py fp32    all in fp32: the sweep equals autograd below 1e-5
    python solution/sweep_check.py fp16    the real casts, against fp32 autograd: rounding only

Cases: rows of different lengths, a batch of 3, a row with one label, a non-zero lora_B, and the
real GradScaler scale. The fp16 tape is on in fp16 mode only. After them, what the sweep must
refuse, that it takes the config of this task as Soup builds it from task/task.yaml, and that
Soup's own code gets the weights back on the GPU.
"""

import copy
import pathlib
import sys
import types

import torch
import torch.nn.functional as F

import peft
from peft import LoraConfig, TaskType, get_peft_model
from transformers import Qwen2Config, Qwen2ForCausalLM

fake = types.ModuleType("triton")
fake.jit = lambda fn=None, **options: fn if fn is not None else (lambda inner: inner)
fake.language = types.ModuleType("triton.language")
fake.language.constexpr = int
fake.cdiv = lambda a, b: -(-a // b)
sys.modules["triton"], sys.modules["triton.language"] = fake, fake.language
sys.modules["torchao"] = None  # peft checks it on a plain nn.Linear and refuses Kaggle's old one
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path[0] = str(ROOT)  # the repository, not solution/: `import solution.adjoint` must work

import solution.adjoint as adj  # noqa: E402

MODE = sys.argv[1] if len(sys.argv) > 1 else "fp32"
SCALE = 65536.0
VOCAB = 300


def stats_py(logits, targets, stats, grad_scale):
    logp = F.log_softmax(logits.float(), dim=-1)
    stats[:, 1] = -(logp.exp() * logp).sum(-1)
    stats[:, 0] = 0.0
    stats[:, 2] = 0.0
    for i in range(logits.shape[0]):
        if targets[i] >= 0:
            stats[i, 0] = -logp[i, targets[i]]
            stats[i, 2] = float(logp[i].argmax() == targets[i])
            grad = logp[i].exp()
            grad[targets[i]] -= 1.0
            logits[i] = (grad * grad_scale).to(logits.dtype)


def norm(x, weight, eps):
    return weight * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps))


def norm_adjoint(grad, x, weight, eps):
    with torch.enable_grad():
        x = x.clone().requires_grad_()
        y = norm(x, weight, eps)
    return torch.autograd.grad(y, x, grad)[0]


def mlp(both):
    gate, up = both.float().chunk(2, dim=1)
    return (F.silu(gate) * up).to(both.dtype)


def mlp_adjoint(grad, both):
    gate, up = both.float().chunk(2, dim=1)
    grad = grad.float()
    sig = torch.sigmoid(gate)
    return torch.cat((grad * up * sig * (1 + gate * (1 - sig)), grad * gate * sig),
                     dim=1).to(both.dtype)


def attention_row(query, key, value):
    return F.scaled_dot_product_attention(query, key, value, is_causal=True), None, None, None


def attention_row_adjoint(ybar, query, key, value, y, state):
    with torch.enable_grad():
        leaves = [t.detach().requires_grad_() for t in (query, key, value)]
        y = F.scaled_dot_product_attention(*leaves, is_causal=True)
    return torch.autograd.grad(y, leaves, ybar)


def rotate(x, cos, sin):
    half = x.shape[-1] // 2
    return x * cos + torch.cat((-x[..., half:], x[..., :half]), dim=-1) * sin


def unrotate(xbar, cos, sin):
    half = xbar.shape[-1] // 2
    return xbar * cos + torch.cat((xbar[..., half:] * sin[..., half:],
                                   -xbar[..., :half] * sin[..., :half]), dim=-1)


def prepare(x, norm_module, mask_q, mask_v, place, bits_q, bits_v, tape=None):
    if tape is not None:
        tape.copy_(x)
    hidden = x.shape[1]
    a = norm(x.float(), norm_module.weight, norm_module.variance_epsilon)
    keep_q, keep_v = mask_q.view(-1, hidden)[place], mask_v.view(-1, hidden)[place]
    bits_q.copy_(keep_q != 0)
    bits_v.copy_(keep_v != 0)
    return a.half(), (a * keep_q).half(), (a * keep_v).half()


def replay(x, norm_module, bits_q, bits_v, keep):
    a = norm(x.float(), norm_module.weight, norm_module.variance_epsilon)
    return (a * bits_q.float() * keep).half(), (a * bits_v.float() * keep).half()


def assemble(joined, lora_q, lora_v, scale, cos_all, sin_all, positions, out_q, out_k, out_v):
    tokens, width = joined.shape[0], cos_all.shape[1]
    cos, sin = cos_all[positions][:, None], sin_all[positions][:, None]
    keys = lora_v.shape[1] // width
    groups = lora_q.shape[1] // lora_v.shape[1]
    q_size, k_size = lora_q.shape[1], lora_v.shape[1]
    full = joined.float()
    query = (full[:, :q_size] + lora_q.float() * scale).view(tokens, -1, width)
    key = full[:, q_size:q_size + k_size].view(tokens, keys, width)
    value = (full[:, q_size + k_size:] + lora_v.float() * scale).view(tokens, keys, width)
    out_q.copy_(rotate(query, cos, sin).half())
    out_k.copy_(rotate(key, cos, sin).half().repeat_interleave(groups, dim=1))
    out_v.copy_(value.half().repeat_interleave(groups, dim=1))


def residual(h, out, norm_module, before):
    h += out.float()
    before.copy_(h.half())
    return norm(h, norm_module.weight, norm_module.variance_epsilon).half()


def merge_adjoint(grad, x, norm_module, total):
    total += norm_adjoint(grad.float(), x.float(), norm_module.weight, norm_module.variance_epsilon)
    return total.half()


def split_adjoint(grad_q, grad_k, grad_v, cos_all, sin_all, positions, keys):
    tokens, heads, width = grad_q.shape
    cos, sin = cos_all[positions][:, None], sin_all[positions][:, None]
    query = unrotate(grad_q.float(), cos, sin).view(tokens, -1).half()
    key = grad_k.float().view(tokens, keys, -1, width).sum(2)
    key = unrotate(key, cos, sin).view(tokens, -1).half()
    value = grad_v.float().view(tokens, keys, -1, width).sum(2).view(tokens, -1).half()
    return torch.cat((query, key, value), dim=1)


def input_adjoint(grad, lora_q, lora_v, bits_q, bits_v, keep, x, norm_module, total):
    full = (grad.float() + lora_q.float() * bits_q.float() * keep
            + lora_v.float() * bits_v.float() * keep)
    total += norm_adjoint(full, x.float(), norm_module.weight, norm_module.variance_epsilon)
    return total.half()


def head_stream(states, targets, grad_scale, store):
    logits = torch.matmul(states, store.table.t())
    stats = torch.empty(states.shape[0], 3)
    stats_py(logits, targets, stats, grad_scale)
    grads = torch.zeros_like(states)
    wanted = targets >= 0
    grads[wanted] = torch.matmul(logits[wanted], store.table)
    return stats, grads


class FakeStore:
    """Weights where they are."""

    def __init__(self, model):
        inner = model.base_model.model
        self.table, self.calls = inner.lm_head.weight.data, []
        self.parked = (len(inner.model.layers) - 1, 2 * inner.config.intermediate_size,
                       inner.config.hidden_size, self.table.dtype)

    def layer(self, index, following=None):
        pass

    def resident(self):
        self.calls.append("resident")

    def rows(self, tokens):
        return self.table[tokens]


class FakeParking:
    """Copies instead of the host: the same numbers without CUDA. Bits are whole booleans here."""

    def __init__(self, layers, width, hidden, dtype, device, keep_first=False):
        self.layers, self.width, self.hidden, self.dtype = layers, width, hidden, dtype
        self.saved = {}

    def start_step(self, tokens):
        self.both = torch.empty(tokens, self.width, dtype=self.dtype)
        self.x = [torch.empty(tokens, self.hidden, dtype=self.dtype) for _ in range(2)]
        self.bits = [[torch.empty(tokens, self.hidden, dtype=torch.bool) for _ in range(2)]
                     for _ in range(2)]
        self.attention = [torch.empty(adj.PIECES * tokens, self.hidden, dtype=self.dtype)
                          for _ in range(2)]

    def end_step(self):
        self.saved = {}

    def wait(self):
        pass

    def slots(self, layer):
        return self.x[layer % 2], self.bits[layer % 2], self.attention[layer % 2]

    def inputs(self, layer):
        return self.slots(layer)

    def send(self, layer):
        self.saved[layer] = (self.x[layer % 2].clone(), [b.clone() for b in self.bits[layer % 2]],
                             self.both.clone(), self.attention[layer % 2].clone())

    def fetch(self, layer):
        x, bits, both, attention = self.saved[layer]
        self.x[layer % 2].copy_(x)
        for slot, value in zip(self.bits[layer % 2], bits):
            slot.copy_(value)
        self.attention[layer % 2].copy_(attention)
        self.both.copy_(both)


class FakePlan:
    def __init__(self, groups):
        self.groups = groups
        self.sizes, self.device = [0], torch.device("cpu")

    def decode(self, columns=False):
        return [torch.cat([weight.half() for weight in group]) for group in self.groups]

    rough = decode


adj.head_stream = head_stream
adj.rms_norm, adj.rms_norm_adjoint, adj.mlp, adj.mlp_adjoint = norm, norm_adjoint, mlp, mlp_adjoint
adj.attention_row, adj.attention_row_adjoint = attention_row, attention_row_adjoint
adj.Plan, adj.Parking, adj.Store = FakePlan, FakeParking, FakeStore
adj.prepare, adj.replay, adj.assemble, adj.residual = prepare, replay, assemble, residual
adj.merge_adjoint, adj.split_adjoint = merge_adjoint, split_adjoint
adj.input_adjoint = input_adjoint
if MODE == "fp32":
    # Copies, as the real casts make: the sweep advances h in place after taping it.
    to_float = torch.Tensor.float
    torch.Tensor.half = lambda self: self.clone()
    torch.Tensor.float = lambda self: to_float(self).clone()

torch.manual_seed(0)
config = Qwen2Config(vocab_size=VOCAB, hidden_size=48, intermediate_size=96, num_hidden_layers=3,
                     num_attention_heads=6, num_key_value_heads=2, max_position_embeddings=64,
                     use_sliding_window=False, tie_word_embeddings=True)
model = Qwen2ForCausalLM(config)
model.train()
for param in model.parameters():
    param.requires_grad = False
model = get_peft_model(model, LoraConfig(r=4, lora_alpha=8, lora_dropout=0.05,
                                         target_modules=["q_proj", "v_proj"]))
model.train()
for name, param in model.named_parameters():
    if "lora_B" in name:
        param.data.normal_(0, 0.02)  # the A gradients and the LoRA path are really tested then


def make_batch(lengths, one_label_row=None):
    width = max(lengths)
    batch = len(lengths)
    ids = torch.randint(1, VOCAB, (batch, width))
    mask = torch.zeros(batch, width, dtype=torch.long)
    labels = torch.full((batch, width), -100)
    for row, length in enumerate(lengths):
        mask[row, :length] = 1
        start = length - 1 if row == one_label_row else length // 2
        labels[row, start:length] = ids[row, start:length]
    return {"input_ids": ids * mask, "attention_mask": mask, "labels": labels}


def reference(batch):
    """Autograd over the padded batch, in fp32, with the loss scale of the real step."""
    copy_of = copy.deepcopy(model)
    torch.manual_seed(5)
    ids, mask, labels = batch["input_ids"], batch["attention_mask"], batch["labels"]
    out = copy_of(input_ids=ids, attention_mask=mask)
    targets = torch.cat((labels[:, 1:], torch.full((ids.shape[0], 1), -100)), dim=1)
    loss = F.cross_entropy(out.logits.view(-1, VOCAB), targets.view(-1), ignore_index=-100)
    (loss * SCALE).backward()
    logp = F.log_softmax(out.logits, dim=-1)
    entropy = (-(logp.exp() * logp).sum(-1) * mask).sum() / mask.sum()
    end = torch.randn(1)
    grads = {n: p.grad.clone() for n, p in copy_of.named_parameters() if p.requires_grad}
    return loss.item(), entropy.item(), grads, end.item()


class Scaler:
    def scale(self, value):
        return value * SCALE


class Rewind:
    failed = False

    def on_micro_batch(self, step, row_loss, row_tokens):
        self.row_loss, self.row_tokens = row_loss, row_tokens


def run(step, batch):
    copy_of = copy.deepcopy(model)
    embedding = copy_of.get_input_embeddings().weight
    embedding.data = embedding.data.half()
    copy_of.base_model.model.store = FakeStore(copy_of)
    trainer = types.SimpleNamespace(
        _prepare_inputs=lambda inputs: inputs, accelerator=types.SimpleNamespace(scaler=Scaler()),
        _metrics={"train": {"entropy": [], "mean_token_accuracy": []}}, _total_train_tokens=0,
        state=types.SimpleNamespace(global_step=0), _soup_rewind_state=Rewind())
    torch.manual_seed(5)
    loss = step(trainer, copy_of, dict(batch))
    return copy_of, loss.item(), trainer, torch.randn(1).item()


def gaps(trained, grads):
    worst, top, total_gap, total_norm = 0.0, "", 0.0, 0.0
    for name, param in trained.named_parameters():
        if param.requires_grad:
            gap = (param.grad - grads[name]).abs().max() / grads[name].abs().max()
            total_gap += (param.grad - grads[name]).pow(2).sum().item()
            total_norm += grads[name].pow(2).sum().item()
            if gap.item() > worst:
                worst, top = gap.item(), name
    return worst, (total_gap / total_norm) ** 0.5, top.split("base_model.model.model.")[-1]


CASES = {"four rows": ([30, 12, 25, 7], None), "three rows": ([20, 41, 9], None),
         "one label": ([14, 33, 8, 26], 1), "one row": ([40], None), "short row": ([31, 2, 17], 2)}

for title, (lengths, one_label) in CASES.items():
    batch = make_batch(lengths, one_label)
    base_loss, base_entropy, base_grads, base_end = reference(batch)
    trained, loss, trainer, end = run(adj.adjoint_step, batch)
    worst, l2, name = gaps(trained, base_grads)
    entropy = trainer._metrics["train"]["entropy"][-1]
    line = (f"{title:11s} loss {loss:.6f}/{base_loss:.6f} entropy {entropy:.6f}/{base_entropy:.6f} "
            f"sweep worst {worst:.2e} l2 {l2:.2e} ({name})")
    print(line)
    assert end == base_end, "the random stream ended somewhere else"
    if MODE == "fp32":
        assert worst < 1e-5 and l2 < 1e-5, title
        assert abs(loss - base_loss) < 1e-5 and abs(entropy - base_entropy) < 1e-5, title
    else:
        assert abs(loss - base_loss) < 2e-2 and abs(entropy - base_entropy) < 2e-2, title
        assert l2 < 2e-3 and worst < 5e-3, title  # v1's group step on these: 1.2e-3, 2.1e-3

# What the sweep must refuse, and that the refusal gets the weights back before the original step.
batch = make_batch([14, 33, 8, 26])
streamed = copy.deepcopy(model)
inner = streamed.base_model.model
inner.get_input_embeddings().weight.data = inner.get_input_embeddings().weight.data.half()
inner.store = FakeStore(streamed)
assert set(streamed.peft_config["default"].target_modules) == {"q_proj", "v_proj"}


def usable(batch=batch, accumulation=1, fp16=True, scaler=Scaler()):
    trainer = types.SimpleNamespace(
        args=types.SimpleNamespace(fp16=fp16, gradient_accumulation_steps=accumulation),
        accelerator=types.SimpleNamespace(scaler=scaler))
    return adj.usable(trainer, streamed, batch)


first_only = dict(batch, labels=torch.full_like(batch["labels"], -100))
first_only["labels"][:, 0] = 1  # the labels shift by one: this token is no target of anything
too_long = {key: value.repeat(1, 100) for key, value in batch.items()}
no_targets = dict(batch, labels=torch.full_like(batch["labels"], -100))
assert usable()
for title, refused in {"accumulation": usable(accumulation=2), "no fp16": usable(fp16=False),
                       "no scaler": usable(scaler=None), "no targets": usable(no_targets),
                       "only a first label": usable(first_only), "too long": usable(too_long),
                       "no labels": usable({"attention_mask": batch["attention_mask"]})}.items():
    assert not refused, title
settings = streamed.peft_config["default"]
for name, value in {"target_modules": {"q_proj"}, "lora_dropout": 0.0, "use_dora": True,
                    "lora_bias": True, "modules_to_save": ["lm_head"]}.items():
    kept = getattr(settings, name)
    setattr(settings, name, value)
    assert not usable(), name
    setattr(settings, name, kept)

# A refusal of the config of this task would turn the whole solution off without a word. It goes
# through Soup and peft as in a run: `auto` only becomes q_proj and v_proj in get_peft_model.
from soup_cli.config.loader import load_config  # noqa: E402
from soup_cli.utils.peft_wiring import build_lora_config, resolve_lora_target_modules  # noqa: E402

task, tiny = load_config(ROOT / "task" / "task.yaml"), Qwen2ForCausalLM(config)
targets = resolve_lora_target_modules(tiny, task.training.lora.target_modules)
built = build_lora_config(task.training.lora, target_modules=targets, task_type=TaskType.CAUSAL_LM)
streamed.peft_config["default"] = get_peft_model(tiny, built).peft_config["default"]
largest = make_batch([task.data.max_length] * task.training.batch_size)
assert usable(largest, task.training.gradient_accumulation_steps), "the config of this task"

import trl  # noqa: E402

calls = inner.store.calls
trl.SFTTrainer.training_step = lambda self, model, inputs, *args, **kwargs: calls.append("original")
trl.SFTTrainer.evaluate = lambda self, *args, **kwargs: calls.append("evaluate")
adj.patch_training_step()
refusing = types.SimpleNamespace(args=types.SimpleNamespace(fp16=False),
                                 accelerator=types.SimpleNamespace(scaler=None))
trl.SFTTrainer.training_step(refusing, streamed, batch)
trl.SFTTrainer.evaluate(types.SimpleNamespace(model=streamed))
assert calls == ["resident", "original", "resident", "evaluate"], calls

# What the trainer no longer walks the model for: the count of its own code, and the tensors of its
# own walk in the same order. A model without a store goes through the Trainer's own code.
import transformers  # noqa: E402

walker = types.SimpleNamespace(model=streamed, args=types.SimpleNamespace(max_grad_norm=1.0))
walker.accelerator = types.SimpleNamespace(clip_grad_norm_=lambda params, limit: list(params))
assert trl.SFTTrainer.floating_point_ops(walker, batch) == \
    transformers.Trainer.floating_point_ops(walker, batch)
walked = [param for param in streamed.parameters() if param.requires_grad]
assert len(walked) == len(adj.trainable(streamed)) and \
    all(a is b for a, b in zip(trl.SFTTrainer._clip_grad_norm(walker, streamed), walked))
for param in walked:
    param.grad = torch.zeros_like(param)
streamed.zero_grad()
assert all(param.grad is None for param in streamed.parameters())
plain = copy.deepcopy(model)
walker.model = plain
assert trl.SFTTrainer.floating_point_ops(walker, batch) == \
    transformers.Trainer.floating_point_ops(walker, batch)
assert len(trl.SFTTrainer._clip_grad_norm(walker, plain)) == len(list(plain.parameters()))

# Reading a batch early changes neither the batches nor their order, and an early one is read once.
ahead = adj.Ahead(iter([1, 2, 3, 4]))
assert next(ahead) == 1
ahead.read()
assert next(ahead) == 2 and ahead.early is None
ahead.read()
assert [*ahead] == [3, 4]

# Only the model of this task keeps its embedding in fp16: any other gets peft's fp32.
adj.patch_kbit_preparation()
for sizes, dtype in (({}, torch.float32), (adj.MODEL, torch.float16)):
    small = Qwen2ForCausalLM(copy.deepcopy(config)).to(torch.float16)
    small.config.update(sizes)
    peft.prepare_model_for_kbit_training(small)
    assert small.get_input_embeddings().weight.dtype == dtype, sizes
print("ok")
