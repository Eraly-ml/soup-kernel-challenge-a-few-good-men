"""Correctness of the kernels of solution/ against plain PyTorch, forward and backward, and of the
order of the copies that carry weights and activations between host and GPU.

Run on a GPU: `python -m pytest solution -q`.
"""

import os
import threading
import time
from types import SimpleNamespace

import pytest
import torch
import triton
import triton.language as tl

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


@pytest.mark.parametrize("tokens", [1, 7, 512])
def test_rms_norm_matches_pytorch(tokens):
    from solution.norm import rms_norm, rms_norm_adjoint

    torch.manual_seed(0)
    x = torch.randn(tokens, 1536, device="cuda", requires_grad=True)
    weight = torch.randn(1536, device="cuda")
    grad = torch.randn(tokens, 1536, device="cuda")
    eps = 1e-6

    ref = weight * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps))
    ref.backward(grad)

    out = rms_norm(x.detach(), weight, eps)
    dx = rms_norm_adjoint(grad, x.detach(), weight, eps)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(dx, x.grad, rtol=1e-5, atol=1e-5)


def test_head_stream_matches_pytorch(monkeypatch):
    """The path of a real step. Everything Soup logs comes from the logits, a plain matmul, so it
    holds to the kernel's own tolerances and does not move when the gradient's matmul stops adding
    up in fp16. That matmul runs as in a step, its rounding bounded here. With plain sums the
    gradient holds to the kernel's tolerances too: that shows the kernel, not the sum, is right."""
    from solution.head import TILE, head_stream

    torch.manual_seed(0)
    vocab, width, tokens = 2 * TILE + 999, 64, 48
    table = (torch.randn(vocab, width, device="cuda") * 0.3).half()
    states = (torch.randn(tokens, width, device="cuda") * 2).half()
    targets = torch.randint(0, vocab, (tokens,), device="cuda", dtype=torch.int32)
    targets[::3] = -100
    targets[1] = (states[1].float() @ table.float().T).argmax()
    scale = 40.0

    class Resident:
        tiles = [table.split(TILE)]

        def tile(self, index):
            return self.tiles[0][index]

        def rows(self, wanted):
            return table[wanted]

    logits = torch.nn.functional.linear(states, table).float()  # fp16, as the kernel sees them
    log_probs = torch.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    wanted = targets >= 0
    ref_loss = torch.zeros(tokens, device="cuda")
    ref_loss[wanted] = -log_probs[wanted, targets[wanted].long()]
    ref_entropy = -(probs * log_probs).sum(-1)
    ref_correct = ((log_probs.argmax(-1) == targets) & wanted).float()
    ref_grad = (probs[wanted] @ table.float() - table[targets[wanted].long()].float()) * scale

    stats, grads = head_stream(states.clone(), targets, scale, Resident())
    monkeypatch.setattr("solution.head.summed_in_fp16", torch.matmul)
    plain_stats, plain_grads = head_stream(states.clone(), targets, scale, Resident())

    torch.testing.assert_close(stats, plain_stats, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(stats[:, 0], ref_loss, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(stats[:, 1], ref_entropy, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(stats[:, 2], ref_correct)
    torch.testing.assert_close(plain_grads[wanted].float(), ref_grad, rtol=5e-3, atol=5e-3)
    assert not torch.equal(grads, plain_grads)  # the fp16 sums did run
    gap = (grads[wanted].float() - ref_grad).norm() / ref_grad.norm()
    assert gap.item() < 0.01  # as test_fp16_sums_hold_for_one_matmul_only
    assert not grads[~wanted].any()


def test_ring_orders_copies_and_reads():
    """A missing event between the two streams gives silent garbage, not an error. So the main
    stream reads each slot behind `enter`, the tail first: a copy runs front to back and is slower
    than a read, so only the tail shows a read that overtook it. Eight more sums keep the main
    stream busy, so that a copy that does not wait for them overwrites the bytes under them. The
    last entries are not announced, and the main stream is empty before them: their copy starts in
    `enter`, and the read right behind it starts at once."""
    from solution.stream import Ring

    torch.manual_seed(0)
    parts = [torch.randint(0, 256, (23 * 2**20 - 4096 * i,), dtype=torch.uint8).pin_memory()
             for i in range(6)]
    expected = [part.sum(dtype=torch.int64).item() for part in parts]
    tails = [part[-4096:].sum(dtype=torch.int64).item() for part in parts]
    ring = Ring(parts, torch.device("cuda", 0))
    order = list(range(6)) + list(range(5, -1, -1))
    entries = [(part, order[step + 1] if step + 1 < len(order) else None)
               for step, part in enumerate(order)] + [(part, None) for part in (3, 0, 4, 1, 5, 2)]
    reads = []
    for part, following in entries:
        if following is None:
            torch.cuda.synchronize()
        ring.enter(part, following)
        slot = ring.slots[ring.active][:parts[part].numel()]
        tail = slot[-4096:].sum(dtype=torch.int64)
        reads.append((part, tail, [slot.sum(dtype=torch.int64) for _ in range(8)]))
    for part, tail, sums in reads:
        assert tail.item() == tails[part]
        assert [value.item() for value in sums] == [expected[part]] * 8


def test_lazy_wandb_gives_the_real_module_and_never_a_hollow_one(monkeypatch):
    """`run` is None without importing anything, the first other attribute imports the library, a
    name bound before that keeps working, and from then on `run` is the library's own."""
    import importlib.util
    import sys

    from solution.startup import Deferred

    monkeypatch.delitem(sys.modules, "colorsys", raising=False)
    lazy = Deferred(importlib.util.find_spec("colorsys"), {"run": None})
    monkeypatch.setitem(sys.modules, "colorsys", lazy)
    import colorsys

    assert colorsys is lazy and colorsys.run is None and sys.modules["colorsys"] is lazy
    assert importlib.util.find_spec("colorsys").name == "colorsys"
    assert colorsys.rgb_to_hsv(1.0, 0.0, 0.0) == (0.0, 1.0, 1.0)
    real = sys.modules["colorsys"]
    assert real is not lazy and real.__file__ == lazy.__file__
    real.run = "a run"
    assert colorsys.run == "a run"


def test_a_stand_in_answers_as_the_real_library():
    """What defer_unused answers for torch._dynamo is what the real one does outside a compile."""
    import torch

    from solution import startup

    def marked():
        return 3

    def plain():
        return 3

    assert startup.assume_constant_result(marked) is marked and marked() == 3
    assert torch._dynamo.assume_constant_result(plain) is plain
    assert marked._dynamo_marked_constant == plain._dynamo_marked_constant
    assert torch._dynamo.graph_break() is None
    assert startup.disabled(plain)() == torch._dynamo.disable(plain)() == 3
    assert startup.disabled()(plain)() == 3
    assert startup.source_of_caller(1) == torch._library.utils.get_source(1)
    from torch.nn.attention.flex_attention import _DEFAULT_SPARSE_BLOCK_SIZE

    assert startup.BLOCK_SIZE == _DEFAULT_SPARSE_BLOCK_SIZE


def test_a_stand_in_takes_the_arguments_of_the_real_one():
    """By name too: a caller is free to write graph_break(msg="...")."""
    import inspect

    import torch._dynamo

    from solution import startup

    for mine, name in ((startup.assume_constant_result, "assume_constant_result"),
                       (startup.allow_in_graph, "allow_in_graph"),
                       (startup.graph_break, "graph_break"), (startup.disabled, "disable")):
        names = [item.name for item in inspect.signature(mine).parameters.values()
                 if item.kind != item.VAR_KEYWORD]
        real = list(inspect.signature(getattr(torch._dynamo, name)).parameters)
        assert real[:len(names)] == names


def test_a_later_is_the_real_thing_for_whatever_is_asked(tmp_path, monkeypatch):
    """A base class, a subclass check, and a member of a family that came in without it."""
    import importlib.util
    import sys

    from solution import startup

    package = tmp_path / "deferred_family" / "sub"
    package.mkdir(parents=True)
    (tmp_path / "deferred_family" / "__init__.py").write_text("")
    (package / "__init__.py").write_text("")
    (package / "part.py").write_text("class Thing:\n    pass\n")
    (package / "other.py").write_text("class Other:\n    pass\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    part, other = "deferred_family.sub.part", "deferred_family.sub.other"
    monkeypatch.setitem(sys.modules, part, startup.Deferred(importlib.util.find_spec(part), {}))
    monkeypatch.setitem(sys.modules, other, startup.Deferred(importlib.util.find_spec(other), {}))
    thing, sibling = startup.Later(part, "Thing"), startup.Later(other, "Other")
    assert not isinstance(3, thing) and not issubclass(int, thing)

    class Mine(thing):
        pass

    assert issubclass(Mine, sys.modules[part].Thing) and isinstance(Mine(), thing)
    assert issubclass(Mine, thing)
    assert other not in sys.modules
    assert not isinstance(3, sibling)
    assert sibling.real().__name__ == "Other"


def test_a_part_that_refuses_attributes_still_imports(tmp_path, monkeypatch):
    """torch._dynamo.config is such a module."""
    import importlib.util
    import sys

    from solution import startup

    package = tmp_path / "deferred_strict"
    package.mkdir()
    (package / "__init__.py").write_text("from . import part\n")
    (package / "part.py").write_text(
        "import sys, types\nclass Strict(types.ModuleType):\n"
        "    def __setattr__(self, name, value):\n        raise AttributeError(name)\n"
        "sys.modules[__name__].__class__ = Strict\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(sys, "meta_path", [startup.PackageFirst(), *sys.meta_path])
    stand_in = startup.Deferred(importlib.util.find_spec("deferred_strict"), {})
    monkeypatch.setitem(sys.modules, "deferred_strict", stand_in)

    import deferred_strict.part as part

    assert part.__spec__.origin.endswith("part.py")


@pytest.mark.parametrize("question", ["""
module = torch.nn.Linear(2, 2)
compiled = torch.compile(module)
assert torch.allclose(compiled(torch.ones(1, 2)), module(torch.ones(1, 2)))
assert isinstance(compiled, torch._dynamo.eval_frame.OptimizedModule)
assert not isinstance(sys.modules["torch._dynamo"], startup.Deferred)
from accelerate.utils.other import extract_model_from_parallel, is_compiled_module
assert is_compiled_module(compiled) and extract_model_from_parallel(compiled) is compiled
assert extract_model_from_parallel(compiled, keep_torch_compile=False) is module
""", """
import torch._dynamo.config as config
assert config.__spec__.origin.endswith("config.py")
""", """
assert callable(torch._dynamo.eval_frame.set_eval_frame)
assert not isinstance(sys.modules["torch._dynamo"], startup.Deferred)
""", """
from torch.nn.attention.flex_attention import BlockMask, create_block_mask
mask = create_block_mask(lambda b, h, q, k: q >= k, 1, 1, 8, 8, device="cpu")
assert isinstance(mask, BlockMask) and type(mask).__name__ == "BlockMask"
from torch.distributed.tensor import DTensor
assert not isinstance(3, DTensor) and DTensor.from_local
import scipy.optimize
from scipy.optimize import linear_sum_assignment
assert linear_sum_assignment([[1, 2], [3, 1]])[1].tolist() == [0, 1]
assert callable(scipy.optimize.minimize)
import wandb
assert wandb.run is None and callable(wandb.init)
"""], ids=["torch_compile", "a_part_first", "more_of_a_part", "the_others"])
def test_a_config_that_uses_a_deferred_library_gets_the_real_one(question):
    """In a process of its own, on this machine's torch: the first thing asked brings it in."""
    import os
    import subprocess
    import sys
    from pathlib import Path

    script = f"import sys, torch\nfrom solution import startup\nstartup.defer_unused()\n{question}"
    done = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                          cwd=Path(__file__).resolve().parents[1],
                          env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
    assert done.returncode == 0, done.stderr[-1500:]


def test_the_collector_comes_back_on_any_trainer(monkeypatch):
    """Off until the first step: a run that never reaches adjoint_step has one too."""
    import gc

    from transformers import Trainer

    from solution import startup

    monkeypatch.setattr(Trainer, "training_step", lambda self, *args: "stepped")
    monkeypatch.setattr(startup, "collecting", False)
    monkeypatch.setattr(startup, "compiling", None)
    startup.patch_first_step()
    gc.disable()
    try:
        assert Trainer.training_step(None) == "stepped"
        assert gc.isenabled()
    finally:
        gc.enable()


def test_nothing_on_meta_draws_or_changes_what_is_not_on_meta(monkeypatch):
    """No random number goes with the fill on meta, so the stream of a run does not move, and the
    frequencies are the same off meta."""
    from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
    from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding

    from solution import startup

    config = Qwen2Config(hidden_size=64, num_attention_heads=4)
    monkeypatch.setattr(torch.nn.Embedding, "reset_parameters", torch.nn.Embedding.reset_parameters)
    monkeypatch.setattr(Qwen2RotaryEmbedding, "compute_default_rope_parameters",
                        Qwen2RotaryEmbedding.__dict__["compute_default_rope_parameters"])
    torch.manual_seed(0)
    state = torch.get_rng_state()
    with torch.device("meta"):
        torch.nn.Embedding(8, 4)
    assert torch.equal(torch.get_rng_state(), state), "the plain fill on meta draws nothing"
    plain = torch.nn.Embedding(8, 4).weight.detach().clone()
    frequencies = Qwen2RotaryEmbedding(config).inv_freq.clone()
    startup.patch_values_on_meta()
    torch.manual_seed(0)
    with torch.device("meta"):
        assert torch.nn.Embedding(8, 4).weight.is_meta
        assert torch.equal(Qwen2RotaryEmbedding(config).inv_freq, frequencies)
    assert torch.equal(torch.get_rng_state(), state)
    assert torch.equal(torch.nn.Embedding(8, 4).weight, plain)
    assert torch.equal(Qwen2RotaryEmbedding(config).inv_freq, frequencies)


def test_importing_a_part_of_a_deferred_library_brings_the_real_one_once(tmp_path, monkeypatch):
    """`import a.b` under a stand-in: the real `a` runs, `a.b` runs once whoever imports it first,
    and its __spec__ stays its own. A name taken at import keeps working: Later."""
    import importlib.util
    import sys

    from solution import startup

    package = tmp_path / "deferred_pkg"
    package.mkdir()
    (package / "__init__.py").write_text("runs = 1\nfrom . import part\nfrom .part import twice\n")
    (package / "part.py").write_text("import builtins\nbuiltins.part_runs += 1\n"
                                     "def twice(x):\n    return 2 * x\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr("builtins.part_runs", 0, raising=False)
    monkeypatch.setattr(sys, "meta_path", [startup.PackageFirst(), *sys.meta_path])
    for name in ("deferred_pkg", "deferred_pkg.part"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    spec = importlib.util.find_spec("deferred_pkg")
    stand_in = startup.Deferred(spec, {"answer": 42})
    monkeypatch.setitem(sys.modules, "deferred_pkg", stand_in)

    from deferred_pkg import answer

    assert answer == 42 and sys.modules["deferred_pkg"] is stand_in
    later = startup.Later("deferred_pkg", "twice")
    assert not isinstance(3, later) and sys.modules["deferred_pkg"] is stand_in
    assert later(4) == 8
    real = sys.modules["deferred_pkg"]
    assert real is not stand_in and real.runs == 1 and sys.modules["builtins"].part_runs == 1

    came = []
    monkeypatch.setitem(sys.modules, "deferred_pkg", startup.Deferred(spec, {}, came.append))
    del sys.modules["deferred_pkg.part"]
    import deferred_pkg.part as part

    assert came == [sys.modules["deferred_pkg"]]
    assert part.twice(2) == 4 and sys.modules["builtins"].part_runs == 2
    assert part.__spec__.origin.endswith("part.py") and part.__spec__.name == "deferred_pkg.part"
    assert sys.modules["deferred_pkg"].runs == 1


def test_pinned_blocks_hold_a_step():
    """Weights, the vocabulary table, and all that parks for the largest batch, in the order of
    Store: a piece that fits the total can still find no block with room for it."""
    from solution.parking import PIECES
    from solution.startup import PINNED
    from solution.stream import MAX_TOKENS, MODEL, aligned, take

    layers, hidden, middle = (MODEL[key] for key in (
        "num_hidden_layers", "hidden_size", "intermediate_size"))
    values = hidden * (hidden + 2 * 256 + hidden + 2 * middle + middle)  # Q, K, V, O, G, U, D
    waiting = layers - 1
    table = MODEL["vocab_size"] * hidden * 2
    rows = ((MAX_TOKENS * 2 * middle * 2,) * waiting + (PIECES * MAX_TOKENS * hidden * 2,) * waiting
            + (MAX_TOKENS * hidden * 2,) * waiting)
    bits = (MAX_TOKENS * (hidden // 32) * 4,) * 2 * waiting
    weights = (values // 2 + values // 64 + 14 * 64,) * layers
    blocks = [torch.empty(size, dtype=torch.uint8, device="meta") for size in PINNED]
    for size in (table, *rows, *bits, *weights):
        take(blocks, aligned(size))
    assert max(PINNED) <= 2 ** 29


def test_store_follows_the_slots_and_gives_the_weights_back(monkeypatch):
    """Each entry of a layer lands in the other slot of the ring, so a decode reads there and
    nowhere else. A step the sweep refuses gets every weight on the GPU, and the next sweep takes
    them back to the slots, and the vocabulary table to the host."""
    from solution import stream
    from solution.nf4 import Plan
    from solution.parking import PIECES

    torch.manual_seed(0)
    monkeypatch.setattr(stream, "pinned_memory", lambda: [torch.empty(
        2 ** 22, dtype=torch.uint8, pin_memory=True)])
    names = "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
    pairs = [[nf4_weight((64, 128)) for _ in names] for _ in range(3)]
    modules = [{name: SimpleNamespace(weight=weight) for name, (weight, _) in zip(names, pair)}
               for pair in pairs]
    table = torch.nn.Parameter((torch.randn(stream.TILE + 10, 32, device="cuda")).half(),
                               requires_grad=False)
    values = table.data.clone()
    layers = [SimpleNamespace(self_attn=SimpleNamespace(**{name: layer[name] for name in names[:4]}),
                              mlp=SimpleNamespace(**{name: layer[name] for name in names[4:]}))
              for layer in modules]
    inner = SimpleNamespace(config=SimpleNamespace(hidden_size=32, intermediate_size=16),
                            model=SimpleNamespace(layers=layers),
                            lm_head=SimpleNamespace(weight=table))
    store = stream.Store(inner, torch.device("cuda", 0))
    both, states, bits, attention = store.parked
    assert [tuple(host.shape) for host in (*both, *states)] == [(stream.MAX_TOKENS, 32)] * 4
    assert [tuple(host.shape) for site in bits for host in site] == [(stream.MAX_TOKENS, 1)] * 4
    assert [tuple(host.shape) for host in attention] == [(PIECES * stream.MAX_TOKENS, 32)] * 2
    plans = [Plan([[weight for weight, _ in pair]]) for pair in pairs]

    def check(index):
        references = torch.cat([reference for _, reference in pairs[index]])
        assert torch.equal(plans[index].decode()[0], references)

    for index, following in [(0, 1), (1, 2), (2, 1), (1, 0), (0, None), (2, None), (1, None)]:
        store.layer(index, following)
        check(index)
    assert not table.is_cuda

    store.resident()
    assert table.is_cuda and torch.equal(table.data, values)
    assert all(weight.is_cuda and weight.quant_state.absmax.is_cuda
               for pair in pairs for weight, _ in pair)
    for index in range(3):
        check(index)

    store.layer(0, 1)
    check(0)
    assert not store.on_gpu and not table.is_cuda and pairs[0][0][0].data.data_ptr() in {
        slot.data_ptr() for slot in store.ring.slots}


def nf4_weight(shape):
    import bitsandbytes as bnb

    dense = (torch.randn(shape, device="cuda") * 0.02).half()
    packed, state = bnb.functional.quantize_4bit(dense, quant_type="nf4", compress_statistics=True)
    weight = bnb.nn.Params4bit(packed, quant_state=state, quant_type="nf4")
    return weight, bnb.functional.dequantize_4bit(packed, state)


@pytest.mark.parametrize("shape", [(8960, 1536), (1536, 8960), (256, 1536), (37, 11)])
def test_nf4_decode_equals_bitsandbytes(shape):
    from solution.nf4 import Plan

    torch.manual_seed(0)
    weight, reference = nf4_weight(shape)
    assert torch.equal(Plan([[weight]]).decode()[0], reference)


def test_nf4_plan_decodes_a_layer_at_once():
    """Seven weights of a decoder layer in one launch, those with one input stacked by rows."""
    from solution.nf4 import Plan

    torch.manual_seed(0)
    shapes = [[(1536, 1536), (256, 1536), (256, 1536)], [(1536, 1536)],
              [(8960, 1536), (8960, 1536)], [(1536, 8960)]]
    pairs = [[nf4_weight(shape) for shape in group] for group in shapes]
    plan = Plan([[weight for weight, _ in group] for group in pairs])
    for _ in range(2):
        for dense, group in zip(plan.decode(), pairs):
            assert torch.equal(dense, torch.cat([reference for _, reference in group]))


def test_nf4_columns_are_the_same_values_the_other_way_round():
    """A layer in the form [in, out], bit for bit. A plan in the buffer stays as it lies, and a
    weight that squares of a block do not cover comes out by rows."""
    from solution import nf4

    torch.manual_seed(0)
    shapes = [[(1536, 1536), (256, 1536), (256, 1536)], [(1536, 1536)],
              [(8960, 1536), (8960, 1536)], [(1536, 8960)]]
    pairs = [[nf4_weight(shape) for shape in group] for group in shapes]
    plan = nf4.Plan([[weight for weight, _ in group] for group in pairs])
    nf4.held = None
    for dense, group in zip(plan.decode(columns=True), pairs):
        assert not dense.is_contiguous() and dense.t().is_contiguous()
        assert torch.equal(dense, torch.cat([reference for _, reference in group]))
    for dense, group in zip(plan.decode(), pairs):
        assert torch.equal(dense, torch.cat([reference for _, reference in group]))
    nf4.held = None
    for dense, group in zip(plan.decode(), pairs):
        assert dense.is_contiguous()
    other, reference = nf4_weight((37, 11))
    assert torch.equal(nf4.Plan([[other]]).decode(columns=True)[0], reference)


def test_nf4_plan_follows_its_weights():
    """Streamed weights sit in one of two slots: a plan reads them where they are now."""
    from solution import nf4

    torch.manual_seed(0)
    weight, reference = nf4_weight((256, 1536))
    state = weight.quant_state
    plan = nf4.Plan([[weight]])
    assert torch.equal(plan.decode()[0], reference)
    moved, moved_absmax = weight.data.clone(), state.absmax.clone()
    weight.data, state.absmax = moved, moved_absmax
    nf4.held = None
    assert torch.equal(plan.decode()[0], reference)
    weight.data.zero_(), state.absmax.zero_()
    nf4.held = None
    assert not torch.equal(plan.decode()[0], reference)


def test_nf4_plan_in_the_buffer_is_not_decoded_again():
    """Frozen weights: a plan still in the buffer keeps its values, and the launch is skipped. The
    weights are wiped to tell the two apart, which a run never does: a skipped launch still gives
    the right values, and a launch after another plan reads the wiped weights."""
    from solution.nf4 import Plan

    torch.manual_seed(0)
    (first, first_ref), (second, second_ref) = (nf4_weight((256, 1536)) for _ in range(2))
    plans = Plan([[first]]), Plan([[second]])
    assert torch.equal(plans[0].decode()[0], first_ref)
    first.data.zero_(), first.quant_state.absmax.zero_()
    assert torch.equal(plans[0].decode()[0], first_ref)
    assert torch.equal(plans[1].decode()[0], second_ref)
    assert not torch.equal(plans[0].decode()[0], first_ref)


def test_nf4_rough_decode_rounds_and_is_for_one_use():
    """Four mantissa bits, to nearest, on the values of the exact decode; the next exact decode of
    the same plan does not take them for its own."""
    from solution.nf4 import Plan

    torch.manual_seed(0)
    weight, reference = nf4_weight((256, 1536))
    plan = Plan([[weight]])
    plan.decode()
    rough = plan.rough()[0].clone()
    assert not (rough.view(torch.int16) & 63).any()
    torch.testing.assert_close(rough.float(), reference.float(), rtol=1 / 31, atol=1e-7)
    assert torch.equal(plan.decode()[0], reference)


from solution.rough import rough as solution_rough  # noqa: E402


@triton.jit
def every_pattern_kernel(in_ptr, out_ptr, CHUNK: tl.constexpr):
    from_here = tl.program_id(0) * CHUNK + tl.arange(0, CHUNK)
    tl.store(out_ptr + from_here, solution_rough(tl.load(in_ptr + from_here)))


def test_rough_holds_on_every_pattern_of_fp16():
    """Nothing finite turns into inf, a NaN stays a NaN, and the rest rounds to nearest, unbiased."""
    patterns = torch.arange(-32768, 32768, dtype=torch.int16, device="cuda")
    values = patterns.view(torch.float16)
    out = torch.empty_like(values)
    every_pattern_kernel[(256,)](values, out, CHUNK=256)
    bits = out.view(torch.int16)
    finite, plain = values.isfinite(), values.abs() < 64512
    assert out[finite].isfinite().all()
    assert out[values.isnan()].isnan().all()
    assert torch.equal(out[values.isinf()], values[values.isinf()])
    assert not (bits[plain] & 63).any()
    moved = (bits.int() - patterns.int())[plain].float()
    assert moved.abs().max() <= 32
    assert moved.mean().abs() < 0.05


def test_a_nan_or_an_inf_in_the_backward_reaches_the_scaler():
    """Through a real kernel, with whichever NaN this GPU's conversion to fp16 writes: a zero here
    would hide from GradScaler a step that base Soup skips."""
    from solution.fused import mlp_adjoint

    grad = torch.ones(2, 16, dtype=torch.float16, device="cuda")
    grad[0, 0], grad[0, 1] = float("nan"), float("inf")
    both = torch.ones(2, 32, dtype=torch.float16, device="cuda")
    out = mlp_adjoint(grad, both)
    assert out[0, 0].isnan() and out[0, 16].isnan()
    assert out[0, 1].isinf() and out[0, 17].isinf()
    assert out[1].isfinite().all() and (out[1] != 0).all()


def test_nf4_plan_refuses_more_weights_than_slots():
    """Past SLOTS the table of a plan would lose its shape without a word."""
    from solution.nf4 import SLOTS, Plan

    weight, _ = nf4_weight((37, 11))
    Plan([[weight] * SLOTS])
    with pytest.raises(AssertionError):
        Plan([[weight] * (SLOTS + 1)])


def test_parking_keeps_what_it_is_given():
    """Copies on their own stream, with compute on the main stream between them."""
    from solution.parking import PIECES, Parking

    torch.manual_seed(0)
    layers, tokens = 3, 300
    hosts = (torch.empty(layers, 512, 64, dtype=torch.float16, pin_memory=True),
             torch.empty(layers, 512, 32, dtype=torch.float16, pin_memory=True),
             torch.empty(2, layers, 512, 4, dtype=torch.int32, pin_memory=True),
             torch.empty(layers, PIECES * 512, 16, dtype=torch.float16, pin_memory=True))
    parked = Parking(*hosts, torch.device("cuda"))
    parked.start_step(tokens)
    both = [torch.randn(tokens, 64, device="cuda").half() for _ in range(layers)]
    x = [torch.randn(tokens, 32, device="cuda").half() for _ in range(layers)]
    bits = [torch.randint(-5, 5, (2, tokens, 4), device="cuda", dtype=torch.int32).unbind()
            for _ in range(layers)]
    attention = [torch.randn(PIECES * tokens, 16, device="cuda").half() for _ in range(layers)]
    for layer in range(layers):
        slot_x, slot_bits, slot_attention = parked.slots(layer)
        slot_x.copy_(x[layer])
        for slot, value in zip(slot_bits, bits[layer]):
            slot.copy_(value)
        slot_attention.copy_(attention[layer])
        parked.wait()
        parked.both.copy_(both[layer])
        parked.send(layer)
    for layer in reversed(range(layers)):
        parked.fetch(layer)
        slot_x, slot_bits, slot_attention = parked.inputs(layer)
        assert torch.equal(slot_x, x[layer])
        assert all(map(torch.equal, slot_bits, bits[layer]))
        assert torch.equal(slot_attention, attention[layer])
        parked.wait()
        assert torch.equal(parked.both, both[layer])
    parked.end_step()


def test_no_idle_stacks_and_a_kernel_still_runs():
    """The limit says what to keep, and a matmul and an attention run behind it as they did."""
    import ctypes

    from solution.startup import CU_LIMIT_STACK_SIZE, no_idle_stacks

    torch.zeros(1, device="cuda")  # the limit belongs to a context, and this makes the first one
    driver = ctypes.CDLL("libcuda.so.1")
    value = ctypes.c_size_t()
    no_idle_stacks()
    assert driver.cuCtxGetLimit(ctypes.byref(value), ctypes.c_int(CU_LIMIT_STACK_SIZE)) == 0
    assert value.value == 0
    a = torch.randn(32, 64, device="cuda").half()
    assert torch.isfinite(torch.matmul(a, a.t())).all()
    x = torch.randn(1, 12, 16, 128, device="cuda").half()
    out = torch.nn.functional.scaled_dot_product_attention(x, x, x, is_causal=True)
    assert torch.isfinite(out).all()
    driver.cuCtxGetLimit(ctypes.byref(value), ctypes.c_int(CU_LIMIT_STACK_SIZE))
    assert value.value < 1024


def test_a_plain_stream_keeps_its_order_and_its_events():
    """A copy behind a wait on the main stream sees what the main stream wrote."""
    from solution.startup import own_stream

    device = torch.device("cuda")
    torch.zeros(1, device=device)  # without a context the plain stream cannot be made
    stream = own_stream(device)
    main = torch.cuda.current_stream(device)
    assert isinstance(stream, torch.cuda.ExternalStream)
    assert stream.cuda_stream != main.cuda_stream
    host = torch.zeros(2 ** 20, dtype=torch.uint8).pin_memory()
    target = torch.zeros(2 ** 20, dtype=torch.uint8, device=device)
    target.fill_(7)
    with torch.cuda.stream(stream):
        stream.wait_stream(main)
        host.copy_(target, non_blocking=True)
        done = stream.record_event()
    main.wait_event(done)
    torch.cuda.synchronize()
    assert int(host.sum()) == 7 * 2 ** 20


def tiny_checkpoint(tmp_path, monkeypatch, dtype=torch.bfloat16):
    """A small model.safetensors where ahead.py looks for it, and ahead.py's state before a read."""
    import huggingface_hub
    from safetensors.torch import save_file

    from solution import ahead, startup

    torch.manual_seed(0)
    names = ("mlp.down_proj", "mlp.gate_proj", "mlp.up_proj", "self_attn.k_proj", "self_attn.q_proj")
    tensors = {ahead.EMBEDDING: torch.randn(70, 16).to(dtype)}
    for layer in (0, 1, 10):
        tensors[f"model.layers.{layer}.input_layernorm.weight"] = torch.randn(16).to(dtype)
        for index, name in enumerate(names):
            tensors[f"model.layers.{layer}.{name}.weight"] = \
                (torch.randn(32 + 16 * index, 16) * 0.1).to(dtype)
    path = tmp_path / "model.safetensors"
    save_file(tensors, str(path))
    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", lambda *args: str(path))
    monkeypatch.setattr(startup, "whole", torch.empty(2 ** 20, dtype=torch.uint8, pin_memory=True))
    monkeypatch.setattr(startup, "pinning", SimpleNamespace(join=lambda: None,
                                                            is_alive=lambda: True))
    monkeypatch.setattr(ahead, "ready", {})
    monkeypatch.setattr(ahead, "embedding", None)
    monkeypatch.setattr(ahead, "ready_file", None)
    startup.quantizable.set()
    return str(path), tensors


def test_read_ahead_makes_the_bytes_of_the_load(tmp_path, monkeypatch):
    import bitsandbytes as bnb

    from solution import ahead

    path, tensors = tiny_checkpoint(tmp_path, monkeypatch)
    ahead.read_and_quantize()

    assert ahead.ready_file == path and ahead.embedding.dtype == torch.float16
    assert torch.equal(ahead.embedding, tensors[ahead.EMBEDDING].to(torch.float16))
    assert len(ahead.ready) == 15
    for name, (packed, state) in ahead.ready.items():
        expected, known = bnb.functional.quantize_4bit(tensors[name].to("cuda", torch.float16),
                                                       **ahead.QUANTIZED)
        assert torch.equal(packed.cuda(), expected)
        assert torch.equal(state.absmax.cuda(), known.absmax)
        assert torch.equal(state.offset, known.offset)
        assert torch.equal(state.code, known.code) and state.code.dtype == known.code.dtype
        assert torch.equal(state.state2.absmax, known.state2.absmax)
        assert state.shape == known.shape and state.dtype == known.dtype


@pytest.mark.parametrize("short", [None, 0, 1, 7])
def test_a_short_read_leaves_the_load_to_go_plain(tmp_path, monkeypatch, short):
    """Read 0 is the embedding, the others are weights. None is the control: a clean read is taken,
    so the others go plain because they were short, not because the reader failed anyway."""
    from solution import ahead, startup

    path, _ = tiny_checkpoint(tmp_path, monkeypatch)
    read, reads = ahead.os.preadv, []

    def preadv(fd, buffers, offset):
        done = read(fd, buffers, offset)
        reads.append(offset)
        return done // 2 if len(reads) - 1 == short else done

    monkeypatch.setattr(ahead.os, "preadv", preadv)
    startup.read_ahead()
    assert ahead.same_file([path]) is (short is None)


def test_a_table_made_in_fp16_leaves_the_load_to_go_plain(tmp_path, monkeypatch):
    """A read that ends inside from_pretrained, where the default dtype is fp16. The kernel decodes
    from this table, so about 21% of the values of a layer would change without a word."""
    from solution import ahead, startup

    path, _ = tiny_checkpoint(tmp_path, monkeypatch)
    torch.set_default_dtype(torch.float16)
    try:
        startup.read_ahead()
    finally:
        torch.set_default_dtype(torch.float32)
    assert not ahead.same_file([path]) and not ahead.ready


@pytest.mark.parametrize("dtype, mapped", [(torch.float16, None), (torch.bfloat16, 2 ** 10)])
def test_a_checkpoint_the_read_cannot_hold_loads_plain(tmp_path, monkeypatch, dtype, mapped):
    """Stored in another dtype, it would be viewed as bf16. Larger than the mapping, touching the
    rest of the mapping would get a negative count."""
    from solution import ahead, startup

    path, _ = tiny_checkpoint(tmp_path, monkeypatch, dtype)
    if mapped:
        monkeypatch.setattr(startup, "PINNED", (mapped,))
    startup.read_ahead()
    assert not ahead.same_file([path])


def memory_to_pin(monkeypatch, code):
    """startup before pin_ahead, with the driver answering `code`."""
    from solution import startup

    monkeypatch.setattr(startup, "PINNED", (2 ** 21, 2 ** 22))
    monkeypatch.setattr(startup, "register", lambda address, size: code)
    monkeypatch.setattr(startup, "own_bytes", 0)
    for name in ("whole", "pinned", "register_code"):
        monkeypatch.setattr(startup, name, None)
    startup.claimed.set()
    startup.filled.set()
    return startup


def test_memory_that_no_one_pinned_is_never_handed_out(monkeypatch):
    """The driver refuses and the old way fails too: Store must crash, not run on the mapping."""
    startup = memory_to_pin(monkeypatch, 1)

    class NoPin:
        def __getattr__(self, name):
            return getattr(torch, name)

        def empty(self, *args, **kwargs):
            raise RuntimeError("no pinned memory")

    monkeypatch.setattr(startup, "torch", NoPin())
    with pytest.raises(RuntimeError):
        startup.pin_ahead()
    assert startup.pinned is None


def test_a_kernel_without_huge_pages_only_costs_time(monkeypatch):
    import errno
    import mmap

    startup = memory_to_pin(monkeypatch, 0)

    class NoHugePages(mmap.mmap):
        def madvise(self, *args):
            raise OSError(errno.EINVAL, "no transparent huge pages")

    monkeypatch.setattr(startup.mmap, "mmap", NoHugePages)
    startup.pin_ahead()
    assert [block.numel() for block in startup.pinned] == [2 ** 21, 2 ** 22]


def test_mapped_memory_is_registered_and_carries_copies(monkeypatch):
    from solution import startup

    torch.empty(1, device="cuda")
    monkeypatch.setattr(startup, "PINNED", (2 ** 21, 2 ** 22))
    monkeypatch.setattr(startup, "context", startup.current_context())
    monkeypatch.setattr(startup, "own_bytes", 0)
    for name in ("whole", "pinned", "register_code"):
        monkeypatch.setattr(startup, name, None)
    startup.claimed.set()
    startup.filled.set()
    startup.pin_ahead()

    assert startup.register_code == 0
    first, second = startup.pinned
    assert (first.numel(), second.numel()) == (2 ** 21, 2 ** 22)
    assert first.data_ptr() + 2 ** 21 == second.data_ptr() == startup.whole.data_ptr() + 2 ** 21
    values = torch.randint(0, 256, (2 ** 21,), dtype=torch.uint8, device="cuda")
    first.copy_(values, non_blocking=True)
    torch.cuda.synchronize()
    back = first.to("cuda", non_blocking=True)
    assert torch.equal(back, values)


def test_the_table_of_a_weight_does_not_follow_the_default_dtype():
    """What verify found in 10 of 196 weights, once they were quantized beside from_pretrained."""
    import bitsandbytes as bnb

    torch.set_default_dtype(torch.float16)
    try:
        table = bnb.functional.get_4bit_type("nf4", device="cuda")
    finally:
        torch.set_default_dtype(torch.float32)
    assert table.dtype == torch.float16  # why ahead.read_and_quantize refuses such a table
    assert bnb.functional.get_4bit_type("nf4", device="cuda").dtype == torch.float32


def test_row_labels_are_those_of_soups_four_tokenizations():
    from soup_cli.data import loss_mask

    from solution import startup

    class Tokenizer:
        chat_template = "{% if add_generation_prompt %}{% endif %}"

        def __init__(self):
            self.calls = 0

        def apply_chat_template(self, messages, tokenize=True, return_dict=False,
                                return_assistant_tokens_mask=False, **kwargs):
            self.calls += 1
            ids = [ord(letter) for message in messages for letter in message["content"] + "|"]
            if return_dict:
                return {"input_ids": ids,
                        "assistant_masks": [0] * len(ids)} if return_assistant_tokens_mask \
                    else {"input_ids": ids}
            return ids

    rows = [[{"role": "user", "content": "ab" * index}, {"role": "assistant", "content": "cd"}]
            for index in range(1, 4)]
    tokenizer = Tokenizer()
    plain = [loss_mask.build_assistant_only_labels(row, tokenizer) for row in rows]
    slow = tokenizer.calls
    attempt, tokenize = loss_mask._apply_template_with_mask, loss_mask._tokenize_only
    try:
        startup.patch_row_labels()
        tokenizer.calls = 0
        assert [loss_mask.build_assistant_only_labels(row, tokenizer) for row in rows] == plain
        assert tokenizer.calls == slow - 2 * len(rows)
    finally:
        loss_mask._apply_template_with_mask, loss_mask._tokenize_only = attempt, tokenize


@pytest.mark.parametrize("template, asked", [
    ("{% generation %}a{% endgeneration %}", True), ("{%generation%}", True),
    ("{%-generation-%}", True), ("{% generation-%}", True), ("{%+ generation %}", True), ("", True),
    ("{% if add_generation_prompt %}{% endif %}", False)])
def test_row_labels_ask_soup_for_the_mask_on_any_spelling_of_the_tag(monkeypatch, template, asked):
    from soup_cli.data import loss_mask

    from solution import startup

    seen = []
    monkeypatch.setattr(loss_mask, "_apply_template_with_mask", lambda *args: seen.append(args))
    monkeypatch.setattr(loss_mask, "_tokenize_only", lambda *args: None)
    startup.patch_row_labels()
    tokenizer = SimpleNamespace(chat_template=template)
    loss_mask._apply_template_with_mask(tokenizer, [{"role": "assistant", "content": "a"}])
    assert bool(seen) is asked


def test_row_label_memo_knows_what_it_rendered(monkeypatch):
    import gc
    import weakref

    from soup_cli.data import loss_mask

    from solution import startup

    class Tokenizer:
        chat_template = "one"

    calls = []
    monkeypatch.setattr(loss_mask, "_tokenize_only", lambda tokenizer, messages: calls.append(
        tokenizer.chat_template) or [len(calls)])
    monkeypatch.setattr(loss_mask, "_apply_template_with_mask", lambda *args: None)
    startup.patch_row_labels()
    tokenizer, row = Tokenizer(), [{"role": "user", "content": "a"}]
    first = loss_mask._tokenize_only(tokenizer, row)
    assert loss_mask._tokenize_only(tokenizer, row) == first and len(calls) == 1
    tokenizer.chat_template = "two"
    assert loss_mask._tokenize_only(tokenizer, row) != first  # the template changed the answer
    held = weakref.ref(tokenizer)
    del tokenizer
    gc.collect()
    assert held() is not None  # so no new tokenizer can come to have its id


def real_tokenizer():
    """A real fast tokenizer under the name of the task's, with a token to change."""
    from tokenizers import AddedToken, Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    from solution.ahead import CHECKPOINT

    backend = Tokenizer(models.WordLevel({"a": 0, "b": 1, "[UNK]": 2}, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    backend.add_tokens([AddedToken("<x>", special=False)])
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]")
    tokenizer.name_or_path = CHECKPOINT
    return tokenizer


class Saves:
    """patch_tokenizer_file over a counted save, with what the tokenizer itself would write."""

    def __init__(self, tmp_path, monkeypatch, tokenizer):
        from transformers.tokenization_utils_tokenizers import TokenizersBackend

        from solution import startup

        real, self.real_saves = TokenizersBackend._save_pretrained, []

        def counted(tokenizer, *args, **kwargs):
            self.real_saves.append(1)
            return real(tokenizer, *args, **kwargs)

        monkeypatch.setattr(TokenizersBackend, "_save_pretrained", counted)
        startup.patch_tokenizer_file()
        self.tmp_path, self.tokenizer = tmp_path, tokenizer

    def save(self, name):
        (self.tmp_path / name).mkdir(exist_ok=True)
        self.tokenizer.save_pretrained(self.tmp_path / name)
        return (self.tmp_path / name / "tokenizer.json").read_bytes()

    def truth(self):
        self.tokenizer.backend_tokenizer.save(str(self.tmp_path / "truth.json"))
        return (self.tmp_path / "truth.json").read_bytes()


def test_tokenizer_file_is_serialized_once_while_nothing_changes(tmp_path, monkeypatch):
    saves = Saves(tmp_path, monkeypatch, real_tokenizer())
    assert saves.save("checkpoint") == saves.save("output") == saves.save("output") == saves.truth()
    assert len(saves.real_saves) == 1
    (tmp_path / "output" / "tokenizer.json").unlink()
    assert saves.save("output") == saves.truth()  # a copy that went comes back
    assert len(saves.real_saves) == 1


@pytest.mark.parametrize("change", [
    "truncation", "normalizer", "pre_tokenizer", "post_processor", "decoder", "flags", "special"])
def test_tokenizer_file_follows_every_part_it_holds(tmp_path, monkeypatch, change):
    """All QA found going stale. Each change must reach the file, or the first copy comes back."""
    from tokenizers import AddedToken, decoders, normalizers, pre_tokenizers, processors

    tokenizer = real_tokenizer()
    saves = Saves(tmp_path, monkeypatch, tokenizer)
    before = saves.save("checkpoint")
    backend = tokenizer.backend_tokenizer
    if change == "truncation":
        tokenizer("a b a", truncation=True, max_length=2)
    elif change == "flags":
        backend.add_tokens([AddedToken("<x>", lstrip=True)])
    elif change == "special":
        backend.add_special_tokens([AddedToken("<x>", special=True)])
    else:
        setattr(backend, change, {"normalizer": normalizers.Lowercase,
                                  "pre_tokenizer": pre_tokenizers.WhitespaceSplit,
                                  "post_processor": processors.ByteLevel,
                                  "decoder": decoders.ByteLevel}[change]())
    assert saves.truth() != before  # the change is one the file holds
    assert saves.save("after") == saves.truth()
    assert len(saves.real_saves) == 2


def test_tokenizer_file_altered_by_anyone_else_is_not_copied(tmp_path, monkeypatch):
    saves = Saves(tmp_path, monkeypatch, real_tokenizer())
    saves.save("checkpoint")
    (tmp_path / "checkpoint" / "tokenizer.json").write_text("{}")
    assert saves.save("output") == saves.truth()
    (tmp_path / "output" / "tokenizer.json").unlink()
    assert saves.save("output") == saves.truth()  # the first copy gone is the same case
    assert len(saves.real_saves) == 3


def test_tokenizer_of_any_other_model_saves_as_it_always_did(tmp_path, monkeypatch):
    tokenizer = real_tokenizer()
    tokenizer.name_or_path = "some/other-model"
    saves = Saves(tmp_path, monkeypatch, tokenizer)
    saves.save("checkpoint")
    saves.save("output")
    assert len(saves.real_saves) == 2


def test_reading_ahead_ends_with_the_epoch():
    """With two epochs the last batch of the first has none behind it, and a StopIteration out of
    a step would end the run."""
    from solution.adjoint import Ahead

    ahead = Ahead(iter([1]))
    assert next(ahead) == 1
    ahead.read()
    assert [*ahead] == []


def test_the_walks_are_the_trainers_unless_it_is_the_one_lora_of_this_task(monkeypatch):
    from solution import adjoint

    settings = SimpleNamespace(target_modules={"q_proj", "v_proj"}, lora_dropout=0.05,
                               use_dora=False, lora_bias=False, modules_to_save=None)
    model = SimpleNamespace(peft_config={"default": settings})
    monkeypatch.setattr(adjoint, "store_of", lambda model: object())
    assert adjoint.plain_lora(model)
    model.peft_config["other"] = settings  # a second adapter moves requires_grad during a run
    assert not adjoint.plain_lora(model)


def test_waits_go_from_soups_tracker_and_no_other(tmp_path):
    import sqlite3

    from solution import adjoint

    connection = sqlite3.connect(tmp_path / "experiments.db")
    connection.execute("PRAGMA journal_mode=WAL")
    assert connection.execute("PRAGMA synchronous").fetchone() == (2,)
    soups, other = SimpleNamespace(_get_conn=lambda: connection), SimpleNamespace(start=None)
    trainer = SimpleNamespace(
        accelerator=SimpleNamespace(_dataloaders=[]),
        callback_handler=SimpleNamespace(callbacks=[SimpleNamespace(tracker=other),
                                                    SimpleNamespace(tracker=soups),
                                                    SimpleNamespace(tracker=None)]))
    adjoint.no_waits_on_the_host(trainer)
    assert connection.execute("PRAGMA synchronous").fetchone() == (1,)


@pytest.mark.parametrize("cpus, away", [
    ({0, 1, 2, 3}, {1, 3}), ({0, 1}, set()), ({0}, set()), (set(range(8)), set()),
    ({1, 2, 3, 4}, set())])
def test_threads_move_on_the_box_of_this_task_and_no_other(monkeypatch, cpus, away):
    from solution import startup

    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: cpus, raising=False)
    assert startup.other_core() == away


def test_a_refused_affinity_leaves_the_job_where_it_is(monkeypatch):
    """Refused and not caught, all three threads died: no pinned memory for Store, or a reader that
    never sets `claimed` and a run that hangs."""
    from solution import startup

    def refuse(*args):
        raise PermissionError(1, "refused")

    monkeypatch.setattr(os, "sched_setaffinity", refuse, raising=False)
    ran = []
    startup.away_from_main(lambda: ran.append(True), {1, 3})()
    assert ran == [True]


def test_threads_leave_main_a_core_until_the_imports_end():
    from solution import startup

    everywhere = os.sched_getaffinity(0)
    away = startup.other_core()
    assert away < everywhere or not away
    seen = []

    def job():
        seen.append(os.sched_getaffinity(0))
        leaving.wait()
        seen.append(os.sched_getaffinity(0))

    leaving = threading.Event()
    thread = threading.Thread(target=startup.away_from_main(job, away))
    thread.start()
    while not seen:
        time.sleep(0.001)
    startup.spread()
    leaving.set()
    thread.join()
    assert seen == [away or everywhere, everywhere]
    assert os.sched_getaffinity(0) == everywhere


def test_compile_ahead_draws_no_random_numbers():
    """A draw before the first step would change the batches or the dropout masks."""
    from solution import startup

    assert {kernel.fn.__name__ for kernel, _, _ in startup.KERNELS} >= {
        "head_tile_kernel", "nf4_decode_kernel", "nf4_columns_kernel", "rms_norm_fwd_kernel",
        "rms_norm_bwd_kernel",
        "prepare_kernel", "replay_kernel", "assemble_kernel", "residual_kernel",
        "merge_adjoint_kernel", "split_adjoint_kernel", "input_adjoint_kernel", "mlp_kernel",
        "mlp_adjoint_kernel"}
    torch.manual_seed(0)
    cpu, gpu = torch.get_rng_state(), torch.cuda.get_rng_state()
    startup.compile_ahead()
    assert torch.equal(torch.get_rng_state(), cpu)
    assert torch.equal(torch.cuda.get_rng_state(), gpu)


def norm_module(hidden=1536, eps=1e-6):
    return SimpleNamespace(weight=torch.randn(hidden, device="cuda"), variance_epsilon=eps)


def norm_ref(x, norm):
    return norm.weight * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + norm.variance_epsilon))


def norm_adjoint_ref(grad, x, norm):
    x = x.clone().requires_grad_()
    return torch.autograd.grad(norm_ref(x, norm), x, grad)[0]


def rotate_ref(x, cos, sin):
    half = x.shape[-1] // 2
    return x * cos + torch.cat((-x[..., half:], x[..., :half]), dim=-1) * sin


def close16(out, ref):
    torch.testing.assert_close(out.float(), ref.float(), rtol=2e-3, atol=2e-3)


def rough16(out, ref):
    """An operand of the backward: four mantissa bits, so 1/32 at most."""
    torch.testing.assert_close(out.float(), ref.float(), rtol=1 / 31, atol=2e-3)
    assert not (out.view(torch.int16) & 63).any()


def dropout_masks(tokens, batch, width, hidden):
    ones = torch.ones(batch, width, hidden, device="cuda")
    place = torch.randperm(batch * width, device="cuda")[:tokens].to(torch.int32)
    return (torch.nn.functional.dropout(ones, 0.05), torch.nn.functional.dropout(ones, 0.05),
            place, 1 / (1 - 0.05))


def test_prepare_and_replay_match_pytorch():
    from solution.fused import prepare, replay

    torch.manual_seed(0)
    tokens, batch, width, hidden = 23, 4, 16, 1536
    x = torch.randn(tokens, hidden, device="cuda")
    norm = norm_module()
    mask_q, mask_v, place, keep = dropout_masks(tokens, batch, width, hidden)

    bits_q, bits_v = torch.empty(2, tokens, hidden // 32, device="cuda", dtype=torch.int32)
    tape = torch.empty_like(x, dtype=torch.float16)
    a, dropped_q, dropped_v = prepare(x, norm, mask_q, mask_v, place, bits_q, bits_v, tape=tape)
    torch.testing.assert_close(tape, x.half(), rtol=0, atol=0)
    ref = norm_ref(x, norm)
    close16(a, ref)
    close16(dropped_q, ref * mask_q.view(-1, hidden)[place])
    close16(dropped_v, ref * mask_v.view(-1, hidden)[place])

    same = replay(x.half(), norm, bits_q, bits_v, keep)
    ref16 = norm_ref(x.half().float(), norm)
    close16(same[0], ref16 * mask_q.view(-1, hidden)[place])
    close16(same[1], ref16 * mask_v.view(-1, hidden)[place])


def test_assemble_matches_pytorch():
    from solution.fused import assemble

    torch.manual_seed(0)
    tokens, heads, keys, width = 19, 12, 2, 128
    joined = torch.randn(tokens, (heads + 2 * keys) * width, device="cuda").half()
    lora_q = torch.randn(tokens, heads * width, device="cuda").half()
    lora_v = torch.randn(tokens, keys * width, device="cuda").half()
    angles = torch.rand(32, width // 2, device="cuda") * 6
    cos_all = torch.cat((angles.cos(), angles.cos()), dim=-1)
    sin_all = torch.cat((angles.sin(), angles.sin()), dim=-1)
    positions = torch.randint(0, 32, (tokens,), device="cuda", dtype=torch.int32)

    query, key, value = (torch.empty(tokens, heads, width, device="cuda").half() for _ in range(3))
    assemble(joined, lora_q, lora_v, 2.0, cos_all, sin_all, positions, query, key, value)
    cos, sin = cos_all[positions][:, None], sin_all[positions][:, None]
    full = joined.float()
    ref_q = (full[:, :heads * width] + lora_q.float() * 2.0).view(tokens, heads, width)
    ref_k = full[:, heads * width:(heads + keys) * width].view(tokens, keys, width)
    ref_v = (full[:, (heads + keys) * width:] + lora_v.float() * 2.0).view(tokens, keys, width)
    close16(query, rotate_ref(ref_q, cos, sin))
    close16(key, rotate_ref(ref_k, cos, sin).repeat_interleave(heads // keys, dim=1))
    close16(value, ref_v.repeat_interleave(heads // keys, dim=1))


def test_residual_matches_pytorch():
    from solution.fused import residual

    torch.manual_seed(0)
    h = torch.randn(17, 1536, device="cuda")
    out = torch.randn(17, 1536, device="cuda").half()
    norm = norm_module()

    ref = h + out.float()
    before = torch.empty_like(out)
    b = residual(h, out, norm, before)
    torch.testing.assert_close(h, ref, rtol=1e-5, atol=1e-5)
    close16(b, norm_ref(ref, norm))
    assert torch.equal(before, h.half())


def test_merge_adjoint_matches_pytorch():
    from solution.fused import merge_adjoint

    torch.manual_seed(0)
    x = torch.randn(17, 1536, device="cuda").half()
    grad = torch.randn(17, 1536, device="cuda").half()
    total = torch.randn(17, 1536, device="cuda")
    norm = norm_module()

    ref = total + norm_adjoint_ref(grad.float(), x.float(), norm)
    out = merge_adjoint(grad, x, norm, total)
    torch.testing.assert_close(total, ref, rtol=1e-5, atol=1e-5)
    rough16(out, ref)


def test_split_adjoint_matches_pytorch():
    from solution.fused import split_adjoint

    torch.manual_seed(0)
    tokens, heads, keys, width = 19, 12, 2, 128
    grads = [torch.randn(tokens, heads, width, device="cuda").half() for _ in range(3)]
    angles = torch.rand(32, width // 2, device="cuda") * 6
    cos_all = torch.cat((angles.cos(), angles.cos()), dim=-1)
    sin_all = torch.cat((angles.sin(), angles.sin()), dim=-1)
    positions = torch.randint(0, 32, (tokens,), device="cuda", dtype=torch.int32)
    cos, sin = cos_all[positions][:, None], sin_all[positions][:, None]

    joined = split_adjoint(*grads, cos_all, sin_all, positions, keys)
    query = torch.empty(tokens, heads, width, device="cuda", requires_grad=True)
    key = torch.empty(tokens, keys, width, device="cuda", requires_grad=True)
    value = torch.empty(tokens, keys, width, device="cuda", requires_grad=True)
    outputs = (rotate_ref(query, cos, sin), rotate_ref(key, cos, sin).repeat_interleave(6, dim=1),
               value.repeat_interleave(6, dim=1))
    refs = torch.autograd.grad(outputs, (query, key, value), [g.float() for g in grads])
    close16(joined, torch.cat([ref.reshape(tokens, -1) for ref in refs], dim=1))


def test_mlp_matches_pytorch():
    from solution.fused import mlp, mlp_adjoint

    torch.manual_seed(0)
    tokens, width = 7, 8960
    both = torch.randn(tokens, 2 * width, device="cuda").half()
    grad = torch.randn(tokens, width, device="cuda").half()
    gate, up = both.float().chunk(2, dim=1)
    gate.requires_grad_(), up.requires_grad_()
    ref = torch.nn.functional.silu(gate) * up
    ref_gate, ref_up = torch.autograd.grad(ref, (gate, up), grad.float())

    close16(mlp(both), ref.detach())
    rough16(mlp_adjoint(grad, both), torch.cat((ref_gate, ref_up), dim=1))


@pytest.mark.parametrize("tokens, width, scale", [(23, 16, 1.0), (2048, 512, 65536 / 2048)])
def test_input_adjoint_matches_pytorch(tokens, width, scale):
    """The second case is the real scale: the largest batch, all of its tokens real, with gradients
    as large as GradScaler's first 65536 over 2,048 targets makes them."""
    from solution.fused import input_adjoint, prepare

    torch.manual_seed(0)
    batch, hidden = 4, 1536
    x = torch.randn(tokens, hidden, device="cuda").half()
    grad, lora_q, lora_v = ((torch.randn(tokens, hidden, device="cuda") * scale).half()
                          for _ in range(3))
    mask_q, mask_v, place, keep = dropout_masks(tokens, batch, width, hidden)
    total = torch.randn(tokens, hidden, device="cuda") * scale
    norm = norm_module()

    bits_q, bits_v = torch.empty(2, tokens, hidden // 32, device="cuda", dtype=torch.int32)
    prepare(x.float(), norm, mask_q, mask_v, place, bits_q, bits_v)
    full = (grad.float() + lora_q.float() * mask_q.view(-1, hidden)[place]
            + lora_v.float() * mask_v.view(-1, hidden)[place])
    ref = total + norm_adjoint_ref(full, x.float(), norm)
    out = input_adjoint(grad, lora_q, lora_v, bits_q, bits_v, keep, x, norm, total)
    torch.testing.assert_close(total, ref, rtol=1e-5, atol=1e-5 * scale)
    rough16(out, ref)


def test_fp16_sums_hold_for_one_matmul_only():
    from solution.sums import summed_in_fp16

    blas = torch.backends.cuda.matmul
    torch.manual_seed(0)
    a = torch.randn(64, 1536, device="cuda").half()
    b = (torch.randn(1536, 2048, device="cuda") * 0.02).half()
    ref = torch.matmul(a.float(), b.float())
    out = torch.empty(64, 2048, device="cuda", dtype=torch.float16)
    assert blas.allow_fp16_accumulation is False
    assert summed_in_fp16(a, b, out=out) is out
    assert blas.allow_fp16_accumulation is False
    assert ((out.float() - ref).norm() / ref.norm()).item() < 0.01
    assert ((summed_in_fp16(a, b).float() - ref).norm() / ref.norm()).item() < 0.01
    with pytest.raises(RuntimeError):
        summed_in_fp16(a, b.t()[:7])
    assert blas.allow_fp16_accumulation is False
