"""What runs before the first step: earlier, while the imports wait anyway, or not at all.

The clock counts the whole process, and a quarter of a run passes before the first step. Nothing
here touches what is trained, and no random number is drawn, since a draw before the first step
would change the batches or the dropout masks:

- the kernels compile in threads next to the imports and the model load, so that the first step
  finds them ready. Compiling them in the first step costs about 4 s;
- the first matmul and the first fused attention load cuBLAS and cuDNN, about 1 s, in that thread;
- host memory is mapped, touched and only then registered with the driver. cuMemHostAlloc would
  hold the process's memory map for as long as it runs, and an import maps something every few
  milliseconds: main stood through all 2.3 s of it whatever the size of the blocks. Touching holds
  nothing, and in huge pages it takes 0.22 s per GiB, against 0.57, and 0.003 s per GiB to give back
  at exit, against 0.18. See pin_ahead;
- trl repeats the scan of 146,000 installed files that transformers has just done, 2.4 s, and
  transformers.models repeats its own scan of 1,541 files, 0.5 s;
- sklearn (for assisted generation) and torch's Inductor (for a CPU path of bitsandbytes) are
  imported and never used, 1.4 s;
- libraries import torch._dynamo, torch.distributed.tensor, sympy, wandb and boto3 for what they
  might be asked, and in this run the questions have one answer: see Deferred and defer_unused.
  These were 2 of the 3.8 s of imports here, starting from a decorator in a file about DeepGEMM;
- gcc builds 12 launchers from the same headers, 4 s of a core: see precompile_headers;
- building the model on the meta device imports sympy: see patch_values_on_meta;
- Dataset.map hashes its function with dill to name a cache file, 1-2 s. A dataset in memory has
  no cache file;
- the checkpoint is read and quantized in a thread of its own, see ahead.py;
- Soup tokenizes each row four times to label it: see patch_row_labels;
- the threads stay off main's physical core while it imports: the 4 vCPUs are 2 cores, and imports
  beside a busy sibling take 5.6 s, against 3.5 s. See other_core;
- Soup writes tokenizer.json three times at the end, 0.11 s each, twice to the same place: see
  patch_tokenizer_file.
"""

import ctypes
import functools
import gc
import importlib.machinery
import importlib.metadata
import importlib.util
import mmap
import os
import re
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import threading
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
import torch.nn.functional as F

KERNELS = []  # (kernel, arguments, constexprs and options), added to by the file of each kernel
# Bytes of each pinned block: stream.Store cuts them up and uses 3.88 GiB. They lie end to end in
# one mapping now; Store's cuts are what fixes their sizes.
PINNED = (2 ** 29, *[2 ** 28] * 13, 2 ** 27, 2 ** 26)
REGISTER_FLAGS = 0x01 | 0x02  # CU_MEMHOSTREGISTER_PORTABLE | CU_MEMHOSTREGISTER_DEVICEMAP
CU_LIMIT_STACK_SIZE = 0
CU_STREAM_NON_BLOCKING = 1
COMPILERS = 3  # threads: gcc and ptxas run outside the GIL
BLOCK_SIZE = 128  # of torch's flex attention, which transformers takes at import
BUILD_FLAGS = "-O1", "-fPIC"
HEADERS = """#include "cuda.h"
#include <dlfcn.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#define PY_SSIZE_T_CLEAN
#include <Python.h>
"""
# The tag a template marks assistant text with. A false hit only costs the render it saves today.
GENERATION = re.compile(r"{%[-+]?\s*generation\b")
compiling, pinning, reading, pinned, reader = None, None, None, None, None
whole, own_bytes, context, register_code = None, 0, None, None
swapping = threading.RLock()  # two threads must not both take a Deferred out of sys.modules
quantizable = threading.Event()  # what bitsandbytes reaches for while it quantizes is imported
claimed, filled = threading.Event(), threading.Event()  # see claim() and pin_ahead()
kept_in_graph, collecting = [], False


class Deferred(types.ModuleType):
    """A library that others import for what they might ask it. It stands in sys.modules and holds
    the true answers to the questions a run does ask; any other question, or anything that imports
    a part of it by name, brings in the real library, which takes its place and answers. So a run
    that does use it gets what it always got."""

    def __init__(self, spec, answers, on_real=None):
        super().__init__(spec.name)
        self.__spec__, self.__file__, self._answers = spec, spec.origin, answers
        self.__path__ = spec.submodule_search_locations or []  # `from a import b` reads it
        self._on_real = on_real

    def _real(self):
        """With its family: the real ones import each other."""
        with swapping:
            family = self.__name__.split(".")[:2]
            waiting = [(name, module) for name, module in sys.modules.copy().items()
                       if isinstance(module, Deferred) and name.split(".")[:2] == family]
            for name, _ in waiting:
                del sys.modules[name]
            real = importlib.import_module(self.__name__)
            for name, module in waiting:
                if module._on_real is not None:
                    module._on_real(importlib.import_module(name))
            return real

    def __getattr__(self, name):
        if sys.modules.get(self.__name__) is self and name in self._answers:
            return self._answers[name]
        return getattr(self._real(), name)


class Loaded:
    """A module that is in sys.modules already, for the import system to find."""

    def __init__(self, module):
        self.module, self.spec = module, module.__spec__

    def create_module(self, spec):
        return self.module

    def exec_module(self, module):
        vars(module)["__spec__"] = self.spec  # torch._dynamo.config refuses a plain assignment


class PackageFirst:
    """`import a.b` when `a` is a Deferred: the real `a` comes first and may bring `a.b` with it."""

    def find_spec(self, name, path=None, target=None):
        package = sys.modules.get(name.rpartition(".")[0])
        if not isinstance(package, Deferred):
            return None
        package._real()
        module = sys.modules.get(name)
        if module is None:
            return importlib.util.find_spec(name)
        vars(module)["__spec__"] = importlib.util.spec_from_loader(name, Loaded(module))
        return module.__spec__  # the import system prefers it to anything found here


def assume_constant_result(fn):
    """torch._dynamo's own body: a mark for a compiler that may never come."""
    fn._dynamo_marked_constant = True
    return fn


def allow_in_graph(fn):
    """Kept for a compiler that may come: see allow_kept."""
    kept_in_graph.append(fn)
    return fn


def allow_kept(dynamo):
    for function in kept_in_graph:
        dynamo.allow_in_graph(function)


def graph_break(msg=""):
    """torch._dynamo's own body is empty too."""


def disabled(fn=None, recursive=True, **options):
    """torch._dynamo.disable. It only means something while a compile runs, and then the real one
    has been imported."""
    if fn is None:
        return lambda inner: disabled(inner, recursive, **options)

    @functools.wraps(fn)
    def call(*args, **kwargs):
        dynamo = sys.modules.get("torch._dynamo")
        if dynamo is None or isinstance(dynamo, Deferred):
            return fn(*args, **kwargs)
        return dynamo.disable(fn, recursive, **options)(*args, **kwargs)

    return call


class Later(type):
    """A class or a function of a Deferred library, which a module took at import and may never
    use. Nothing can be an instance or a subclass of it before the library is there, and anything
    else asked of it brings the library in."""

    def __new__(cls, *args):
        if len(args) == 3:  # `class Mine(this)`: the real one is the base
            name, bases, namespace = args
            bases = tuple(base.real() if isinstance(base, Later) else base for base in bases)
            return types.new_class(name, bases, exec_body=lambda body: body.update(namespace))
        library, name = args
        return super().__new__(cls, name, (), {"library": library, "__module__": library})

    def __init__(cls, library, name):
        super().__init__(name, (), {})

    def there(cls):
        """The real one, if its library has come. A family comes together, see Deferred._real,
        and may leave a member out of sys.modules until something imports it."""
        module = sys.modules.get(cls.library)
        return None if module is None or isinstance(module, Deferred) \
            else getattr(module, cls.__name__, None)

    def __instancecheck__(cls, instance):
        real = cls.there()
        return real is not None and isinstance(instance, real)

    def __subclasscheck__(cls, subclass):
        real = cls.there()
        return real is not None and issubclass(subclass, real)

    def real(cls):
        module = importlib.import_module(cls.library)
        return getattr(module._real() if isinstance(module, Deferred) else module, cls.__name__)

    def __call__(cls, *args, **kwargs):
        return cls.real()(*args, **kwargs)

    def __getattr__(cls, name):
        if name.startswith("__"):  # what inspect and typing look for on anything
            raise AttributeError(name)
        return getattr(cls.real(), name)


def later(library, *names, **values):
    return library, {**{name: Later(library, name) for name in names}, **values}


def spec_of(name):
    """Of a module that does not run for it: asking for a part of a package imports the package."""
    package = name.rpartition(".")[0]
    if not package or package in sys.modules:
        return importlib.util.find_spec(name)
    if importlib.util.find_spec(name.partition(".")[0]) is None:
        return None
    return importlib.machinery.ModuleSpec(name, None, is_package=True)


def defer_unused():
    """What these libraries answer until something real is asked of them:
    - torch._dynamo: `torch.compile` is off in this task. transformers decorates one function with
      it, torch.optim calls graph_break, a no-op, around every step, and accelerate asks whether a
      module is an OptimizedModule, which none is. The first call of a bitsandbytes op goes through
      `disable`;
    - torch.distributed.tensor and .fsdp, torch's flex attention, and sympy behind them:
      transformers takes their names at import, for models spread over GPUs, and only asks whether
      a tensor is a DTensor, which none is;
    - wandb: `run` is None until wandb.init() has been called;
    - boto3: accelerate imports it for its SageMaker command, and torchaudio is imported by
      transformers for audio files;
    - scipy.optimize: the losses of object detection take linear_sum_assignment at import."""
    libraries = dict([
        ("torch._dynamo", {"assume_constant_result": assume_constant_result,
                           "graph_break": graph_break, "allow_in_graph": allow_in_graph,
                           "disable": disabled}),
        later("torch._dynamo.eval_frame", "OptimizedModule"),
        ("wandb", {"run": None}), ("boto3", {}), ("torchaudio", {}),
        later("scipy.optimize", "linear_sum_assignment"),
        later("transformers.distributed.tensor_parallel", "_get_parameter_tp_plan",
              "_validate_tp_plan_styles", "apply_tensor_parallelism", "gather_state_dict_for_save",
              "replace_layer_number_by_wildcard", "verify_tp_plan"),
        later("transformers.distributed.sharding_utils", "DtensorShardOperation",
              "_dtensor_from_local_like"),
        later("torch._dynamo.utils", "is_compile_supported"),
        later("torch._dynamo._trace_wrapped_higher_order_op", "TransformGetItemToIndex"),
        later("torch.nn.attention.flex_attention", "AuxRequest", "BlockMask", "create_block_mask",
              "flex_attention", _DEFAULT_SPARSE_BLOCK_SIZE=BLOCK_SIZE),
        later("torch.distributed.tensor", "DTensor"),
        later("torch.distributed.fsdp", "CPUOffloadPolicy", "MixedPrecisionPolicy"),
        later("torch.distributed.fsdp.fully_sharded_data_parallel", "FullyShardedDataParallel"),
        later("torch.distributed._composable.fsdp", "fully_shard")])
    for name, answers in libraries.items():
        spec = spec_of(name)
        if spec is not None:
            on_real = allow_kept if name == "torch._dynamo" else None
            sys.modules.setdefault(name, Deferred(spec, answers, on_real))
            package, _, part = name.rpartition(".")
            if package in sys.modules and part not in vars(sys.modules[package]):
                setattr(sys.modules[package], part, sys.modules[name])
    sys.meta_path.insert(0, PackageFirst())


def precompile(kernel, *args, **kwargs):
    """Compile a kernel ahead of its first launch. Tensors are given by their dtype; everything
    must match the real launch, down to constexprs and options, or it compiles again in step 1.
    A runtime int counts too, by being divisible by 16 or not."""
    KERNELS.append((kernel, args, kwargs))


def warm_libraries():
    """No autograd here: its engine is a thread of its own, and every thread that reaches cuBLAS
    keeps a handle and a workspace to the end of the run, 16 MiB of the GPU on a T4."""
    x = torch.zeros(1, 12, 8, 128, dtype=torch.float16, device="cuda")
    out, *state = torch._scaled_dot_product_efficient_attention(x, x, x, None, True, 0.0, True)
    torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(
        out, x, x, x, None, out, *state, 0.0, [True, True, True, False], True)
    F.linear(out[0, 0], x[0, 0])


def compile_one(item):
    kernel, args, kwargs = item
    kernel.warmup(*args, grid=(1,), **kwargs).run  # reading .run loads it on the GPU


def python_headers():
    scheme = sysconfig.get_default_scheme()
    scheme = "posix_prefix" if scheme == "posix_local" else scheme
    return sysconfig.get_paths(scheme=scheme)["include"]


def precompile_headers(folder):
    """gcc spends half of the 0.2 s of a launcher at -O1 reading the same lines of cuda.h and
    Python.h, and Triton calls it 12 times. Made once, in this run, for the builds of this run."""
    from triton.backends.nvidia import driver

    header = os.path.join(folder, "launcher.h")
    with open(header, "w", encoding="utf-8") as handle:
        handle.write(HEADERS)
    includes = [f"-I{path}" for path in (*driver.include_dirs, python_headers())]
    subprocess.check_call([shutil.which("gcc"), *BUILD_FLAGS, "-x", "c-header", header, "-o",
                           header + ".gch", *includes], stdout=subprocess.DEVNULL)
    return header


def build_launcher(header):
    """knobs.build.impl: the command of triton.runtime.build._build, with -O1 for -O3 and the
    headers above: 0.11 s for a launcher, against 0.35 s."""
    def build(name, source, folder, library_dirs, include_dirs, libraries):
        target = os.path.join(folder, name + sysconfig.get_config_var("EXT_SUFFIX"))
        command = [shutil.which("gcc"), source, *BUILD_FLAGS, "-include", header, "-shared",
                   "-Wno-psabi", "-o", target]
        command += [f"-l:{name}" if ".so" in name else f"-l{name}" for name in libraries]
        command += [f"-L{path}" for path in library_dirs]
        command += [f"-I{path}" for path in (*include_dirs, folder, python_headers())]
        subprocess.check_call(command, stdout=subprocess.DEVNULL)
        return target

    return build


def compile_ahead():
    import triton
    from triton.runtime import cache

    # The key of a compiled kernel would hash Triton's sources and library, 2 s, to tell its
    # versions apart in a cache. This cache starts empty and holds one version.
    cache.triton_key = lambda: triton.__version__
    triton.knobs.compilation.store_binary_only = True  # the text of each IR stage is never read
    with tempfile.TemporaryDirectory() as folder:
        try:
            triton.knobs.build.impl = build_launcher(precompile_headers(folder))
            first, args, kwargs = KERNELS[0]
            first.warmup(*args, grid=(1,), **kwargs)  # alone: it builds what the others share
            with ThreadPoolExecutor(COMPILERS, thread_name_prefix="compile_ahead") as pool:
                list(pool.map(compile_one, KERNELS))
        finally:  # what compiles later compiles the way Triton does by itself
            triton.knobs.build.impl = None
    warm_libraries()


def no_idle_stacks():
    """A CUDA context keeps a stack for each of the 40,960 hardware threads of a T4 whether a kernel
    asks for one or not: 1,024 bytes each is 40 MiB, measured. A step needs almost none of it: our
    kernels have no spills, and the limit reads 32 bytes after a whole run. The limit only says what
    to keep: the driver raises it for a kernel that needs more."""
    driver = ctypes.CDLL("libcuda.so.1")
    driver.cuCtxSetLimit.argtypes = [ctypes.c_int, ctypes.c_size_t]
    driver.cuCtxSetLimit(CU_LIMIT_STACK_SIZE, 0)


def own_stream(device):
    """A CUDA stream for copies beside the main stream. The first torch.cuda.Stream costs the
    context 20 MiB of the GPU on a T4, measured, for a reason we did not find. A plain non-blocking
    stream overlaps the main stream as well and costs nothing."""
    driver = ctypes.CDLL("libcuda.so.1")
    found = ctypes.c_void_p()
    driver.cuStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint]
    if driver.cuStreamCreate(ctypes.byref(found), CU_STREAM_NON_BLOCKING) != 0:
        return torch.cuda.Stream(device)
    return torch.cuda.ExternalStream(found.value, device=device)


def current_context():
    driver = ctypes.CDLL("libcuda.so.1")
    found = ctypes.c_void_p()
    driver.cuCtxGetCurrent.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    driver.cuCtxGetCurrent(ctypes.byref(found))
    return found


def register(address, size):
    """cuMemHostRegister from a thread that has no context of its own yet. 0 means pinned."""
    driver = ctypes.CDLL("libcuda.so.1")
    driver.cuCtxSetCurrent.argtypes = [ctypes.c_void_p]
    driver.cuMemHostRegister_v2.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint]
    driver.cuCtxSetCurrent(context)
    return driver.cuMemHostRegister_v2(address, size, REGISTER_FLAGS)


def claim(size):
    """The first `size` bytes of `whole` are the reading thread's: it touches them by reading."""
    global own_bytes
    own_bytes = size
    claimed.set()


def pin_ahead():
    """Map, touch, register. Reading into the memory touches it as well, so the first part of it is
    left to the thread that reads, and registering waits for that thread. If the driver refuses, the
    blocks are pinned the old way."""
    global pinned, whole, register_code
    size = sum(PINNED)
    area = mmap.mmap(-1, size, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
    try:
        area.madvise(mmap.MADV_HUGEPAGE)
    except OSError:  # a kernel without them: touching is slower, nothing else changes
        pass
    full = torch.frombuffer(area, dtype=torch.uint8)
    starts = [sum(PINNED[:index]) for index in range(len(PINNED))]
    blocks = [full[start:start + block] for start, block in zip(starts, PINNED)]
    whole = full
    claimed.wait()
    ctypes.memset(full.data_ptr() + own_bytes, 0, size - own_bytes)
    filled.wait()
    register_code = register(full.data_ptr(), size)
    if register_code != 0:
        blocks = [torch.empty(block, dtype=torch.uint8, pin_memory=True) for block in PINNED]
    pinned = blocks


def other_core():
    """The vCPUs of the second physical core on the box of this task, a Kaggle T4: 4 vCPUs, with
    0, 2 and 1, 3 as siblings. On any other shape, nothing. Main is not held anywhere: with the
    threads held to one core, the scheduler leaves it the other."""
    return {1, 3} if os.sched_getaffinity(0) == {0, 1, 2, 3} else set()


def away_from_main(job, cpus):
    """`job`, and what it starts, on `cpus` until spread()."""
    def run():
        try:
            if cpus:
                os.sched_setaffinity(0, cpus)
        except OSError:  # refused: the job runs where it is, which only costs time
            pass
        job()

    return run


def spread():
    """The imports are over: what is left beside main may use every vCPU."""
    everywhere = os.sched_getaffinity(0)
    for thread in threading.enumerate():
        try:
            os.sched_setaffinity(thread.native_id, everywhere)
        except (OSError, TypeError):  # gone since enumerate(), or never started
            pass


def beside(job):
    """Run `job` in a thread next to the imports. It must leave anything it did not finish to the
    code that would do it anyway: an exception ends it quietly."""
    global reader
    reader = job


def read_ahead():
    try:
        reader()
    except Exception:  # noqa: BLE001 - the load does what is left, as it always did
        pass
    finally:
        claimed.set()
        filled.set()


def read_done():
    if reading is not None:
        reading.join()


def pinned_memory():
    """The blocks of pinned host memory, which stream.Store cuts up: pinning is slow, so it is
    done once and ahead."""
    pinning.join()
    return pinned


def reuse_models_scan():
    """transformers.models reads its 1,541 files for a structure `import transformers` has already
    made: the same scan of the same folder, cached under another key, with a prefix on the names and
    the names of the package itself mixed in."""
    import transformers.utils.import_utils as utils

    scan = utils.define_import_structure
    models = Path(utils.__file__).parents[1] / "models"

    def define_import_structure(path, prefix=None):
        if prefix is not None or str(path) != str(models / "__init__.py"):
            return scan(path, prefix)
        known = scan(models, prefix="models")
        found = {key: {name[len("models."):]: names for name, names in value.items()
                       if name.startswith("models.")} for key, value in known.items()}
        return {key: value for key, value in found.items() if value}

    utils.define_import_structure = define_import_structure


def source_of_caller(stacklevel):
    """torch._library.utils.get_source without inspect.getmodule, which walks all of sys.modules
    for each of the 19 operators bitsandbytes registers. The same "file:line"."""
    frame = sys._getframe(stacklevel)
    return f"{frame.f_code.co_filename}:{frame.f_lineno}"


def import_lean():
    """Import what the run needs without the parts it never uses. The replacements here are in place
    only during these imports, and the libraries read them once, at import; the stand-ins of
    defer_unused stay for the whole run. All the imports of a run are here, so that prepare() can
    keep the collector away from them."""
    import transformers.utils as utils
    from torch._library import utils as library
    from transformers.utils.import_utils import PACKAGE_DISTRIBUTION_MAPPING

    reuse_models_scan()
    scan, sklearn, compile_fn, source = (importlib.metadata.packages_distributions,
                                         utils.is_sklearn_available, torch.compile,
                                         library.get_source)
    importlib.metadata.packages_distributions = lambda: PACKAGE_DISTRIBUTION_MAPPING
    utils.is_sklearn_available = lambda: False
    torch.compile = lambda fn, **options: fn
    library.get_source = source_of_caller
    try:
        import bitsandbytes  # noqa: F401
        quantizable.set()
        import datasets  # noqa: F401
        import peft  # noqa: F401
        import transformers.data  # noqa: F401
        import transformers.generation.candidate_generator  # noqa: F401
        import trl.trainer.sft_trainer  # noqa: F401
        import soup_cli.cli  # noqa: F401
    finally:
        importlib.metadata.packages_distributions = scan
        utils.is_sklearn_available = sklearn
        torch.compile = compile_fn
        library.get_source = source


def patch_map_fingerprint():
    """map() takes a fingerprint and then does not hash anything. A random one is what datasets
    itself falls back to, and it draws from a generator of its own."""
    from datasets import Dataset
    from datasets.fingerprint import generate_random_fingerprint

    original = Dataset.map

    def map_in_memory(self, *args, **kwargs):
        if not self.cache_files and not kwargs.get("new_fingerprint"):
            kwargs["new_fingerprint"] = generate_random_fingerprint()
        return original(self, *args, **kwargs)

    Dataset.map = map_in_memory


def patch_values_on_meta():
    """The model is built on the meta device, where a tensor holds no values and no random number
    is drawn, but two things still compute there: the embedding fills itself, and the rotary
    embedding works out its frequencies, which transformers works out again after the load. On meta
    these run through torch's Python references, which import sympy and symbolic_shapes: 0.4 s, to
    get nothing. The embedding skips the fill, and the frequencies wait on the host, which also
    spares them transformers's empty_like of a meta buffer."""
    from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding

    fill = torch.nn.Embedding.reset_parameters
    frequencies = Qwen2RotaryEmbedding.__dict__["compute_default_rope_parameters"].__func__

    def reset_parameters(self):
        if not self.weight.is_meta:
            fill(self)

    def compute_default_rope_parameters(config, device=None, **kwargs):
        if not torch.empty(0).is_meta:
            return frequencies(config, device, **kwargs)
        with torch.device("cpu"):
            return frequencies(config, **kwargs)

    torch.nn.Embedding.reset_parameters = reset_parameters
    Qwen2RotaryEmbedding.compute_default_rope_parameters = staticmethod(
        compute_default_rope_parameters)


def patch_row_labels():
    """To label a row Soup first asks the template for a mask. A template without {% generation %}
    gives an all-zero mask, which Soup rightly throws away before it starts over, so that render and
    tokenization are wasted on every row. Then it renders the row, then each growing prefix, and the
    last prefix is the row again. The labels come out the same without the first and the last."""
    import soup_cli.data.loss_mask as loss_mask

    attempt, tokenize = loss_mask._apply_template_with_mask, loss_mask._tokenize_only
    recent = {}  # the ids of the row at hand and of its prefixes

    def apply_template_with_mask(tokenizer, messages):
        template = getattr(tokenizer, "chat_template", None)
        if isinstance(template, str) and template and not GENERATION.search(template) and any(
                message.get("role") == "assistant" for message in messages):
            return None
        return attempt(tokenizer, messages)

    def tokenize_only(tokenizer, messages):
        key = id(tokenizer), str(getattr(tokenizer, "chat_template", None)), repr(messages)
        if key not in recent:
            if len(recent) >= 8:
                recent.clear()
            recent[key] = tokenizer, tokenize(tokenizer, messages)  # held: its id stays its own
        return list(recent[key][1])

    loss_mask._apply_template_with_mask = apply_template_with_mask
    loss_mask._tokenize_only = tokenize_only


def patch_tokenizer_file():
    """Soup saves the tokenizer into the checkpoint and twice into the output, and serializing
    tokenizer.json takes 0.1 s each time. For the tokenizer of this task the later saves copy the
    first file while it is still what a save would write: the file is as it was written, and all it
    holds is the same. Reading that back costs microseconds for every part but the vocabulary and
    merges, which are the 0.1 s and which nothing edits after the load: their size is compared. The
    other files are written as usual."""
    from transformers.tokenization_utils_tokenizers import TOKENIZER_FILE, TokenizersBackend

    from .ahead import CHECKPOINT

    save = TokenizersBackend._save_pretrained
    first = [None, None, None]  # what the file holds, the file, its size and time when written

    def held_in_file(tokenizer):
        backend = tokenizer.backend_tokenizer
        parts = backend.normalizer, backend.pre_tokenizer, backend.post_processor, backend.decoder
        return (tokenizer, backend.truncation, backend.padding, backend.get_vocab_size(True),
                [None if part is None else part.__getstate__() for part in parts],
                [(index, token.__getstate__())
                 for index, token in backend.get_added_tokens_decoder().items()])

    def stamp(path):
        info = os.stat(path) if os.path.isfile(path) else None
        return info and (info.st_size, info.st_mtime_ns)

    def save_once(self, save_directory, file_names, legacy_format=None, filename_prefix=None):
        target = os.path.join(str(save_directory), TOKENIZER_FILE)
        if filename_prefix or self.name_or_path != CHECKPOINT:
            return save(self, save_directory, file_names, legacy_format, filename_prefix)
        held = held_in_file(self)
        if first[0] != held or stamp(first[1]) != first[2]:
            files = save(self, save_directory, file_names, legacy_format, filename_prefix)
            first[:] = held, target, stamp(target)
            return files
        if not (os.path.exists(target) and os.path.samefile(first[1], target)):
            shutil.copyfile(first[1], target)
        return (*file_names, target)

    TokenizersBackend._save_pretrained = save_once


def patch_first_step():
    """A run on another Trainer, or on this one's own step, never reaches adjoint_step, and the
    collector has to come back there too."""
    from transformers import Trainer

    step = Trainer.training_step

    def training_step(self, *args, **kwargs):
        finish()
        return step(self, *args, **kwargs)

    Trainer.training_step = training_step


def prepare():
    """Called first in apply(), before anything imports trl, peft or bitsandbytes. The collector
    stays out until the first step, see finish(), and what the imports built stays out of its
    sight after them: millions of objects that live to the end of the run, and full passes over
    them cost 0.9 s."""
    global compiling, pinning, reading, context
    gc.disable()
    defer_unused()
    if torch.cuda.is_available():
        context = current_context()
        no_idle_stacks()
        cpus = other_core()
        compiling = threading.Thread(target=away_from_main(compile_ahead, cpus),
                                     name="compile_ahead")
        pinning = threading.Thread(target=away_from_main(pin_ahead, cpus), name="pin_ahead")
        reading = threading.Thread(target=away_from_main(read_ahead, cpus), name="read_ahead")
        compiling.start()
        pinning.start()
        reading.start()
    try:
        import_lean()
        patch_map_fingerprint()
    finally:
        quantizable.set()
    if pinning is not None:
        spread()
    patch_values_on_meta()
    patch_row_labels()
    patch_tokenizer_file()
    patch_first_step()
    gc.freeze()


def finish():
    """Called at the top of every step. The first one waits for the kernels, if they are late,
    takes the whole heap out of the collector's sight, and lets the collector start: its later
    passes, and the ones of the interpreter's exit, would only walk millions of objects that live
    to the end anyway."""
    global collecting
    if not collecting:
        collecting = True
        if compiling is not None:
            compiling.join()
        gc.freeze()
        gc.enable()
