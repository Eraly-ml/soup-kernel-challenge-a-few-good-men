# Soup Kernel Challenge — A Few Good Men

`soup train` fine-tunes **Qwen2.5-1.5B-Instruct** (4-bit NF4, LoRA r=16 on `q_proj`/`v_proj`,
batch 4, up to 512 tokens, 256 Kazakh instructions, 64 steps) on one free **Kaggle T4**. This
repository makes that same run **2.26x faster** and **25.8x lighter** with our own Triton kernels
and patches, without moving the trained adapter: eval loss is identical to the base to five
decimals.

- **Final code commit**: https://github.com/BAITC-Hacks/hack-623dda6b-a-few-good-men/commit/e3062178433394bb461dcb06794f0d62dd5274c0
  (`solution/` SHA-256 `c622ff372b6a6259754104150f6812e075280f36c852a28b6c16da2e2ddf387d`)
- **Public Kaggle notebook** (from `templates/kaggle_notebook.ipynb`, `COMMIT` pinned to the tip of
  `main`, runs with **Run All** without edits):
  https://www.kaggle.com/code/eralygainulla/soup-t4-runall-9c41

## Results on Kaggle T4

Two base and two solution runs in one session, jury order `base → solution → solution → base`,
Tesla T4, fp16, means of the two runs per side, from `runs/result.json` of the notebook above.

| | Base Soup 0.77.0 | Ours | Ratio |
|---|---|---|---|
| Whole `soup train`, s | 84.29 (87.60 / 80.98) | **37.27** (37.37 / 37.16) | **2.2616x speedup** (time x0.442) |
| Peak GPU memory, MiB | 14,348 | **556** (536 / 556) | **25.8058x less** (memory x0.039) |
| Training loop, s | 57.73 | **24.35** | 2.37x |
| Eval loss, public rows | 2.04965 | 2.04965 | **+0.000%** (limit 0.5%) |

`harness/check_rules.py`: **10/10 passed**. Our own `@triton.jit` kernels take **20.7%** of the
training time (check 4, limit 5%). Soup regression tests with our patches: **79 passed** (check 10).
Our own suite: **87 tests passed**. Loss curve vs base: first step 0.00%, mean of all steps
0.01–0.02% (check 6, limits 1% / 3%).

## What we sped up, and why

The task profile says ~44% of GPU time goes to small element-wise ops and dtype casts, ~7% to the
softmax over the 152k-token vocabulary, ~5% to NF4 unpacking, and that the base peak is ~14 GB of
15 GB. We attacked all four, plus the thing the profile does not show: the autograd graph and the
saved activations of 28 decoder layers.

Instead of letting PyTorch build a graph, `solution.apply()` replaces the training step with our
own forward and backward sweep:

1. **Frozen weights never live on the GPU as a whole.** `solution/stream.py` keeps the NF4 weights
   and the tied embedding in pinned host memory and streams them through two GPU slots on a side
   CUDA stream, with CUDA events so a copy never overwrites a layer still in use.
2. **Padding is dropped.** `solution/adjoint.py` packs the real tokens of the batch into one
   compact array, keeps the original positions for RoPE and the row boundaries for causal
   attention, and runs each of the 28 layers under `torch.no_grad()`.
3. **Activations are parked in host memory.** `solution/parking.py` moves what the backward needs
   (layer input, gate/up results, q/k/v, attention output, packed dropout bits) to pinned RAM and
   prefetches them back per layer. v16 keeps the *first* layer's tape resident on GPU and writes it
   inside `prepare_kernel`, which removes one separate copy.
4. **The LM head never materializes all logits.** `solution/head.py` walks the 151,936-row
   vocabulary in tiles of 7,680 rows, updating online-softmax statistics, and produces loss,
   entropy, accuracy, the gradient w.r.t. hidden states and the `rewind.jsonl` fields from the same
   pass.
5. **The backward is written by hand**, layer 28 down to layer 1, and writes the LoRA gradients of
   `q_proj`/`v_proj` straight into `.grad`. Unscaling, clipping and the optimizer step stay exactly
   as Soup/TRL do them.

### Our kernels

Fourteen `@triton.jit` kernels, all launched in every measured step. Full table with per-kernel
shares and what each one replaces is in [SUBMISSION.md](SUBMISSION.md).

| File | Kernels | Role |
|---|---|---|
| `solution/nf4.py` | `nf4_decode_kernel`, `nf4_columns_kernel` | unpack NF4 + double-quantized block scales straight into the matmul layout, row-major and column-major (11.4% of the loop together) |
| `solution/fused.py` | `prepare_kernel`, `replay_kernel`, `assemble_kernel`, `residual_kernel`, `mlp_kernel`, `merge_adjoint_kernel`, `split_adjoint_kernel`, `input_adjoint_kernel`, `mlp_adjoint_kernel` | RMSNorm + LoRA input prep + dropout, dropout-mask recomputation, RoPE + q/k/v assemble, residual + norm, SwiGLU forward, and the four backward kernels |
| `solution/head.py` | `head_tile_kernel` | online-softmax statistics for one vocabulary tile: loss, entropy, accuracy, gradient |
| `solution/norm.py` | `rms_norm_fwd_kernel`, `rms_norm_bwd_kernel` | standalone RMSNorm where the fused variant does not apply |

`solution/rough.py` defines one more `@triton.jit` helper, `rough`, called from inside the kernels
above, so it is not a separate launch. `solution/startup.py` overlaps memory setup, weight reading
and kernel compilation and defers imports; it contains no kernels.

Ideas came from published work, code is ours (no kernel code copied, no kernel library installed —
check 2 reports `loaded: none`): QLoRA for NF4 and double quantization
(https://arxiv.org/abs/2305.14314), Liger-Kernel and Unsloth for the fused linear + chunked
cross-entropy idea (https://github.com/linkedin/Liger-Kernel,
https://github.com/unslothai/unsloth), ZeRO-Offload and PyTorch activation offloading for parking
activations in host memory (https://arxiv.org/abs/2101.06840), gradient checkpointing for
recompute-instead-of-store (https://arxiv.org/abs/1604.06174).

## How to run it

### On Kaggle (what the jury runs)

1. Open https://www.kaggle.com/code/eralygainulla/soup-t4-runall-9c41
2. Right panel: **Accelerator: GPU T4 x2**, **Internet: on**.
3. **Run All**. It clones this repository at the pinned `COMMIT`, runs `task/prepare_data.py`,
   `pytest solution`, `harness/run_task.py`, `harness/check_rules.py` and `harness/score.py`.
   About 10–15 minutes. Nothing is edited by hand.

The notebook is the unmodified `templates/kaggle_notebook.ipynb`; only the `REPO_URL` and `COMMIT`
cell is filled in.

### Locally (Linux with a CUDA T4)

```bash
pip install "soup-cli[train]==0.77.0" "transformers==5.16.1" "peft==0.20.0" "trl==0.29.1" \
            "bitsandbytes==0.50.2" "accelerate==1.14.0" pytest safetensors
python task/prepare_data.py
python -m pytest solution -q          # 87 tests
python harness/run_task.py --repeats 2
python harness/check_rules.py         # 10 checks
python harness/score.py runs/result.json
```

`harness/`, `tests/`, `task/` and `templates/` are untouched from the starter kit.

## Limitations — where this is weaker

- **The demo video and the presentation were not finished before the deadline.** The links in
  `SUBMISSION.md` are placeholders. This is a missing deliverable of section 10.
- **The fast path is not bit-exact autograd.** Parts of the backward round to fp16,
  `solution/rough.py` keeps four mantissa bits on some frozen-weight products, and
  `solution/sums.py` accumulates in fp16 inside selected matmuls. On this run the eval loss matches
  the base to five decimals and the loss curve stays within 0.02%, but the guarantee is
  tolerance-based, not bitwise.
- **It is narrow.** The custom step only engages for this exact configuration: Qwen2.5-1.5B-Instruct
  in NF4, one default LoRA adapter on `q_proj`/`v_proj`, fp16, one batch per optimizer step, and an
  input size the packing path accepts. Anything else falls back to the standard trainer step with
  the weights moved back to the GPU, so none of this generalizes to other models or settings.
- **It trades host memory for GPU memory.** Frozen weights and most activations sit in pinned RAM;
  on a machine with little free RAM the streaming path would not fit.
- **The memory result caps the speed.** 556 MiB peak is only possible because weights and
  activations cross PCIe every layer. The two NF4 decode kernels alone are 11.4% of the training
  loop; with a bigger GPU-memory budget that time could be traded back.
- **What did not work.** Dropping the head tile from 7,680 to 7,616 rows failed
  `test_head_stream_matches_pytorch` (gradient outside tolerance) and was discarded. An earlier
  v15 candidate reached 476 MiB in a *different* Kaggle session; v16 is intentionally ~80 MiB
  heavier to keep the first layer's tape resident, and the 0.39 s wall-time difference between the
  two sessions is not a proven gain. Per-optimization attribution (how many MiB or seconds each of
  the five changes above is worth) was never measured — it would need one run per disabled block.
