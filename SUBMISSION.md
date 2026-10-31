# Submission: A Few Good Men

## Links
- Repository, final code commit: https://github.com/BAITC-Hacks/hack-623dda6b-a-few-good-men/commit/e3062178433394bb461dcb06794f0d62dd5274c0
  Branch `main` tip is this documentation commit; it differs from the code commit only in
  `README.md` and `SUBMISSION.md`, and `solution/` is byte-identical in both. The notebook's
  `COMMIT` cell pins `e306217`, so its runs use exactly the submitted `solution/`.
- Public Kaggle notebook, the saved version (built from `templates/kaggle_notebook.ipynb` without
  edits other than `REPO_URL`/`COMMIT`; the same link is in our README):
  https://www.kaggle.com/code/eralygainulla/soup-t4-runall-9c41
- Public mirror of the same commits (same SHA-1 history, so the notebook's pinned `COMMIT`
  resolves there too): https://github.com/Eraly-ml/soup-kernel-challenge-a-few-good-men — used
  because the platform repository is currently `private`, so an anonymous clone from a Kaggle
  notebook fails. The mirror and the platform repository are the same objects: `git log` shows
  `e306217` and `1bc211f` in both.
- Demo video (up to 3 minutes): *[not ready — missing deliverable, see Limitations]*
- Presentation: *[not ready — missing deliverable, see Limitations]*

## Team (up to 4 people)
| Name | Role | GitHub | Kaggle | Contact (captain) |
|---|---|---|---|---|
| *[to be filled by the captain]* | kernels, training, measurements | *[GitHub]* | eralygainulla | *[contact]* |

## Our kernels
Fourteen `@triton.jit` kernels run in every measured training step. The shares below are our own
timings from the validated run (`runs/result.json`, `metrics.own_kernel_ms` of `solution_1`,
divided by its timed training loop of 23.251 s, steps 3+). `harness/check_rules.py` check 4
independently reports **20.7%** for the same run (limit 5%).

| Kernel (`@triton.jit` name) | File | What it replaces | Share of the training time (check 4) |
|---|---|---|---|
| `nf4_columns_kernel` | `solution/nf4.py` | `bitsandbytes` NF4 dequantization into the column-major layout used by the `k_proj`/`o_proj`/`down_proj` matmuls | 6.10% |
| `nf4_decode_kernel` | `solution/nf4.py` | `bitsandbytes` NF4 dequantization into the row-major layout used by `q_proj`/`v_proj`/gate/up | 5.30% |
| `mlp_adjoint_kernel` | `solution/fused.py` | SwiGLU backward: `silu`/`mul` autograd nodes plus the gate/up gradient split | 2.47% |
| `mlp_kernel` | `solution/fused.py` | SwiGLU forward (`silu(gate) * up`) as one launch instead of three element-wise ops | 1.42% |
| `head_tile_kernel` | `solution/head.py` | `lm_head` + cross-entropy + entropy/accuracy over the 151,936-token vocabulary, replaced by an online-softmax pass over vocabulary tiles | 1.07% |
| `prepare_kernel` | `solution/fused.py` | `Qwen2RMSNorm` forward + LoRA input projection + dropout on `q`/`v` + writing the first layer's input tape (v16: the tape write is fused in here, removing a separate GPU copy) | 0.92% |
| `input_adjoint_kernel` | `solution/fused.py` | backward of the fused prepare step: RMSNorm backward, LoRA weight gradients, dropout unmask | 0.75% |
| `residual_kernel` | `solution/fused.py` | residual add + `Qwen2RMSNorm` forward for the post-attention norm | 0.66% |
| `assemble_kernel` | `solution/fused.py` | `apply_rotary_pos_emb` for q and k, q/k/v scatter into the packed layout, LoRA add | 0.60% |
| `merge_adjoint_kernel` | `solution/fused.py` | backward of the residual + norm block (frozen-weight bypass) | 0.58% |
| `split_adjoint_kernel` | `solution/fused.py` | RoPE backward (transpose rotation, no saved copy) | 0.49% |
| `replay_kernel` | `solution/fused.py` | recomputation of the dropout mask in backward instead of storing it as fp32/fp16 values | 0.30% |
| `rms_norm_bwd_kernel` | `solution/norm.py` | `Qwen2RMSNorm` backward where the fused variant does not apply | 0.03% |
| `rms_norm_fwd_kernel` | `solution/norm.py` | `Qwen2RMSNorm` forward where the fused variant does not apply | 0.01% |
| **Together** | | | **20.7%** |

`solution/rough.py` has one more `@triton.jit` function, `rough`, but it is called from inside the
kernels above and is not a separate launch, so it is not counted.

Ideas taken from other work, with links (we wrote all the code ourselves, no kernel code copied):
- NF4 quantization and double quantization of the block scales — QLoRA, Dettmers et al. 2023,
  https://arxiv.org/abs/2305.14314
- Fused linear layer + cross-entropy over vocabulary chunks with an online softmax — the idea used
  by Liger-Kernel (https://github.com/linkedin/Liger-Kernel) and Unsloth
  (https://github.com/unslothai/unsloth). We implement our own `head_tile_kernel`; neither library
  is installed or imported (check 2 reports `loaded: none`).
- Keeping activations in pinned host memory and streaming them back for the backward pass — the
  idea of ZeRO-Offload (https://arxiv.org/abs/2101.06840) and PyTorch activation offloading.
- Recomputing instead of storing (dropout masks, RoPE) — gradient checkpointing,
  https://arxiv.org/abs/1604.06174. We do not switch Soup's `gradient_checkpointing` mode; the
  recomputation happens inside our own backward sweep.

## Results on Kaggle T4 (from `runs/result.json` and `runs/checks.json` of the notebook above)
Two base and two solution runs in one session, jury order `base → solution → solution → base`,
Tesla T4, fp16. Means of the two runs per side.

| | Base Soup | Ours | Ratio |
|---|---|---|---|
| Time of `soup train`, s | 84.29 | **37.27** | **2.2616x speedup** (time x0.442) |
| Peak GPU memory, MiB | 14,348 | **556** | **25.8058x less** (memory x0.039) |
| Eval loss (public rows) | 2.04965 | 2.04965 | **+0.000%** (limit 0.5%) |

Per-run detail: base 87.60 s / 80.98 s wall, both 14,348 MiB peak; solution 37.37 s / 37.16 s wall,
536 / 556 MiB peak. Training loop 57.73 s base vs 24.35 s ours. Solution source SHA-256
`c622ff372b6a6259754104150f6812e075280f36c852a28b6c16da2e2ddf387d`, which is byte-identical to the
`solution/` package of the final commit.

The saved public notebook (`soup-t4-runall-9c41`, version 4) ran later on the same T4 and produced
`base 98.59 s / solution 40.11 s` wall, peak `14,345 / 533 MiB`, eval loss `2.04964 / 2.04956`
(`speedup 2.458x`, `memory base/solution 26.91x`, quality `-0.004%`) with all 10 checks passing
and own kernels at 20.0% of the training loop. We report it separately because the two sessions
were not measured back to back.

- [x] All 10 checks of `harness/check_rules.py` passed
      (1 runs finished; 2 no ready-made kernel libraries; 3 own kernels run in training;
      4 own kernels 20.7% >= 5%; 5 same task and Soup outputs identical within tolerance;
      6 loss curve first step 0.00%, mean 0.01–0.02%; 7 eval loss +0.000%;
      8 time x0.442, memory x0.039 of base; 9 clean measurement on one idle T4;
      10 Soup regression tests 79 passed)
- [x] README.md rewritten for our solution: what we did, how to run it, results, limits

## Presentation covers
Which operations we sped up and why (the 44% of element-wise work and the 152k-token softmax from
the task profile), how the kernels are built, results against the base for speed, memory and
quality, limits and next steps.

## Limitations
- The demo video and the presentation were not finished before the deadline; the links above are
  placeholders.
- The fast path is not a bit-exact replay of autograd. Parts of the backward round to fp16,
  `solution/rough.py` keeps four mantissa bits for some frozen-weight products and
  `solution/sums.py` accumulates in fp16 inside selected matmuls. On this run the eval loss is
  identical to the base to five decimals and the loss curve stays within 0.02%, but the guarantee
  is tolerance-based, not bitwise.
- The custom step only engages for this exact configuration: Qwen2.5-1.5B-Instruct, NF4, one
  default LoRA adapter on `q_proj`/`v_proj`, fp16, one batch per optimizer step, and an input size
  the packing path accepts. Anything else falls back to the standard TRL step with the weights
  moved back to the GPU, so the speedup does not generalize to other models or settings.
- The memory result depends on host memory: frozen weights and most layer activations live in
  pinned RAM. On a machine with little free RAM the streaming path would not fit.
- Streaming weights and activations over PCIe is what makes the peak 556 MiB, and it caps the
  speedup: two NF4 decode kernels alone are 11.4% of the training loop. A larger GPU-memory budget
  would trade that time back.
- v16 is deliberately ~80 MiB heavier than our earlier v15 candidate (556 vs 476 MiB in separate
  sessions) because it keeps the first layer's activation tape resident and writes it inside
  `prepare_kernel`. The 0.39 s wall-time difference between the two was measured in different
  Kaggle sessions, so we do not claim it as a speed gain.
