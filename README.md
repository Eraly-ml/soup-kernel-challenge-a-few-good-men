# Soup Kernel Challenge — v16

This is the v16 solution for the fixed Qwen2.5-1.5B-Instruct QLoRA task on a Kaggle Tesla T4. The training code lives in `solution/`; the harness calls `solution.apply()` before constructing the model and trainer.

## What v16 does

- Streams frozen NF4 weights and the tied embedding from pinned host memory through two GPU slots.
- Packs non-padding tokens and runs a hand-written reverse sweep for LoRA gradients, without a PyTorch autograd graph for the decoder.
- Parks most layer activations in host memory; keeps the first layer's activation tape on GPU.
- Writes the first layer input tape inside the Triton `prepare_kernel`, avoiding a separate GPU copy.
- Processes the LM head and cross-entropy in vocabulary tiles instead of materializing all logits.
- Uses custom Triton kernels for NF4 decode, RMSNorm, fused decoder operations, and head statistics.

## Validation

[Private Kaggle validation notebook](https://www.kaggle.com/code/eralygainulla/soup-v15-kernel-cache-fixed) ran two base and two v16 solution training runs on Tesla T4. The solution source SHA-256 was `c622ff372b6a6259754104150f6812e075280f36c852a28b6c16da2e2ddf387d`. All 87 solution tests and all 10 harness checks passed.

| Mean of two runs | Base Soup | v16 |
| --- | ---: | ---: |
| Full wall time | 84.29 s | 37.27 s |
| Training loop | 57.73 s | 24.35 s |
| Peak GPU memory | 14,348 MiB | 556 MiB |
| Public eval loss | 2.04965 | 2.04965 |

The earlier v15 validation in a separate session measured 37.66 s and 476 MiB for its solution. Cross-session differences of a fraction of a second do not establish a speed gain for v16 over v15; v16 intentionally uses more GPU memory to keep one layer's tape resident.

## Reproduce

On Linux with a CUDA T4, install the pinned dependencies and run:

```bash
pip install "soup-cli[train]==0.77.0" "transformers==5.16.1" "peft==0.20.0" "trl==0.29.1" "bitsandbytes==0.50.2" "accelerate==1.14.0" pytest safetensors
python task/prepare_data.py
python -m pytest solution -q
python harness/run_task.py --repeats 2
python harness/check_rules.py
python harness/score.py runs/result.json
```

The custom fast path is limited to this task's fp16 Qwen2 configuration, one default LoRA adapter on `q_proj` and `v_proj`, and one batch per optimizer step. Other supported calls fall back to the standard trainer path.
