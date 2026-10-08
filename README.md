# ds4-convert

Convert **DeepSeek-V4 / V4.1-Flash** HuggingFace checkpoints (safetensors) to
`deepseek4` / `deepseek41` GGUF for the [joshua](https://github.com/rexlunae/joshua)
engine — the only engine today whose `deepseek4`/`deepseek41` loader is
logit-pinned against llama.cpp's `deepseek41.cpp` graph (hyper-connections,
compressed-stream attention, lightning indexer, engram n-gram tables and all).

Standalone, streaming, pure Rust. Never holds more than ~1 GB of intermediate
state per tensor (the engram tables alone are 384 M rows × 256).

## Why

The official `deepseek-ai/DeepSeek-V4.1-Flash` checkpoint is 510 GB
(FP8 dense + FP4-packed experts). The published GGUF conversions are
168–508 GB. On a 73 GiB-RAM box (e.g. a Ryzen 9950X + Arc B50) the smallest
of those still streams from NVMe. The REAP-pruned checkpoints
(LibertAIDAI `REAP-272E` / `REAP-256E`) are safetensors-only — this
converter turns them into joshua-loadable GGUFs with the quantization mix
you choose.

## Usage

```
ds4-convert --input /path/to/reap-272e --out model.gguf --preset balanced
```

- `--input` — the HF repo directory (`config.json`, `tokenizer.json`,
  `model.safetensors(.index.json)`), or a single safetensors file.
- `--preset`:
  - `parity` — lossless F32 (logit-parity testing)
  - `balanced` (default) — dense/experts/engram Q2_K, down-projections Q4_K,
    head Q8_0, router BF16, norms F32
  - `size` — like balanced but down Q2_K, head Q4_K
- `--<class>-quant` overrides any class: `embd head dense experts down
  shexp router hc engram static` (kinds: `f32 f16 bf16 q4_0 q8_0 q2_k q4_k`).
- `--dry-run` prints the full plan with sizes.
- `--engram-constants` / `--multipliers` / `--token-map` override the
  embedded V4.1-Flash engram assets (see below).

The output directory also gets a copy of `tokenizer.json` (joshua's layout).

## What is mapped

- **Metadata**: every `{arch}.*` key the joshua loader reads
  (`block_count`, MLA dims, indexer, compress ratios, hyper-connections,
  YaRN scaling, swiglu clamps, `expert_gating_func`, engram constants,
  tokenizer.ggml.*).
- **Tensor names**: HF → llama.cpp conventions
  (`layers.N.attn.wq_a.weight` → `blk.N.attn_q_a.weight`, …), including the
  routed-expert restack (`ffn.experts.E.w1/w3/w2` → stacked
  `ffn_gate_exps/ffn_up_exps/ffn_down_exps`) and shared experts.
- **Input dtypes**: BF16, F32, FP8 E4M3 with E8M0 (`ue8m0`) block scales
  (`weight_block_size` from config), and FP4 (E2M1, two per byte) for the
  routed experts.
- **Skipped**: vision tower, aligner, MTP draft blocks (`--keep-mtp` to keep).

## Engram constants

V4.1's engram n-gram tables need hash constants (multipliers, 48 primes,
offsets, a 129 280-entry token map). This repo embeds the exact constants
from the published `deepseek41` GGUFs as assets
(`assets/engram_constants.json`, `assets/token_map_v41_flash.bin`) —
stable across the V4.1-Flash checkpoint family (they depend only on the
tokenizer and config). The prime search itself is implemented and **unit-
tested to reproduce the published primes/offsets exactly**; only the numpy
`default_rng` multipliers are embedded rather than reimplemented.

## Verification

- `cargo test` — encoder/decoder round-trips per k-quant, name mapping,
  engram-constant reproduction, token-map checks.
- `tests/parity/` — an end-to-end harness (requires a sibling joshua
  checkout): a tiny deepseek41 model is written twice, once as GGUF directly
  and once as HF safetensors converted through the converter, and both are
  loaded with joshua's real `deepseek41` loader. All 14 positions × 16
  logits must match to 2e-5. `cd tests/parity && cargo run --release`.
- **Real-data validation (2026-10-07)**: expert-0 of layer 0 of the official
  `DeepSeek-V4.1-Flash` checkpoint — 20 rows × 5120 fp4 values, dequantized
  here — matches the lossless MXFP4 repack in the published
  vcruz305/DeepSeek-V4.1-Flash-GGUF Q8_0 file **exactly: 102400/102400
  (100.00%)**. The wrong nibble hypotheses score 9–15%. The FP4 packing
  convention (sequential, element 2i in the low nibble, 2i+1 in the high
  one; E8M0 biased-127 scales per 32 elements) is confirmed against the
  reference implementation.

## Known limits

- Down-projections land on Q4_K rather than the published Q3_K_S (same
  decoder family, ~9% larger); `--down-quant q2_k` if size matters more.
- FP4 nibble order defaults to sequential low-nibble-first — **validated
  exactly** against the reference implementation for the
  DeepSeek-V4.1-Flash family (see Verification). `--fp4-high-first` flips
  it for a different checkpoint family.
- The numpy multipliers for the engram hash are embedded, not regenerated.
- Checkpoints that prune the engram tables (e.g. REAP-272E) convert as
  no-engram models: the `{arch}.engram.*` metadata keys are written only
  when the engram tensors are actually present, so joshua's loader runs
  the model without the n-gram lookup.

## License

MIT
