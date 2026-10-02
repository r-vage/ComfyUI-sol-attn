# Changelog

Changes to ComfyUI-sol-attn, grouped by date and version. Entries through v0.6.2 come from the original project and its contributors; their hardware measurements remain historical. Dates for those entries follow the preserved release-tag commits.

## 2026-10-02

### Version: 0.6.6

- **Fix**
  - Preserve positional and keyword attention arguments, including masks and reshape flags. Use the captured fallback for masks and precision opt-outs; reject conflicting arguments.
  - Protect H3 conditioning with the current model call's packed layout. Support older layout constructors, clear stale layout state, and fall back to dense attention when protection cannot be established without consuming the low-VRAM activation handoff.
  - Preserve explicit attention callbacks through modulation fusion and eager fallback. Use the original eager block for per-token modulation rows and propagate errors after residual mutation without retrying modified input.
  - Preserve captured third-party attention fallbacks when fixed or scheduled H3 patches exclude blocks. Resolve loaded-model forwards through the model patcher and retain unrelated wrappers.
  - Separate optional layout and Triton imports, retain useful dependency errors, and keep feed-forward chunking usable without the attention backend.
- **Refactor**
  - Share H3 patch installation between fixed and scheduled nodes while preserving node IDs, inputs, defaults, output types and kernel mathematics.
- **Docs**
  - Update English and Chinese READMEs for ComfyUI 0.30.1 through 0.37, capability-based fallbacks and the limits of current compatibility and generation validation.
  - Update node descriptions and hardware guidance for Linux Mint 22.3 and RTX 4070 Ti SUPER (Ada Lovelace, SM89). Retain Windows/RTX 5090 measurements as historical upstream results.
  - Document current-GPU attention benchmarks against SDPA and SageAttention, including timing methodology and sparse versus quantization error. Distinguish kernel throughput from full-generation speed and quality.
  - Identify r-vage as the continuation maintainer after the original repository and account became unavailable. Preserve Saganaki22/drbaph, contributor and NVIDIA credits and link the new repository.
  - Move release notes from both READMEs into the changelog and retain links beside the current version.
- **Chore**
  - Add pyproject.toml with ComfyUI metadata, current repository URLs, maintainer and original-author attribution, and optional attention and preview dependencies.
  - Add Eclipse's Comfy Registry publishing workflow.
  - Set the registry display name to ComfyUI Sol-Attn (continued by r-vage).

**Changed files:**

- `nodes.py`
- `minimax.py`
- `README.md`
- `README_ZH.md`
- `BENCHMARKS.md`
- `pyproject.toml`
- `.github/workflows/publish_action.yml`

## 2026-08-13

### Version: 0.6.2

- **Feat**
  - SM86 / RTX 30-series support — MiniMax H3 Sol Attention now supports Ampere SM86 GPUs through the existing pointer-kernel family used by SM89 and SM120. SM90/SM100/SM121 remain on their existing TMA paths.
- **Docs**
  - Community hardware validation — an RTX 3090 Ti completed a strict-mode 25,323-token MiniMax H3 workflow with Sol active, then completed a second generation with both `int8_qk` and `int8_pv` disabled. The residual-int8 QK/PV paths were also exercised successfully.
  - No numerical changes — this release enables an additional architecture without changing attention math, weights, sparsity settings, or output-quality behavior. Existing SM89, SM90, SM100, SM120, and SM121 dispatch is unchanged. SM86 performance has not yet been formally benchmarked.
- **Chore**
  - Regression coverage — the six repository tests applicable to SM86 plus the contributor's local runtime-stack check passed; the SM120-only pointer-vs-TMA comparison was correctly skipped. After merge, all seven repository tests also passed on SM120. Coverage includes architecture dispatch, pointer INT8 paths, H3 modulation fusion, and KJNodes handoff.

**Changed files:**

- `BENCHMARKS.md`
- `README.md`
- `README_ZH.md`
- `minimax.py`
- `nodes.py`
- `sol_kernel/fwd.py`
- `sol_kernel/preprocess.py`
- `sol_kernel/quant.py`

## 2026-08-10

### Version: 0.6.1

- **Fix**
  - KJNodes Low-VRAM compatibility — both MiniMax H3 Sol nodes now support the single-item activation-list handoff used by KJNodes' `MiniMax H3 Low VRAM Attention`. Sol peeks at the tensor while applying its eligibility gates, leaves the handoff intact for dense fallback, and consumes/releases it when the Sol path runs.
  - Combination covered by regression tests — `KJ MiniMax H3 Low VRAM Attention → MiniMax H3 Memory Efficient Sol Attention` and the scheduled Sol variant now run together without the previous `'list' object has no attribute 'shape'` error. KJ's early activation release is preserved on both sparse and dense calls.
- **Docs**
  - No numerical or kernel changes — attention math, model weights, SM89/SM120 pointer dispatch, SM90/SM100/SM121 TMA dispatch, and output accuracy are unchanged. All seven regression tests pass, and a real KJ low-VRAM block-forward GPU integration test also passed. The v0.6.0 benchmark matrix remains current.
  - Existing limitation remains — KJNodes' `MiniMax H3 Low VRAM Attention` still should not be combined with `MiniMax H3 Fused Modulation`, because both patch the complete H3 block forward. This release fixes its composition with the two local H3 Sol Attention nodes.

**Changed files:**

- `README.md`
- `README_ZH.md`
- `minimax.py`

## 2026-08-09

### Version: 0.6.0

- **Perf**
  - Faster SM120 forward dispatch — RTX 5090 now uses the pointer forward kernels, while SM89 remains pointer and SM90/100/121 remain TMA. At H3's `B=1, T=8192, H=56, D=128` shape, the SM120 pointer path measured 1.25× the TMA throughput in bf16 and produced bit-identical output; residual-int8 also remained bit-identical.
  - Inline residual-int8 Q preparation — SM89/SM120 diagonal-threshold pointer kernels quantize Q and derive the routing threshold from the BF16 Q tile already loaded by the forward. This removes the materialized Q-int8/Q-scale/threshold producer. At 32K H3 tokens it reduced measured peak allocation by 189 MiB; output matched the former path bit-for-bit for aligned/ragged lengths, exact sinks, and `int8_pv` on/off.
  - Fresh full release matrix — a warmed-cache rerun at 8K/16K/32K/65K puts bf16 at 1.38–1.65× SageAttention throughput, residual `int8_qk` at 1.73–1.97×, and opt-in `int8_qk+pv` at 1.98–2.33×. Accuracy stayed at relative L2 `0.00802`/`0.01396` versus the bf16 Sol path.
- **Feat (New)**
  - Exact MiniMax H3 modulation fusion — the new `MiniMax H3 Fused Modulation` node fuses segmented AdaLN scale/shift and gated residual updates across all 50 DiT blocks. It explicitly reproduces eager BF16 rounding and matched a real ComfyUI `DiTBlock` bit-for-bit. At 38,247 × 5,376, scale/shift measured 1.91× faster and gate/add 1.22× faster in isolation.
- **Feat**
  - Attention patches still compose — the fusion resolves each block's attention and MLP dynamically, so the recommended `global KJ Sage → H3 memory-efficient Sage → local H3 Sol` chain remains intact. It does not install an attention backend or change calls outside Sol.

**Changed files:**

- `BENCHMARKS.md`
- `README.md`
- `README_ZH.md`
- `minimax.py`
- `nodes.py`
- `sol_kernel/fwd.py`
- `sol_kernel/h3_fusion.py`
- `sol_kernel/preprocess.py`
- `sol_kernel/quant.py`

## 2026-08-09

### Version: 0.5.9

- **Perf**
  - Faster residual-int8 preprocessing — K's 64-token block-mean reduction and residual quantization now run in one Triton kernel with one read of K. On an RTX 5090, the isolated K/V-summary + K-quant preprocessing segment measured 26–36% faster at 8K, 16K, and 65K tokens (the 32K result was noisier). The residual-int8 formulation and FP32 accumulation are unchanged; validation found no routing changes.
  - All supported architectures keep their forward path — this is a shared preprocessing optimization. SM89 still uses the pointer forward kernels; SM90/100/120/121 still use the TMA forward kernels. No architecture dispatch was changed.
- **Docs**
  - KJNodes composition is documented explicitly — for the three-patch MiniMax H3 stack, apply global KJ Sage first, KJ's MiniMax memory-efficient Sage patch second, and this repository's MiniMax Sol patch last. Tokens that Sol declines use the captured memory-efficient Sage forward; attention calls outside that H3 object patch continue to use the global Sage override.

**Changed files:**

- `README.md`
- `README_ZH.md`
- `sol_kernel/preprocess.py`
- `sol_kernel/quant.py`
