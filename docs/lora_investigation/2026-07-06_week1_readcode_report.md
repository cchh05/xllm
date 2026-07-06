# Week 1 read-code report: atb_speed NoPack + native LoRA on Qwen3-8B

**Date**: 2026-07-06
**Branch**: `feature/lora-p0c-pathb` @ `28d7dadd`
**Purpose**: Verified file:line references and slot-layout facts that seed
Week 2 (loader NoPack) and Week 3 (open atb LoRA channel).

## TL;DR

The plan is workable, but three assumptions from `2026-07-06_p0c_pathb_plan.md`
need to be corrected before touching code:

1. **`packQuantType` gate**: setting it to `PACK_QUANT_UNDEFINED` with
   `linearDescs = [BF16 x 7]` does NOT force NoPack -- `CheckPack` returns
   true when all seven descs agree. Force NoPack via
   `packQuantType = MIX_W8A8` (or any `MIX_*` enum outside the pack
   allowlist). This is a one-line change to `initialize_quantization_parameters`.
2. **`WEIGHT_COUNT_PER_LAYER` bump 56 -> 70 has a hidden atb ordering
   dependency**: atb's `ConstructInTensorMap` puts LoRA slots AFTER the
   runtime block, so the atb graph indices are `[67..81]`, not
   `[68..82]`. And even that ordering means treating LoRA as static
   weights would require reordering atb's inTensorList (a third-party
   library change we should avoid). **Recommendation: keep
   `WEIGHT_COUNT_PER_LAYER = 56`**; wire the 15 LoRA slots as runtime
   tensors in `build_node_variant_pack`, exactly like `cos_pos`/`sin_pos`
   are wired today.
3. **Loader change surface is larger than "line 283"**: the K/V weights
   are wiped to `at_placeholder_ = torch::zeros({1})` at lines 308-311
   (BF16 path) AND K/V dequant/scale/bias/offset at 234-243 (W8A8 path).
   Path B has to preserve K/V through both wipe blocks, not just skip the
   `torch::cat`.

## Verified file:line reference

### Loader today (BF16 path)

`xllm/core/layers/npu/loader/qwen3_decoder_loader.cpp`:

- L21-87: enum. `IN_Q_WEIGHT=4, IN_K_WEIGHT=10, IN_V_WEIGHT=16`. K_NORM=54, Q_NORM=55.
- L89-100: `WEIGHT_MAPPING` -- `q_proj` -> IN_Q_WEIGHT (91), `k_proj` -> IN_K_WEIGHT (92), `v_proj` -> IN_V_WEIGHT (93). Loaded independently via `set_weight` at L201-207.
- L177: `at_placeholder_ = torch::zeros({1})`.
- L217: `merge_loaded_weights()` function.
- **L283-287**: `torch::cat({IN_Q_WEIGHT, IN_K_WEIGHT, IN_V_WEIGHT}, 0).transpose(0,1)` -> new_q_weight.
- **L289-290**: `npu_format_cast(FRACTAL_NZ)` on the fused Q slot.
- L292: FRACTAL_NZ on `IN_ATTENTION_OUT_WEIGHT` (o_proj).
- L301: FRACTAL_NZ on fused MLP gate+up (writes to `IN_MLP_W2_WEIGHT`).
- L304: FRACTAL_NZ on `IN_MLP_CPROJ_WEIGHT`.
- **L308-311**: BF16 wipe. `IN_MLP_W1_WEIGHT, IN_K_WEIGHT, IN_V_WEIGHT, IN_K_BIAS, IN_V_BIAS` -> placeholder.
- L234-243: W8A8 wipe. `IN_K_DEQSCALE/IN_V_DEQSCALE/IN_K_BIAS/IN_V_BIAS/IN_K_OFFSET/IN_V_OFFSET/IN_K_SCALE/IN_V_SCALE` -> placeholder before the L283 cat runs.

### atb_speed NoPack branch

`third_party/xllm_atb_layers/operations/fusion/attention/qkv_linear_split.cpp`:

- L91: `bool isPack = CheckPack(param.packQuantType, param.layerLinearDescs, qkvLinearIndex)` (map ctor).
- L476: `bool isPack = CheckPack(...)` (graph builder).
- L481: graph name `"QKVLinearSplitPack"` or `"QKVLinearSplitNoPack"`.
- **L493**: `AddQNormLinearNode` runs unconditionally.
- **L503-514**: `else` branch -- runs `AddKNormLinearNode` (513) + `AddVNormLinearNode` (514). Skips `AddSplitQKVNode` / `AddSplitMixedQKVNode` entirely.
- L42-45: candidate LoRA name group -- `"in_seq_len_cum_sum", "in_qkv_lora_a_0/b_0/1/1/2/2"`.

**Base weight slots per linear** in NoPack:

| Linear | Node | Base weight slots (from atb) |
|---|---|---|
| Q | `AddQNormLinearNode` (L137) | in_qkv_weight_0, scale_0, offset_0, descale_0, bias_0, compress_idx_0 (L183-188) |
| K | `AddKNormLinearNode` (L294) | in_qkv_weight_1, scale_1, offset_1, descale_1, bias_1, compress_idx_1 (L313-317) |
| V | `AddVNormLinearNode` (L378) | in_qkv_weight_2, scale_2, offset_2, descale_2, bias_2, compress_idx_2 (L397-402) |

**LoRA slots per linear** (guarded by `if (param.supportLora)`, wired from `LayerParam::enableLora` at `models/base/layer/decoder_layer.cpp:487`):

| Linear | LoRA slots |
|---|---|
| Q | `in_qkv_lora_a_0`, `in_qkv_lora_b_0` (L199-200) |
| K | `in_qkv_lora_a_1`, `in_qkv_lora_b_1` (L323-324) |
| V | `in_qkv_lora_a_2`, `in_qkv_lora_b_2` (L408-409) |

Every LoRA-guarded branch pushes `in_seq_len_cum_sum` first.

### `packQuantType` / `CheckPack` truth table

**Set at**: `xllm/core/layers/npu/npu_qwen3_decoder_layer_impl.cpp:114-151` `initialize_quantization_parameters`:
- BF16 path (L124-125): `param.packQuantType = {PACK_QUANT_UNDEFINED, PACK_QUANT_UNDEFINED}`, `linearDescs = {BFLOAT16, INVALID, INVALID, BFLOAT16, BFLOAT16, INVALID, BFLOAT16}` (L117-123).
- W8A8 path (L141-142): `param.packQuantType = {ALL_W8A8, ALL_W8A8}`.

**Threading**: `models/base/layer/decoder_layer.cpp:473`:
`fusionAttentionParam.packQuantType = this->param.packQuantType.at(0)`.

**Enum**: `third_party/xllm_atb_layers/operations/fusion/utils.h:46-77`. 26 values total.

**`CheckPack` at `utils.cpp:274-316`** (in order of evaluation):
1. If `packQuantType` in `PackableQuantTypes` (`ALL_FP, ALL_W8A16(_ANTI), ALL_W4A16(_ANTI), ALL_W8A8(_ANTI), ALL_W8A8SC(_ANTI), ALL_W8A8_DYNAMIC(_ANTI), ALL_W4A8(_ANTI)`): return **true**.
2. If `packQuantType != PACK_QUANT_UNDEFINED` and not in whitelist: return **false**. **Any `MIX_*` value forces NoPack.**
3. If `packQuantType == PACK_QUANT_UNDEFINED`: walk `linearDescs[qkvLinearIndex]`; skip INVALID; if all valid descs agree return **true**; if any two disagree return **false**.

**Consequence for BF16 today**:
`packQuantType = PACK_QUANT_UNDEFINED`, `linearDescs = [BF16, INVALID, INVALID, ...]` -> only the Q slot is a valid desc, all valid descs "agree" trivially -> `CheckPack` returns **true** -> Pack path taken.

**Path B NoPack trigger**: change L124-125 to
`param.packQuantType = {MIX_W8A8, MIX_W8A8};`
Alternatively make three `linearDescs` disagree, but the enum-based path is
cleaner and has zero effect on runtime quant math (the descs are the source
of quant behaviour).

### `WEIGHT_COUNT_PER_LAYER = 56` composition

At `xllm/core/layers/npu/npu_qwen3_decoder_layer_impl.cpp:33`.

Composition (walked through `QwenDecoderLayer::ConstructInTensorMap` at
`third_party/xllm_atb_layers/models/qwen3/layer/decoder_layer.cpp:51-77`
with current qwen3 config `useQKNorm=true, enableIntraLayerAddNorm=true`,
LoRA off):

| Group | Slots | Range |
|---|---:|---|
| input_norm_weight | 4 | 0..3 |
| attn_weight | 24 | 4..27 |
| post_attn_norm_weight | 4 | 28..31 |
| mlp_weight | 18 | 32..49 |
| add_rmsnorm_quant | 4 | 50..53 |
| qk_norm | 2 | 54..55 |
| **Total** | **56** | |

### LoRA slot names (14 weight tensors + 1 runtime scalar)

At `third_party/xllm_atb_layers/models/base/layer/decoder_layer.cpp:70-77`:

```
lora_common: in_seq_len_cum_sum       # runtime scalar, per-forward
lora_attn:   in_qkv_lora_a_0, in_qkv_lora_b_0,  # Q
             in_qkv_lora_a_1, in_qkv_lora_b_1,  # K
             in_qkv_lora_a_2, in_qkv_lora_b_2,  # V
             in_qkv_dense_lora_a, in_qkv_dense_lora_b   # o_proj (8 total)
lora_mlp:    in_mlp_lora_a_0, in_mlp_lora_b_0,  # gate
             in_mlp_lora_a_1, in_mlp_lora_b_1,  # up
             in_mlp_down_lora_a, in_mlp_down_lora_b   # down (6 total)
```

Total = 14 static weights + 1 runtime = 15 additional slots per layer when
`enableLora=true`.

### atb graph inTensorList ordering

From `models/base/layer/decoder_layer.cpp:117-204`. The default order is:

```
[0..49]  static base weights (attn + mlp)
[50..53] add_rmsnorm_quant
[54..55] qk_norm
[56..66] runtime "default" tensors (in_hidden_states, in_position_ids, ...)
[67]     in_seq_len_cum_sum       (only when enableLora)
[68..75] 8 LoRA attn weights      (only when enableLora)
[76..81] 6 LoRA mlp weights       (only when enableLora)
```

**LoRA slots sit AFTER the "default" runtime block**, at atb-graph indices
`[67..81]`, not `[68..82]` as the plan wrote. The value 82 in the plan
overshoots by 1.

### Downstream code that does NOT assume packed QKV

Cross-checked -- no offenders:

- `xllm/models/llm/npu/qwen3.h`: no `IN_Q_WEIGHT`, no `split_with_stride`, no `slice`/`view` on QKV. Layer is opaque.
- `xllm/core/layers/npu/npu_qwen3_decoder_layer_impl.cpp`: iterates `atb_weight_tensors_[0..WEIGHT_COUNT_PER_LAYER]` at L222-225 and L364-368; runtime tensors bound at L295-347 starting from `WEIGHT_COUNT_PER_LAYER`. No references to `IN_Q_WEIGHT`.
- **RoPE**: applied inside `AddSplitQKVNode` (`qkv_linear_split.cpp:227-254`) which ONLY runs on the Pack path. NoPack skips it (L512-514). Q/K/V come out of three independent linears; downstream RoPE + attention nodes see three separate tensors regardless of pack mode.
- **KV cache**: `KVCache& kv_cache` is bound as two separate tensors -- `kv_cache.get_k_cache()` and `kv_cache.get_v_cache()` (L341-344). No fused Q+K+V head-dim assumption. **NoPack-safe out of the box** (inference, needs empirical verification).
- **`splitWithStride`** in the Pack path uses `param.selfAttentionParam.headNum / kvHeadNum, 1, 1` (`qkv_linear_split.cpp:232`) implicitly assuming Q+2*KV interleaved. Only executes on Pack -- not our concern.

**The pack assumption is entirely contained to the loader.**

### Sanity checks flagged

- `npu_qwen3_decoder_layer_impl.cpp:52` sets `param.loraEnableGMM = false;` and there is NO `param.enableLora = true` in that file. Path B needs both flipped in `param_from_args`.
- `param.linearHasBias = {0, 0, 0, 0}` at L74 -- `qkvHasBias=false`. **Qwen3-8B HF checkpoint may or may not have QKV bias**. Need to `safe_open` on `/export/home/models/Qwen3-8B/*.safetensors` and check for `q_proj.bias` etc. before Week 2 loader edits, otherwise base baseline chat will break.
- `models/base/param/param.h:94` has `bool enableLora = false;` as the top-level flag; it's threaded down to `fusionAttentionParam.supportLora` correctly.

## Corrections vs original P0-C plan (2026-07-06_p0c_pathb_plan.md)

**Original plan Week 2**:
> `packQuantType` -> `PACK_QUANT_UNDEFINED`, `layerLinearDescs` -> explicit `[BF16 x 7]`

**Corrected**:
> `packQuantType` -> `MIX_W8A8` (any `MIX_*` value works). `layerLinearDescs`
> may stay `[BFLOAT16, INVALID, INVALID, BFLOAT16, BFLOAT16, INVALID, BFLOAT16]`
> since `CheckPack` returns false immediately on the enum.

**Original plan Week 3**:
> `WEIGHT_COUNT_PER_LAYER` 56 -> 70 (+14 LoRA slot), attach to variantPack indices 68..82

**Corrected**:
> `WEIGHT_COUNT_PER_LAYER` stays 56. The 14 LoRA weight tensors + 1
> `in_seq_len_cum_sum` are wired as runtime tensors in
> `build_node_variant_pack`, at variantPack indices 56 + N (mirroring how
> `cos_pos`/`sin_pos` are wired). This keeps atb's inTensorList ordering
> intact (LoRA at atb-graph indices `[67..81]`, matching what atb expects).

**Original plan Week 2 loader edit**:
> Delete `torch::cat({Q, K, V}, 0)` at L283. Cast K/V to FRACTAL_NZ independently.

**Corrected -- expanded surface**:
1. Delete the `torch::cat` at L283-287.
2. Apply `npu_format_cast(FRACTAL_NZ)` to `IN_Q_WEIGHT`, `IN_K_WEIGHT`,
   `IN_V_WEIGHT` independently (three casts).
3. Remove `IN_K_WEIGHT, IN_V_WEIGHT, IN_K_BIAS, IN_V_BIAS` from the L308-311
   BF16 wipe list (keep `IN_MLP_W1_WEIGHT` in the list -- MLP gate+up
   fusion is independent of Path B, MLP fusion stays as-is).
4. On W8A8 path, remove `IN_K_DEQSCALE/IN_V_DEQSCALE/IN_K_BIAS/IN_V_BIAS/
   IN_K_OFFSET/IN_V_OFFSET/IN_K_SCALE/IN_V_SCALE` from the L234-243 wipe list.
5. Also verify Qwen3-8B HF checkpoint has (or doesn't have) QKV bias --
   if bias exists, `param.linearHasBias` at L74 needs updating.

**Original plan Week 3 M2 loader**:
> Extend M2 LoRAAdapterLoader to emit three independent Q/K/V A/B pairs.

**Corrected**:
> The taboo-ship adapter is already stored per-proj (504 keys, 36 layers x
> 7 modules x 2 A/B). M2 already canonicalizes to `<subkey>#A/#B`. **No M2
> extension needed** -- the extension is only needed if we get an adapter
> that stores fused QKV, which is not what open-source PEFT adapters do.

## Deliverables in this branch

- `docs/lora_investigation/2026-07-06_week1_readcode_report.md` (this file)
- Path B implementation touch points (revised, actionable checklist for Week 2):

### Week 2 checklist (revised, actionable)

Files to edit, in order:

1. **`xllm/core/layers/npu/npu_qwen3_decoder_layer_impl.cpp:117-142`**
   `initialize_quantization_parameters`: BF16 branch
   `param.packQuantType = {MIX_W8A8, MIX_W8A8}`.
2. **`xllm/core/layers/npu/loader/qwen3_decoder_loader.cpp:283-311`**
   `merge_loaded_weights`:
   - drop cat, keep three independent Q/K/V slots
   - three FRACTAL_NZ casts (Q, K, V)
   - shrink placeholder wipe list to just `IN_MLP_W1_WEIGHT`
3. **W8A8 path L234-243**: shrink K/V wipe list.
4. **Verify Qwen3-8B HF bias**: `python3 -c "from safetensors import safe_open; ..."` on `/export/home/models/Qwen3-8B/*.safetensors`. Update `param.linearHasBias` L74 if needed.
5. **Regression test loop**: after every edit, rebuild + start Qwen3-8B baseline (LoRA off) + curl chat + compare to v0.9.0-chunk output. NoPack Qwen3 baseline chat must be byte-for-byte equal to v0.9.0-chunk before entering Week 3.

### Test adapter ready

`/export/home/caihao.40/lora_adapters/qwen3-8b-taboo-ship/`:
- 349 MB safetensors
- `base_model_name_or_path = "Qwen/Qwen3-8B"` (matches our base)
- `r=32`, `lora_alpha=64` -> scaling = 0.5 (standard, `use_rslora=false`)
- 504 keys = 36 layers x 7 modules (q/k/v/o/gate/up/down) x 2 (A/B)
- M2 loader eats it as-is (`.lora_A.weight`/`.lora_B.weight` PEFT format)
