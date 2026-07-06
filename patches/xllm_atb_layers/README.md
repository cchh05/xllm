# atb_speed Path B patches

Path B (atb_speed native LoRA channel via NoPack QKV) requires patching
the third-party xllm_atb_layers library (as of xllm_multilora submodule
commit 1147537). This directory holds the patch that must be applied
to xllm_atb_layers before rebuilding.

## Applying the patch

```bash
cd third_party/xllm_atb_layers
git apply ../../patches/xllm_atb_layers/pathb_nopack_qk_norm.patch
```

Or from anywhere in the tree:

```bash
git -C third_party/xllm_atb_layers apply patches/xllm_atb_layers/pathb_nopack_qk_norm.patch
```

## What it fixes

Bug in atb_speed 8.5 combined with `useQKNorm=true` on the NoPack QKV path:

- `ConstructQKVTensorMap` only registers `qk_norm` inTensor + intermediate
  slots when `isPack=true`. NoPack + useQKNorm leaves them unregistered,
  causing `AddFAttnQKVLinearSplitNode` to reference undefined tensor
  indices and Attention graph to fail with
  `node[0].inTensorIds.size 25 != operation.inputNum 23`.
- `QKVLinearSplit` NoPack branch never calls `AddQKNormNode`, silently
  dropping QK Norm on the NoPack path.
- `AddQNormLinearNode` / `AddKNormLinearNode` unconditionally emit to
  `out_q` / `out_k` regardless of whether QK Norm should be applied
  afterward.

Fix (four hunks):

1. Move `qk_norm` register in `ConstructQKVTensorMap` outside the
   `if (isPack)` guard, gated instead on `param.useQKNorm`.
2. `AddQNormLinearNode` outputs `intermediate_q` on NoPack + useQKNorm
   (else `out_q`).
3. `AddKNormLinearNode` outputs `intermediate_k` on useQKNorm
   (else `out_k`).
4. `QKVLinearSplit` NoPack branch calls `AddQKNormNode` when useQKNorm.

Combined effect: on NoPack + useQKNorm, Q/K/V linears output to
intermediate slots, `AddQKNormNode` applies RMSNorm to `intermediate_q`
and `intermediate_k` producing the final `out_q` / `out_k` (V linear
skips QK Norm and outputs directly to `out_v`, which matches Qwen3's
architecture).

## Verification

Applied on 2026-07-06, xllm boots successfully with NoPack QKV +
useQKNorm=true and Qwen3-8B chat request returns coherent Chinese
output. Boot log shows no `Attention check failed` errors.

Prior to this patch, xllm boot crashed with `Check failed:
'node.operation' Must be non NULL` at
`xllm/core/layers/npu/npu_qwen3_decoder_layer_impl.cpp:229`.

## Upstream

This is a bug in xllm_atb_layers itself and should be reported upstream
to CANN/atb-speed. The fix is a small, additive change with no impact
on the existing Pack path -- the four hunks either add new NoPack code
or gate old Pack code more tightly.
