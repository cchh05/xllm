# xllm_atb_layers patches for Path C prod v3

The `xllm_atb_layers` submodule ships upstream commit `1147537`, which
has two build-time issues in the `multi_lora_ch` container (arm A3,
CANN 8.5, no `PYTHON_INCLUDE_PATH` / `NPU_HOME_PATH` env vars). This
directory holds a single patch that must be applied to the submodule
before compiling xllm on this branch.

## Applying the patch

```bash
cd third_party/xllm_atb_layers
git apply ../../patches/xllm_atb_layers/build_hardcode_paths_and_sparse_stub.patch
```

Alternatively, from the repo root:

```bash
git -C third_party/xllm_atb_layers apply patches/xllm_atb_layers/build_hardcode_paths_and_sparse_stub.patch
```

Both work; nothing else in the tree depends on which method you use.

## What it fixes

### 1. `CMakeLists.txt` env-var expansion

`$ENV{PYTHON_INCLUDE_PATH}` and `$ENV{NPU_HOME_PATH}` are unset in the
`quay.io/jd_xllm/xllm-ai:xllm-dev-a3-arm-20260306` image. `cmake` then
emits empty `-I` flags and downstream compilation fails with:

- `torch/csrc/python_headers.h:12:10: fatal error: Python.h`
- `-I/opp/vendors/xllm/op_api/include: No such file or directory`

The patch hardcodes the four paths that work in this container:

- `/usr/include/python3.11`
- `/usr/local/lib64/python3.11/site-packages/torch/include`
- `/usr/local/Ascend/ascend-toolkit/latest`
- `/usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_1`

### 2. `sparse_flash_attention_operation.cpp` API mismatch

`aclnnSparseFlashAttentionGetWorkspaceSize` in `libcust_opapi.so`
(CANN 8.5) has a different argument list (int64_t vs pointer types)
than the caller expects (22-arg variant). Qwen3-8B never runs sparse
attention, so `SetAclNNWorkspaceExecutor` is stubbed to return -1.
Linker keeps the symbol; runtime never touches it.

## Verification

Applied on 2026-07-06. After apply + full rebuild:

- `libxllm_atb_layers.a` links cleanly
- xllm boots with `--lora-modules taboo=/path/to/adapter` and hits the
  Path C prod v3 preload chain end-to-end (see project_xllm_multilora
  memory for the SUCCESS log)

## Upstream

This is a container-specific patch, not an upstream bug. If we ever
move to a container image that sets `PYTHON_INCLUDE_PATH` /
`NPU_HOME_PATH` and ships a matching `libcust_opapi.so`, drop the
patch and use the vanilla submodule.
