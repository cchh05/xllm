# CANN 8.5 CPU→NPU Copy Thread Restriction Investigation

**Date**: 2026-07-06
**Branch**: `feature/lora-spike-hardcoded` @ `28d7dadd` (P0-B broadcast baseline)
**Host**: 82 (`A03-R40-I191-82-4100037.JD.LOCAL`) container `29fd45ce649b` (`quay.io/jd_xllm/xllm-ai:xllm-dev-a3-arm-20260306`)
**Hardware**: Ascend 910 A3 (arm64), CANN 8.5.0, torch_npu 2.7.1.post2

## Problem statement

xllm P0-A/M9 delivered end-to-end HTTP `/v1/load_lora_adapter` plumbing but
LoRA weight loading via `.to(device)` crashed with `aclrtMemcpy 107017 invalid
handle` from every worker thread pool tested. P0-B added broadcast wiring to
route the load onto worker `threadpool_` (assumed to be the model init
thread) — same failure. This investigation tries three routes to bypass the
restriction.

## Route B: raw `aclrtMemcpy` bypasses torch_npu opapi stream

**Hypothesis**: torch_npu's `.to(device)` uses a per-thread opapi memcpy stream
that is only primed on the model init thread. Native ACL `aclrtMemcpy(H2D)`
should bypass this and work from any thread that has done `aclrtSetDevice`.

**Implementation**: `LoRARuntime::load_and_activate` replaces `A_cpu.to(device)`
/ `B_cpu.to(device)` with a helper `cpu_to_npu_via_aclrt` that:
1. `aclrtSetDevice(dev_index)`
2. `torch::empty(shape, opts_on_device)` — allocates NPU memory (no copy)
3. `aclrtMemcpy(dst_ptr, nbytes, src_ptr, nbytes, ACL_MEMCPY_HOST_TO_DEVICE)`
4. `aclrtSynchronizeDevice()`

Called from `WorkerImpl::load_lora_adapter` on the worker's
`threadpool_` (post P0-B routing).

**Result**: FAIL

```
I20260706 12:59:19.166447  3443 adapter_loader.cpp:235] [LoRAAdapterLoader] /tmp/dummy_adapter/adapter_model.safetensors taken=14 skipped=0
I20260706 12:59:19.166477  3443 adapter_loader.cpp:70] [LoRAAdapterLoader] loaded 'biz-A-v1' r=8 scaling=2 target_modules=7 tensors=14
E20260706 12:59:19.167183  3443 lora_runtime.cpp:193] [LoRARuntime] Route B copy A failed for 'biz-A-v1': aclrtMemcpy H2D failed: 107017
```

**Conclusion**: CANN 8.5 native ACL `aclrtMemcpy(H2D)` is *also* per-thread
gated. The restriction is at driver level, not torch_npu opapi.

## Route F: static `--lora-modules` preload from model init thread

**Hypothesis**: If the restriction is thread-scoped, calling `.to(device)` from
inside the `QWen3ModelImpl` ctor (which runs on the WorkerImpl `threadpool_`
task that owns `set_device()` / `init_device_context()`) should succeed because
that is the exact thread the CANN docs say holds the primary device context.

**Implementation**: `qwen3.h` ctor, right after `set_model_device_dtype(...)`,
walks `LoRARuntime::config_snapshot().lora_modules` and for each entry:
1. `LoRARuntime::load_and_activate(name, path, "")` (queues CPU tensors)
2. `LoRARuntime::active_delta()` (promotes pending → active, still CPU)
3. `ad->A.to(options.device()).contiguous()` — the CPU→NPU copy
4. `cached_lora_A_.slice(0, 0, r).copy_(A_dev)` (bake into pre-allocated slot)

Launch: `--lora_modules=biz-A-v1=/tmp/dummy_adapter`.

**Result**: FAIL — from the model init thread itself

```
I20260706 13:07:57.352682  4318 lora_config.cpp:119] [LoRAConfig] enabled: max_loras=16 max_cpu_loras=64 max_lora_rank=32 target_modules=9 static_modules=1 runtime_updates=on
I20260706 13:07:57.353291  4318 lora_runtime.cpp:33] [LoRARuntime] initialised, enable=1
I20260706 13:08:00.160203  4356 lora_runtime.cpp:43] [LoRARuntime] model registered device=npu:0 dtype=BFloat16
I20260706 13:08:00.161468  4356 qwen3.h:131] [Path C] pre-allocated LoRA slots max_r=32 hidden=4096 device=npu:0 dtype=c10::BFloat16 (dummy content, per-adapter fill deferred to P0-B)
I20260706 13:08:00.161496  4356 qwen3.h:152] [Route F] preloading 1 static adapter(s) from --lora-modules
I20260706 13:08:00.161710  4356 lora_runtime.cpp:146] [LoRARuntime] queued 'biz-A-v1' id=1 A_cpu.shape=[8, 4096] B_cpu.shape=[4096, 8] scaling=2 (device migration deferred to first forward)
I20260706 13:08:00.161720  4356 lora_runtime.cpp:194] [LoRARuntime] promoted 'biz-A-v1' id=1 (CPU-side; caller performs .to(device))
E20260706 13:08:00.162580  4356 qwen3.h:189] [Route F] .to(device) failed for 'biz-A-v1': copy_between_host_and_device_opapi:build/CMakeFiles/torch_npu.dir/compiler_depend.ts:55 NPU function error: aclrtMemcpy, error code is 107017
frame #2: at_npu::native::copy_between_host_and_device_opapi(at::Tensor&, at::Tensor const&, aclrtMemcpyKind, bool) + 0xd00 (0xffff989d72f4 in /usr/local/libtorch_npu/lib/libtorch_npu.so)
```

Thread `4356` is the WorkerImpl `threadpool_` init task — the same thread that
just ran `set_device()` / `init_device_context()` two lines above and
`torch::empty(device)` for the pre-allocated slot successfully. Yet
`.to(device)` from the same thread crashes.

**Conclusion**: The "model init thread can copy" premise is FALSE on this
stack. Even the primary device context holder cannot use torch_npu opapi
memcpy for HtoD.

## Route A: dedicated executor with warmup — logically excluded

**Hypothesis**: Give `LoRARuntime` its own single-thread executor. Warm it once
by calling `aclrtSetDevice + torch::empty(device) + copy_(cpu)` from the
model init thread's context, then all subsequent loads marshal onto this
warmed thread.

**Result**: NOT ATTEMPTED — logically dead.

**Rationale**: If the model init thread itself cannot do `.to(device)` (Route F
result), then a hand-crafted executor using the same `.to(device)` mechanism
cannot succeed. A separate thread starts with strictly less privileged NPU
context than the model init thread.

## Root cause

xllm's base weight loading path
(`WorkerImpl::init_model` → `model->load_model` → `xllm_atb_layers::set_weight`)
does NOT use torch_npu opapi memcpy. It uses the atb-managed CANN stream,
which is what's actually wired for the primary device context on this kernel.

`torch_npu`'s `copy_between_host_and_device_opapi` — the code path that both
`.to(device)` and `Tensor::copy_` cross-device use — appears entirely broken
for HtoD on this build (2.7.1.post2 + CANN 8.5.0 + arm64 A3), regardless of
which thread invokes it.

## Sole viable path forward: Path B

Path B threads LoRA A/B tensors through the *same* atb `set_weight` channel
that base weights use. Because it never touches torch_npu opapi memcpy, it
bypasses the entire class of failures documented above.

Cost: rework `xllm/core/layers/npu/loader/qwen3_decoder_loader.cpp:344` to
stop packing base QKV into a single `IN_Q_WEIGHT` slot (must be per-proj so
atb's `in_qkv_lora_a_0/b_0` etc. slots line up), plus enable
`param.enableLora = true` / `loraEnableGMM = true`, extend
`WEIGHT_COUNT_PER_LAYER` from 56 to 70 (14 new LoRA slots per layer), and
generate `in_seq_len_cum_sum` in the model forward path. See
`project_xllm_multilora.md` §"关键红利" for the full 5-file change list.

## Files touched during this investigation

Uncommitted diffs on branch `feature/lora-spike-hardcoded`:

- `xllm/core/framework/lora/CMakeLists.txt` +11 (add ACL include for USE_NPU)
- `xllm/core/framework/lora/lora_runtime.h` +1 (add `config_snapshot()` accessor)
- `xllm/models/llm/npu/qwen3.h` +59 (Route F preload block in ctor)
- `xllm/core/layers/npu/npu_qwen3_decoder_layer_impl.cpp` -1+1
  (spike Day 5b `useQKNorm=false`, unrelated to Route B/F)

`lora_runtime.cpp` was temporarily replaced with the Route B version
(`aclrtMemcpy` direct call) but restored to P0-B `.to(device)` version after
Route B failed. All four `*.bak_pre_route_b` backup files remain in the
source tree.

## Rollback plan

1. Revert Route B/F changes in `xllm/models/llm/npu/qwen3.h` (drop the
   `[Route F]` block) and `xllm/core/framework/lora/lora_runtime.h` (drop
   `config_snapshot()`).
2. Keep the CMakeLists.txt ACL include patch — it's genuinely needed if
   NPU_HOME_PATH is empty.
3. Relink xllm binary to get a clean P0-B baseline.
4. Save Route B `.cpp` and Route F patches on a side branch
   (`feature/lora-route-bf-attempts`) so the exploration is preserved.

## Build environment quirks (write down for future rebuilds)

New container's `NPU_HOME_PATH` is empty. Root CMakeLists.txt's include
directives silently expand to `-I/include -I/opp/vendors/xllm/op_api/include`
so acl.h and cust_opapi lookups fail unless we pass explicit paths.

Manual compile command surgery required:
1. Add `-I/usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/include` for `acl/acl.h`
2. Change `atb/cxx_abi_0/include` → `atb/cxx_abi_1/include` (07-06 latest symlink no longer has 0)
3. Add `-I/usr/include/python3.11` for `Python.h`

Manual link command surgery required:
1. `LIBRARY_PATH` must include `/export/home/caihao.40/xllm_multilora/third_party/xllm_ops/build`
   for `-lcust_opapi`
2. `sed -i 's|-Wl,-rpath,|-Wl,-rpath-link=/usr/local/lib64/python3.11/site-packages/torch.libs -Wl,-rpath,|'`
   to satisfy `libgfortran-*.so` transitive from libopenblasp

Runtime: `export LD_LIBRARY_PATH=/export/home/caihao.40/xllm_multilora/third_party/xllm_ops/build:$LD_LIBRARY_PATH`
before launching xllm, or `libcust_opapi.so: cannot open shared object file`.
