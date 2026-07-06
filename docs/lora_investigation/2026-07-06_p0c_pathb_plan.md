# P0-C Path B Roadmap (Corrected)

**Date**: 2026-07-06
**Baseline**: `feature/lora-spike-hardcoded` @ `28d7dadd` (P0-B, HTTP + broadcast working, shared dummy weights)
**Development branch**: `feature/lora-p0c-pathb` (branched from `28d7dadd` today)
**Goal**: end-to-end PEFT weight loading through xllm_atb_layers native LoRA channel
        so `--lora-modules a=/p1,b=/p2,c=/p3` starts, tenant requests with
        `model="a"/"b"/"c"` produce output aligned with offline-merge references,
        QPS regression ≤ 15%.

## Status of prior investigation

| Route | Status | Reason |
|---|---|---|
| Route A/B/F (`.to(device)` / `aclrtMemcpy`) | 2026-07-06 falsified on 82 | CANN 8.5 opapi memcpy is broken for HtoD from every in-process thread including model init. See `docs/lora_investigation/2026-07-06_route_bf_dead_ends.md` |
| Path A (atb_speed native LoRA + QKV pack) | 2026-07-03 falsified | `NormLinear -> FusionLinearWithLora -> AddLoraA` sub-graph tensor-id shift. Third-party bug. |
| **Path B** (Qwen3 base QKV **NoPack** + atb_speed native LoRA) | **only path proven feasible** | LoRA weights ride the same `xllm_atb_layers::set_weight` -> atb-managed CANN stream that base weights use, bypassing torch_npu opapi entirely. |

## Path B semantic guarantee

LoRA A/B tensors are attached to atb decoder-layer weight slots (indices 56..69
per layer once `WEIGHT_COUNT_PER_LAYER` is bumped 56 -> 70) via
`base_layer.h merge_loaded_weights` -> `AtTensor2Tensor`, which is exactly the
codepath that already loads Qwen3 base weights successfully on 82. This is the
only proven way to move CPU tensors onto NPU on this stack.

## Corrected timeline (7-9 weeks, not 5-7)

The original plan's 5-7 week estimate is optimistic on three fronts we now
call out explicitly:

- Week 2 (loader NoPack) has cross-module blast radius (rope/kv-cache/split_with_stride)
  we cannot fully bound before entering it.
- Week 5 scheduler + GroupMatmul work is more subtle than "~5 lines".
- CI infra is not a 1-week task; MVP acceptance uses smoke tests + a manual
  regression checklist instead.

## Phase 1: NoPack loader + native LoRA channel (4-5 weeks)

### Week 1 - branch + read atb_speed NoPack

Day 1 (already done):
- `feature/lora-route-bf-attempts` branch pushed with Route B + Route F
  attempts and evidence doc (`ce5c58d9` and `d0cdd96d`)
- `feature/lora-spike-hardcoded` rolled back to `28d7dadd`, xllm binary
  relinked from clean sources, `strings | grep "Route [BF]"` returns 0
- P0-B smoke: HTTP `/v1/load_lora_adapter` returns `status:ok`, `[Path C] pre-allocated`
  log clean, no Route B/F strings in binary

Day 2:
- Branch off `28d7dadd` -> `feature/lora-p0c-pathb`
- Fetch an open-source Qwen3-8B LoRA from HuggingFace (any published adapter
  targeting Qwen3-8B qkv/mlp; used only to validate loader + forward, not to
  measure task quality). **Not training our own** per user decision.

Day 3-5:
- Read `qkv_linear_split.cpp` NoPack branch (line ~476-580):
  * `AddQNormLinearNode` / `AddKNormLinearNode` / `AddVNormLinearNode`
  * confirm each independent linear owns its own `in_qkv_lora_a_i/b_i` slot
    (Path A investigation already suggested this; verify empirically)
- Read `qwen3_decoder_loader.cpp:283` where `torch::cat({Q, K, V}, 0)`
  currently packs into `IN_Q_WEIGHT`
- Read `qwen3_decoder_loader.cpp:337` `merge_loaded_weights` NZ format
  cast (`ACL_FORMAT_FRACTAL_NZ`), see whether it needs to run on 3
  independent K/V slots or only Q
- Confirm `packQuantType`, `layerLinearDescs`, and `CheckPack` gating
- Read `xllm/models/llm/npu/qwen3.h` split_with_stride / KV cache slot
  math and RoPE apply -- flag every place that assumes packed head layout
- Deliverable: **read-code report** listing exact touch points for Week 2

### Week 2 - loader NoPack + Qwen3 baseline chat regression

Loader edits:

- `qwen3_decoder_loader.cpp:283` -- remove `torch::cat({Q,K,V}, 0)`,
  keep three independent tensors
- `qwen3_decoder_loader.cpp:337` -- `npu_format_cast(FRACTAL_NZ)` on Q, K,
  V independently
- `packQuantType` -> `PACK_QUANT_UNDEFINED`
- `layerLinearDescs` -> explicit `[BF16, BF16, BF16, BF16, BF16, BF16, BF16]`
  (Q/K/V/o/gate/up/down each BF16)

Regression:

- **Every day** rebuild + start Qwen3-8B baseline (no adapter, LoRA path off)
  and diff chat output against v0.9.0-chunk binary reference. Any divergence
  -> stop and diagnose before proceeding.
- Track downstream fallout in `split_with_stride`, RoPE, KV cache slot
  layout as it surfaces.

**Escalation gate (mandatory)**:

If NoPack Qwen3-8B baseline chat is not byte-for-byte equal to v0.9.0-chunk
by end of Week 2, we escalate to design review before entering Week 3.
Options at that point:
- (a) keep both packed and unpacked base weight in memory (~+6 GB HBM per
  layer, acceptable if only loader path is affected)
- (b) revisit whether atb_speed NoPack is really the only-fit; look at
  atb GroupMatmul-ish alternatives that support QKV pack
- (c) accept a longer runway if the fix is bounded

We do NOT push forward on a broken NoPack baseline hoping Week 3 will
compensate.

### Week 3 - open atb_speed native LoRA channel

5-file change set (all touch points located during Path A investigation):

| File | Change |
|---|---|
| `npu_qwen3_decoder_layer_impl.cpp:52` | `param.enableLora = true; param.loraEnableGMM = true;` |
| `npu_qwen3_decoder_layer_impl.cpp:20` | `WEIGHT_COUNT_PER_LAYER` 56 -> 70 |
| `qwen3_decoder_loader.cpp:81-114` | `WEIGHT_MAPPING` extend with 14 PEFT `.lora_A.weight` / `.lora_B.weight` names |
| `npu_qwen3_decoder_layer_impl.cpp` `build_node_variant_pack` | attach `in_seq_len_cum_sum` + 14 A/B tensors at variant-pack indices 68..82 |
| `xllm/models/llm/npu/qwen3.h` forward | generate `seq_len_cum_sum` (int64 [num_seq]) and route adapter selection |

Correctness plumbing:

- Extend `M2 LoRAAdapterLoader` to emit **three independent** Q/K/V A/B pairs
  (one per proj) instead of the current fused whole-block pair. This lands
  on top of P0-A `adapter_loader.cpp` -- keep the existing PEFT parse path,
  add a per-proj destructuring step.
- LoRA weights must go through `base_layer.h merge_loaded_weights` ->
  `AtTensor2Tensor` at layer init time. This is the atb-managed CANN
  stream path, distinct from torch_npu opapi. If loader currently only
  loops over 56 base slots, extend the loop to 70.
- Path B is the only route we've proven can move new tensors onto NPU on
  this stack.

Adapter selection at forward: single-request-per-batch first (no mixed
batching yet). Adapter routing is by `RequestParams::adapter_id` -> lookup
`LoRARegistry` -> pick the corresponding per-layer LoRA slots to activate
in the atb variant pack.

### Week 4 - single adapter e2e + baseline no-regression

Acceptance gates (in order):

1. Start Qwen3-8B with `--lora-modules biz-A=<real HF PEFT path>`
2. `curl model="biz-A"` -> valid coherent output (not just non-crash)
3. `rougeL diff` vs offline-merge reference < 0.05 (gate),
   goal < 0.01 -- if we land in [0.01, 0.05] this is acceptable for MVP,
   we do not chase the 0.01 target this phase.
4. Load a second adapter `biz-B`, alternate requests, each one hits its
   correct adapter (log line + qualitative check).
5. Baseline `model="Qwen3-8B"` request (LoRA path off) -- output must be
   byte-for-byte equal to v0.9.0-chunk. Zero regression on the base flow.

If (3) is > 0.05 we investigate numerical stability in atb GroupMatmul vs
offline merge before advancing; that's the divergence we care about, not
0.01 chasing.

### Week 5 - buffer for Path B unforeseen issues

Explicit buffer week. Common consumers:

- Week 2 escalation carryover
- HF adapter format quirks (target_modules mismatch, key naming variants)
- atb `set_weight` API surprise (older vs newer atb versions on 82 vs 98)

If Week 5 goes unused we advance to Phase 2 early -- do not add scope.

## Phase 2: multi-adapter batching + observability (3-4 weeks, P0-D)

### Week 6a - single-iteration single-adapter through GroupMatmul + early QPS

- Single-adapter iteration through the full atb GroupMatmul path
- Measure QPS relative to bare Qwen3-8B baseline. If regression is > 15%
  we investigate GroupMatmul overhead now, not at Week 8. Adjust
  expectation vs replan if needed.

### Week 6b - multi-adapter mixed-batch

- Batch composer splits requests by adapter, forms sub-batches
- `in_seq_len_cum_sum` computed across sub-batches: e.g. 3 requests, 3
  adapters -> `[tokens_a, tokens_a+tokens_b, tokens_a+tokens_b+tokens_c]`
- atb GroupMatmul dispatches per-segment against the corresponding A/B
- Scheduler M8 hard cap: `max_loras=16` mixed per batch, overflow spills
  to next iteration
- Deferring: iteration-vs-sub-iteration boundary policy (initially:
  greedy fill up to `max_loras`, spill excess).

### Week 7 - observability + unload drain + idempotency

Prometheus metrics:
- `lora_load_total{adapter=...}`
- `lora_active`
- `lora_inflight_requests{adapter=...}`
- `lora_load_latency_seconds`

Semantics:
- `unload_lora_adapter` blocks until adapter has zero in-flight requests,
  with configurable timeout and force-unload escape hatch (never hang the
  API forever)
- Repeat load of same name returns 200 (idempotent)
- `load_inplace` -> deferred to Phase 3 (v1.1)

### Week 8 - stress test + regression checklist (NOT full CI)

- 5-adapter mixed batch, QPS regression < 15% vs single-adapter merge
  baseline
- rougeL diff < 0.05 across all 5 (goal < 0.01)
- Manual regression checklist covering:
  * baseline chat unchanged
  * all 5 adapters produce distinct outputs
  * unload drain works under load
  * repeat-load idempotent
  * OOM behavior when max_loras exceeded

Full three-tier CI (kernel / layer / e2e) is **P1 territory**, not MVP.

## Phase 3: productionization (P1, 6 weeks)

- Two-level LoRA cache: pinned CPU pool + NPU device slot pool + LRU
  (avoid re-reading safetensors on every request)
- rank-bucket dispatch (LoRAX independent optimization: adapters bucketed
  by rank so no padding)
- Prefix-cache-aware v2 (SGLang extra_key)
- Multi-hardware backends: CUDA CUTLASS grouped-GEMM, MLU vendor lib
  (NPU-only until MVP passes)
- Full CI: kernel unit / layer integration / e2e regression

## Explicit non-goals for MVP

Held back to Phase 3 or later:

- Prefix cache under LoRA (v1 disables prefix cache for LoRA requests)
- disagg P/D
- fully-sharded S-LoRA
- Vision LoRA / VLM vision encoder / mm_projector
- LoRADrainer anti-starvation
- api_key -> adapter mapping (gateway team owns)
- MoE

## Corrections vs original plan

- Original plan's Week 1 Day 1-2 (rollback) is already done as of today.
- Original rougeL < 0.01 gate replaced with < 0.05 gate + < 0.01 goal.
  BF16 + atb GroupMatmul vs offline-merge fused deltas typically land in
  [0.02, 0.05] on other stacks; treating 0.01 as gate would burn time
  chasing numerical stability instead of shipping the loader.
- Real Qwen3-8B PEFT adapter: use HuggingFace open-source, do not train
  our own.
- Explicit Week 2 escalation gate: NoPack Qwen3 baseline chat must be
  byte-equal v0.9.0-chunk before Week 3.
- Explicit Week 5 buffer.
- Week 5 (original) split into 6a (early QPS measure) + 6b (mixed batch)
  so QPS surprises surface Week 6 rather than Week 8.
- CI shrunk to smoke tests + manual regression for MVP.

## Success = P0 MVP checklist

1. `--lora-modules a=/p1,b=/p2,c=/p3` at launch
2. `/v1/models` lists base + 3 adapters
3. `POST /v1/load_lora_adapter` for a 4th, next request routes to it,
   `POST /v1/unload_lora_adapter` drains
4. rougeL diff < 0.05 vs offline merge for all 4 adapters
5. QPS regression < 15% vs single-adapter merge baseline
6. Baseline `model="Qwen3-8B"` unchanged (byte-equal v0.9.0-chunk)
