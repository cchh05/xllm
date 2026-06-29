# Probes

Developer-driven validation binaries that need real hardware (NPU + mpirun)
to run. **Probes do not run under ctest** — they are gated behind a custom
build target so they cannot accidentally break CI on build hosts without NPUs.

## Build

From the repo root:

```bash
# Build everything in this folder
cmake --build build --target all_probes

# Or build a single probe
cmake --build build --target comm_switch_probe
```

The probe binary lands under `build/tests/probes/<probe_name>`.

---

## comm_switch_probe — Runtime CP↔DP communicator rebuild

**Question this probe answers:** can we destroy and rebuild the HCCL +
ATB collective communicators at runtime, switching between
`cp_size=N, dp_size=1` and `cp_size=1, dp_size=N` over the same world,
without leaks or correctness regressions?

This is the gating experiment for the larger "P-pool runtime CP↔DP" feature
(plan: `~/.claude/plans/b-cozy-valley.md`). If the probe passes, we walk
the full 4-week implementation plan; if it fails, the failure mode tells
us which fallback path to take (always-on dual groups, ATB API extension,
LoongServe-style ESP, etc.).

### Run on 910_82 (4 cards, single host)

This probe targets the **TORCH** NPU kernel backend, where xllm owns the
HCCL `ProcessGroup` objects directly. Under the **ATB** backend
`CollectiveCommunicator::create_process_groups` early-returns and the comms
live inside `atb_speed::ExternalCommManager` — that path needs a separate
probe and is intentionally out of scope here.

The xllm-ai container does not bundle OpenMPI, so the launcher uses
**torchrun** (ships with torch 2.7+) to spawn one process per NPU. torchrun
exports `RANK` / `LOCAL_RANK` / `WORLD_SIZE` / `MASTER_ADDR` / `MASTER_PORT`
the way the probe expects.

```bash
# Recommended: use the launcher script (sources NPU/ATB env, sets
# ASCEND_RT_VISIBLE_DEVICES, picks a sane port, etc.).
bash tests/probes/run_comm_switch_probe.sh --iters=50

# Or invoke torchrun directly if you need finer control.
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
export NPU_HOME_PATH="${ASCEND_HOME_PATH}"
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
torchrun --standalone --nproc_per_node=4 --no_python \
    ./build/lib.linux-aarch64-cpython-311/xllm/comm_switch_probe \
    --npu_kernel_backend=TORCH \
    --master_addr=127.0.0.1:29500 \
    --world_size=4 \
    --iters=50
```

`rank_tablefile` is **not required** for single-node runs: when empty,
`MappingNPU::get_num_nodes()` falls back to `1`, which is what we want for
a 4-card single-host probe. Multi-host probes will need a real rank table.
`EPLB_CONFIG`; see `xllm/core/framework/config/eplb_config.cpp` for the
exact lookup chain. For a quick smoke test on 2 cards, override the env
with `WORLD_SIZE=2 DEVICES=0,1 bash tests/probes/run_comm_switch_probe.sh`.

### Output to look for

Per-iteration log:

```
[probe] iter=0 cp_setup=...ms cp_forward=...ms cp_destroy=...ms \
        dp_setup=...ms dp_forward=...ms dp_destroy=...ms \
        free_mb=... mem_drop_mb=...
```

Final summary:

```
[probe] summary iters=50 failures=0 avg_setup_ms=... avg_destroy_ms=... \
        worst_setup_ms=... worst_destroy_ms=... \
        final_free_mb=... total_drop_mb=...
[probe] PASS: 50 CP<->DP switches succeeded
```

### Interpreting the result

| Outcome | What it means | Next action |
|---------|---------------|-------------|
| **A1 PASS** | All iters succeed; setup < 2 s, destroy < 1 s, mem drop < 100 MB | Walk the full 4-week plan in `b-cozy-valley.md` |
| **A2 SLOW** | Pass but `worst_setup_ms` / `total_drop_mb` exceeds budget | Switch to "always-on dual group" fallback (CP and DP comms both built at startup; runtime only flips traffic) |
| **A3 HCCL FAIL** | `failures > 0` with `HcclCommInit*` errors after `HcclCommDestroy` | Stop the rebuild path; pivot to LoongServe-style ESP (ranks bind both groups simultaneously) |
| **A4 ATB FAIL** | `failures > 0` because ATB's `ExternalCommManager` rejects the second `InitGlobalCommDomain` | Open a request to ATB for `RemoveCommDomain`; v1 falls back to "single switch only" |

### Flags

| Flag | Default | Notes |
|------|---------|-------|
| `--iters` | 50 | Number of CP→DP→CP cycles |
| `--world_size` | 4 | Must match `mpirun -n N` |
| `--master_addr` | `127.0.0.1:29500` | TCPStore rendezvous; reads `MASTER_ADDR` / `MASTER_PORT` env first |
| `--forward_iters` | 3 | Smoke `allreduce` calls per phase per iter |
| `--forward_numel` | 1024 | Tensor size for the smoke `allreduce` |
| `--mem_growth_threshold_mb` | 100 | Probe fails if free-memory drop exceeds this |

### Known limitations of the probe

- Only validates **attention CP↔DP** (`ep_size=1`). MoE EP rebuild is out of
  scope for v1; once the attention path lands, retest EP with a second probe.
- The smoke `allreduce` uses a fixed-value tensor and a 1e-3 tolerance. It
  catches "comm is corrupt" but does not catch subtle numerics drift —
  that is for the integration suite, not the probe.
- Does not exercise the KV cache reshape path. v1 design uses **drop KV** on
  switch (drain → discard), so this is intentional; KV migration is v2.
- Single-host only. Cross-host comm rebuild has different failure modes
  (TCPStore lifetime, rank table re-read) that warrant a separate probe.
