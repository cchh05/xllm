#!/usr/bin/env python3
"""Phase 0/1/2/3 bench wrapper — v2 with proper concurrent warmup.

Fix vs v1:
  - Old warmup was single-request per adapter -> only captured num_tokens=1
    bucket. Bench-time recapture of buckets 2/4/8 tanked scenario A wall
    (bucket=1 recapture in middle of A took ~76s).
  - v2 warmup fires N concurrent requests per adapter, holding decode
    batch at the max, so all buckets (1/2/4/8) capture BEFORE bench.

Usage:
    python3 phase_bench.py --port 28700 --tag tp16 --repeat 40 --warmup 8
"""

import argparse
import concurrent.futures as cf
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, "/export/home/caihao.40/tmp_probe")
import bench_affinity_v2 as bench


def _fire_concurrent(port, name_prompt_pairs, n_concurrent, timeout=180):
    """Fire len(pairs) requests concurrently and wait for all. Returns
    (n_ok, n_err).

    Uses sample_max_tokens() (bench distribution) so requests complete at
    DIFFERENT times, causing batch size to drain incrementally
    (concurrency -> concurrency-1 -> ... -> 1). This triggers ALL buckets
    to lazy-capture during warmup, not just the top bucket. Fixed
    max_tokens would only cover the top bucket + a sudden drop.
    """
    with cf.ThreadPoolExecutor(max_workers=n_concurrent) as ex:
        futs = [ex.submit(bench.chat_stream, port, n, p,
                          bench.sample_max_tokens(), timeout=timeout)
                for n, p in name_prompt_pairs]
        ok, err = 0, 0
        for f in cf.as_completed(futs):
            try:
                f.result()
                ok += 1
            except Exception:
                err += 1
    return ok, err


def concurrent_warmup(port, medqa, medqa2, warmup_concurrency):
    """5-round warmup with `warmup_concurrency` concurrent requests using
    sample_max_tokens() distribution.

    Warmup concurrency MUST >= bench concurrency, otherwise bench-time
    lazy capture of larger bucket (bucket num_tokens >= 8) pollutes wall.
    See acl_graph_executor_impl.cpp:1255 for bucket sizing rules.

    5 rounds ensures ALL buckets capture stably (based on 08-13 finding
    that same bucket num_tokens can capture 4x due to batch composition
    variance). Rounds alternate single-adapter (fast_path warm) and mixed
    (slow_path warm).
    """
    prompt = "Explain the pathophysiology of diabetes in detail."
    half = max(warmup_concurrency // 2, 1)

    rounds = [
        ("R1 single medqa", [(medqa, prompt)] * warmup_concurrency),
        ("R2 single medqa2", [(medqa2, prompt)] * warmup_concurrency),
        ("R3 mixed 50/50", [(medqa, prompt)] * half + [(medqa2, prompt)] * half),
        ("R4 single medqa", [(medqa, prompt)] * warmup_concurrency),
        ("R5 mixed 50/50", [(medqa, prompt)] * half + [(medqa2, prompt)] * half),
    ]
    for label, pairs in rounds:
        print("=== warmup {}: {}x concurrent ===".format(label, warmup_concurrency),
              flush=True)
        ok, err = _fire_concurrent(port, pairs, warmup_concurrency)
        print("  {}: ok={} err={}".format(label, ok, err), flush=True)
        time.sleep(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=28700)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--repeat", type=int, default=40)
    ap.add_argument("--warmup_concurrency", type=int, default=0,
                    help="warmup concurrency; 0 (default) = same as --concurrency. "
                         "MUST >= bench concurrency or lazy-capture pollutes wall")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--medqa", default="medqa")
    ap.add_argument("--medqa2", default="medqa2")
    ap.add_argument("--out_dir",
                    default="/export/home/caihao.40/phase5_122b_throughput_sweep")
    ap.add_argument("--scenarios", default="A,C")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "bench_{}.json".format(args.tag))

    pool = bench.build_pool(args.repeat)

    warmup_conc = args.warmup_concurrency if args.warmup_concurrency > 0 \
        else args.concurrency
    assert warmup_conc >= args.concurrency, \
        "warmup_concurrency ({}) must be >= bench concurrency ({})".format(
            warmup_conc, args.concurrency)
    concurrent_warmup(args.port, args.medqa, args.medqa2, warmup_conc)

    results = []
    if "A" in args.scenarios:
        print("\n=== [{}] Scenario A single_medqa ===".format(args.tag),
              flush=True)
        r_a = bench.run_scenario("A_single_{}".format(args.medqa), args.port,
                                  [(args.medqa, 1)], pool, args.concurrency,
                                  verbose=True)
        results.append(r_a)
        time.sleep(5)

    if "C" in args.scenarios:
        print("\n=== [{}] Scenario C mixed_50_50 ===".format(args.tag),
              flush=True)
        r_c = bench.run_scenario("C_mixed_50_50", args.port,
                                  [(args.medqa, 1), (args.medqa2, 1)], pool,
                                  args.concurrency, verbose=True)
        results.append(r_c)

    payload = {"config": vars(args), "results": results}
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print("\n=== Saved {} ===".format(out_path), flush=True)

    print("\n=== SUMMARY ===", flush=True)
    for r in results:
        print("{:<28} n={:>3} wall={:>7} e2e_p50={:>7} e2e_p95={:>7} "
              "ttft_p50={:>7} ttft_p95={:>7} thr={:>7} fast={:>3} slow={:>3}".format(
              r["scenario"], r["n_ok"], r["wall_s"],
              r["e2e_p50"], r["e2e_p95"],
              r["ttft_p50"], r["ttft_p95"],
              r["throughput_tok_per_s"],
              r["fast_path_batches"], r["slow_path_batches"]),
              flush=True)


if __name__ == "__main__":
    main()
