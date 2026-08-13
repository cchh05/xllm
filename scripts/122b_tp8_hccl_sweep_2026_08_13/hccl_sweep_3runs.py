#!/usr/bin/env python3
"""HCCL sweep helper: given a warmed service, run 3x A + 3x C, report min/median/max."""

import argparse, sys, time, statistics
sys.path.insert(0, "/export/home/caihao.40/tmp_probe")
import bench_affinity_v2 as bench

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=28700)
ap.add_argument("--tag", required=True)
ap.add_argument("--repeat", type=int, default=40)
ap.add_argument("--concurrency", type=int, default=16)
ap.add_argument("--out_dir", default="/export/home/caihao.40/phase5_122b_throughput_sweep")
args = ap.parse_args()

import os, json
os.makedirs(args.out_dir, exist_ok=True)
out_path = os.path.join(args.out_dir, "bench_{}_3runs.json".format(args.tag))

pool = bench.build_pool(args.repeat)

print("=== [{}] 3x A + 3x C bench ===".format(args.tag), flush=True)

results = {"a_runs": [], "c_runs": []}

for i in range(3):
    print("\n--- A run {}/3 ---".format(i + 1), flush=True)
    r = bench.run_scenario("A_{}_run{}".format(args.tag, i + 1), args.port,
                            [("medqa", 1)], pool, args.concurrency, verbose=False)
    results["a_runs"].append({
        "wall_s": r["wall_s"],
        "e2e_p50": r["e2e_p50"],
        "e2e_p95": r["e2e_p95"],
        "ttft_p50": r["ttft_p50"],
        "ttft_p95": r["ttft_p95"],
        "throughput": r["throughput_tok_per_s"],
        "fast": r["fast_path_batches"],
        "slow": r["slow_path_batches"],
        "gen_tokens": r["gen_tokens_total"],
    })
    print("  wall={:.2f}s thr={:.2f} p50={:.3f} p95={:.3f} fast={} slow={}".format(
        r["wall_s"], r["throughput_tok_per_s"], r["e2e_p50"], r["e2e_p95"],
        r["fast_path_batches"], r["slow_path_batches"]), flush=True)
    time.sleep(3)

for i in range(3):
    print("\n--- C run {}/3 ---".format(i + 1), flush=True)
    r = bench.run_scenario("C_{}_run{}".format(args.tag, i + 1), args.port,
                            [("medqa", 1), ("medqa2", 1)], pool, args.concurrency, verbose=False)
    results["c_runs"].append({
        "wall_s": r["wall_s"],
        "e2e_p50": r["e2e_p50"],
        "e2e_p95": r["e2e_p95"],
        "ttft_p50": r["ttft_p50"],
        "ttft_p95": r["ttft_p95"],
        "throughput": r["throughput_tok_per_s"],
        "fast": r["fast_path_batches"],
        "slow": r["slow_path_batches"],
        "gen_tokens": r["gen_tokens_total"],
    })
    print("  wall={:.2f}s thr={:.2f} p50={:.3f} p95={:.3f} fast={} slow={}".format(
        r["wall_s"], r["throughput_tok_per_s"], r["e2e_p50"], r["e2e_p95"],
        r["fast_path_batches"], r["slow_path_batches"]), flush=True)
    time.sleep(3)

with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print("\n=== Saved {} ===".format(out_path), flush=True)

print("\n=== SUMMARY [{}] ===".format(args.tag), flush=True)
for scenario, runs in [("A", results["a_runs"]), ("C", results["c_runs"])]:
    walls = [r["wall_s"] for r in runs]
    thrs = [r["throughput"] for r in runs]
    p95s = [r["e2e_p95"] for r in runs]
    print("{}: wall min/med/max = {:.2f}/{:.2f}/{:.2f}, thr min/med/max = {:.2f}/{:.2f}/{:.2f}, p95 min/med/max = {:.2f}/{:.2f}/{:.2f}".format(
        scenario,
        min(walls), statistics.median(walls), max(walls),
        min(thrs), statistics.median(thrs), max(thrs),
        min(p95s), statistics.median(p95s), max(p95s),
    ), flush=True)
