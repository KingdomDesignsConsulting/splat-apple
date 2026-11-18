#!/usr/bin/env python3
"""Phase118: replicate randomized-order highres result in independent processes.

Each subprocess runs the existing Phase116 randomized-order script with a
distinct deterministic seed. Preserve original reports and logs. Correctness
required; performance effect and interval are observational.
"""
import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--blocks", type=int, default=4)
    p.add_argument("--rounds", type=int, default=8)
    p.add_argument("--seed", type=int, default=1182026)
    args = p.parse_args()
    if args.out_dir.exists():
        p.error("output directory already exists")
    if args.blocks < 4 or args.rounds < 4:
        p.error("requires blocks >=4, rounds >=4")
    args.out_dir.mkdir(parents=True)
    rows = []
    summary_path = args.out_dir / "phase118_summary.json"
    try:
        for i in range(args.blocks):
            result_file = args.out_dir / f"block_{i:02d}.json"
            log_file = args.out_dir / f"block_{i:02d}.log"
            seed = args.seed + i * 9973
            cmd = [sys.executable, str(HERE / "bench_phase116_randomized_order.py"),
                   "--rounds", str(args.rounds), "--seed", str(seed),
                   "--out", str(result_file)]
            with log_file.open("w") as log:
                completed = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
            if completed.returncode:
                raise RuntimeError(f"block {i} exited {completed.returncode}; see {log_file}")
            data = json.loads(result_file.read_text())
            if data.get("phase") != 116 or not str(data.get("status", "")).startswith("PASS correctness"):
                raise AssertionError(("child correctness failed", i))
            expected = (args.rounds - 1) * 4
            if data.get("balanced_pose_round_pairs") != expected or data.get("warm_pairs") != 2 * expected:
                raise AssertionError(("unexpected pair counts", i))
            row = {
                "block": i, "seed": seed,
                "order_averaged_mean_delta_ms": data["order_averaged_mean_delta_ms"],
                "order_averaged_median_delta_ms": data["order_averaged_median_delta_ms"],
                "first_order_bias_mean_ms": data["first_order_bias_mean_ms"],
                "round_cluster_bootstrap95_mean_delta_ms": data["round_cluster_bootstrap95_mean_delta_ms"],
                "peak_mlx_memory_bytes": data["peak_mlx_memory_bytes"],
                "json": result_file.name, "log": log_file.name,
            }
            rows.append(row)
            print("PASS block", i, "mean_delta_ms",
                  round(row["order_averaged_mean_delta_ms"], 3), flush=True)
        means = [row["order_averaged_mean_delta_ms"] for row in rows]
        summary = {
            "phase": 118, "status": "PASS correctness; performance observational",
            "blocks": rows, "blocks_favor_512": sum(x < 0 for x in means),
            "median_process_mean_delta_ms": statistics.median(means),
            "mean_of_process_means_delta_ms": statistics.mean(means),
            "min_process_mean_delta_ms": min(means),
            "max_process_mean_delta_ms": max(means),
            "notes": [
                "Independent child processes with different randomized seeds.",
                "Neither process confidence interval nor mean-of-means proves GPU-kernel improvement.",
                "All child evidence files and logs retained; no source changes.",
            ],
        }
    except Exception as exc:
        summary = {"phase": 118, "status": "INCOMPLETE", "blocks": rows, "error": str(exc)}
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        raise
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print("RESULT phase118 PASS", json.dumps({k: v for k, v in summary.items() if k != "blocks"}), flush=True)


if __name__ == "__main__":
    main()
