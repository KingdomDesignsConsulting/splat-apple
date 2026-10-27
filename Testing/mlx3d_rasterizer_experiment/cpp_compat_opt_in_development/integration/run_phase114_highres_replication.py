#!/usr/bin/env python3
"""Phase114: fresh-process high-resolution repeatability of the Phase113 result.

Each block uses a fresh process and writes independent Phase113 exact-parity
evidence. No renderer modifications. Performance regression is reported, not
hidden by a correctness PASS. Peak MLX memory is per process, not per engine.
"""
import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--gaussians", type=int, default=700000)
    args = parser.parse_args()
    if args.out_dir.exists():
        parser.error("output directory exists: " + str(args.out_dir))
    if args.blocks < 4 or args.blocks % 2 or args.rounds < 3 or args.gaussians < 1:
        parser.error("requires even blocks>=4, rounds>=3, gaussians>=1")
    args.out_dir.mkdir(parents=True)
    summary = args.out_dir / "phase114_summary.json"
    report = {
        "phase": 114, "status": "INCOMPLETE",
        "configuration": {"blocks": args.blocks, "rounds": args.rounds,
                          "gaussians": args.gaussians, "resolution": [2560, 1440]},
        "blocks": [],
        "notes": [
            "Independent subprocess blocks; all must pass exact RGB/alpha and workload parity.",
            "Median of per-block relative percentages measures direction, not formal significance.",
            "GPU clocks, thermal conditions and competing workloads are not controlled.",
            "MLX peak memory is a separate process high-water mark, not isolated sorter memory.",
            "No production/default changes; timing must not be compared with growth frames.",
        ],
    }
    try:
        for block in range(args.blocks):
            output = args.out_dir / f"phase113_highres_block{block}.json"
            logfile = args.out_dir / f"phase113_highres_block{block}.log"
            cmd = [
                sys.executable, str(HERE / "bench_phase113_workload_survey.py"),
                "--cases", "highres", "--gaussians", str(args.gaussians),
                "--rounds", str(args.rounds), "--order-offset", str(block % 2), "--out", str(output),
            ]
            print("RUN block", block, " ".join(cmd), flush=True)
            completed = subprocess.run(cmd, capture_output=True, text=True)
            logfile.write_text(completed.stdout + completed.stderr)
            if completed.returncode:
                raise RuntimeError(f"block {block} exited {completed.returncode}; see {logfile}")
            result = json.loads(output.read_text())
            if result.get("phase") != 113 or not str(result.get("status", "")).startswith("PASS"):
                raise AssertionError(("Phase113 did not pass", block))
            cases = result.get("cases", [])
            if len(cases) != 1 or cases[0].get("label") != "highres":
                raise AssertionError(("unexpected highres output", block))
            case = cases[0]
            expected_pairs = (args.rounds - 1) * 4
            if (case["parity"] != "PASS exact RGB and alpha, equal workload"
                    or case["warmed_pairs"] != expected_pairs
                    or case["eligible_warmed_pairs"] != expected_pairs):
                raise AssertionError(("incorrect parity, capacity, or eligibility", block, case))
            row = {
                "block": block, "order_offset": block % 2, "relative_percent_512": case["median_relative_percent"],
                "median_pair_delta_ms": case["median_pair_delta_ms"],
                "win_rate_512": case["win_rate_512"],
                "median_ms": case["median_ms"],
                "peak_mlx_memory_bytes": case["peak_mlx_memory_bytes_cumulative"],
                "warmed_pairs": expected_pairs,
                "result_file": str(output), "log_file": str(logfile),
            }
            report["blocks"].append(row)
            print("PASS block", block, "relative_percent_512",
                  round(row["relative_percent_512"], 3), "wins",
                  round(row["win_rate_512"], 3), flush=True)

        changes = [x["relative_percent_512"] for x in report["blocks"]]
        report["median_block_relative_percent_512"] = statistics.median(changes)
        report["order_group_median_relative_percent_512"] = {\n            str(offset): statistics.median(x["relative_percent_512"] for x in report["blocks"] if x["order_offset"] == offset)\n            for offset in (0, 1)\n        }\n        report["blocks_faster_512"] = sum(v < 0 for v in changes)
        report["blocks_slower_512"] = sum(v > 0 for v in changes)
        report["total_warmed_pairs"] = sum(x["warmed_pairs"] for x in report["blocks"])
        report["status"] = "PASS correctness; performance observational"
    except Exception as exc:
        report["status"] = "FAIL"
        report["error"] = str(exc)
        summary.write_text(json.dumps(report, indent=2) + "\n")
        raise
    summary.write_text(json.dumps(report, indent=2) + "\n")
    print("RESULT phase114", report["status"], json.dumps({
        "median_block_relative_percent_512": report["median_block_relative_percent_512"],
        "blocks_faster_512": report["blocks_faster_512"],
        "blocks_slower_512": report["blocks_slower_512"],
        "total_warmed_pairs": report["total_warmed_pairs"],\n        "order_group_median_relative_percent_512": report["order_group_median_relative_percent_512"],
    }), flush=True)


if __name__ == "__main__":
    main()
