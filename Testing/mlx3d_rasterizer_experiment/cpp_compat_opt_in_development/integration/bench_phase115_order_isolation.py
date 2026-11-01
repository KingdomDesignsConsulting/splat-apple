#!/usr/bin/env python3
"""Phase115: within-pose ABBA/BAAB order-effect isolation at 2560x1440.

For each pose, run both renderers twice in reversed orders. Paired comparisons
are classified by which sorter ran first; stable scene capacity is warmed first.
No production modifications or claims of GPU timestamp accuracy.
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
import mlx.core as mx
from mlx_gs.renderer.mlx3d_cpp_compat import PersistentCppCompatRenderer, _renderer_class
from bench_phase49_breakdown import scene
from bench_phase57_active_sort import pose
from bench_phase85_final3 import fast_camera, assert_render_equal

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--rounds", type=int, default=4)
    p.add_argument("--gaussians", type=int, default=700000)
    args = p.parse_args()
    if args.out.exists():
        p.error("output already exists: " + str(args.out))
    if args.rounds < 2 or args.gaussians < 1:
        p.error("requires rounds >= 2, gaussians >= 1")
    _renderer_class()
    gs = scene(args.gaussians)
    engines = {
        "256": PersistentCppCompatRenderer(gs, duplicate_sort_mode="guarded", tile_range_mode="auto", tile_sort_impl="selective_final3"),
        "512": PersistentCppCompatRenderer(gs, duplicate_sort_mode="guarded", tile_range_mode="auto", tile_sort_impl="selective_final3_scatter512"),
    }
    width, height = 2560, 1440
    positions = [3, 14, 29, 40]
    # Establish the maximum capacity before measuring; avoid growth bias.
    camera = fast_camera(pose(width, height, 40))
    for name in ("256", "512"):
        for _ in range(2):
            frame = engines[name]._renderer.render(camera)
            mx.eval(frame["image"], frame["alpha"])
    records = []
    for round_id in range(args.rounds):
        for step, pos in enumerate(positions):
            camera = fast_camera(pose(width, height, pos))
            # Each order appears once for the SAME pose within each round.
            # Reverse order-block sequence on alternate rounds/poses.
            order_blocks = [("256", "512"), ("512", "256")]
            if (round_id + step) % 2:
                order_blocks.reverse()
            for pair_index, order in enumerate(order_blocks):
                frames, ms, counts = {}, {}, {}
                for mode in order:
                    begin = time.perf_counter()
                    frame = engines[mode]._renderer.render(camera)
                    mx.eval(frame["image"], frame["alpha"])
                    ms[mode] = (time.perf_counter() - begin) * 1000.0
                    frames[mode] = frame
                    counts[mode] = dict(engines[mode]._renderer.last_frame_counts)
                assert_render_equal({"radix4": frames["256"], "final3": frames["512"]}, ("phase115", round_id, step, pair_index))
                if counts["256"] != counts["512"]:
                    raise AssertionError(("workload mismatch", round_id, step, counts))
                slots = counts["256"].get("sorted")
                if not isinstance(slots, int) or slots < 8000000:
                    raise AssertionError(("unexpected capacity", slots))
                row = {"round": round_id, "pose": pos, "pair": pair_index, "first": order[0],
                       "ms": ms, "delta_512_minus_256_ms": ms["512"] - ms["256"], "slots": slots}
                records.append(row)
                print("PASS", round_id, pos, "first", order[0], "delta_ms", round(row["delta_512_minus_256_ms"], 3), flush=True)
    # Exclude round zero from analysis, while retaining exact parity evidence.
    warm = [row for row in records if row["round"] > 0]
    by_first = {key: [r["delta_512_minus_256_ms"] for r in warm if r["first"] == key] for key in ("256", "512")}
    if not all(len(v) == len(warm) // 2 for v in by_first.values()):
        raise AssertionError("imbalanced order groups")
    summary = {key: {"pairs": len(v), "median_delta_ms": statistics.median(v),
                     "wins_512": sum(x < 0 for x in v)} for key, v in by_first.items()}
    report = {"phase": 115, "status": "PASS correctness; performance observational",
              "rounds": args.rounds, "gaussians": args.gaussians, "resolution": [width, height],
              "warmed_pairs": len(warm), "by_first": summary,
              "order_effect_median_delta_ms": summary["256"]["median_delta_ms"] - summary["512"]["median_delta_ms"],
              "peak_mlx_memory_bytes": int(mx.get_peak_memory()), "records": records,
              "notes": ["Within-pose two-order comparisons, no randomization; temporal drift can remain.",
                        "Round zero excluded from timings. Capacity eligibility inferred from thresholds.",
                        "No renderer or default changes; do not promote mode based on this result."]}
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print("RESULT phase115 PASS", json.dumps({k: v for k, v in report.items() if k != "records"}), flush=True)

if __name__ == "__main__":
    main()
