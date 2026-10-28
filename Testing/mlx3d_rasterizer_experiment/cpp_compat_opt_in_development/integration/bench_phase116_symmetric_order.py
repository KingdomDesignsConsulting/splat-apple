#!/usr/bin/env python3
"""Phase116: symmetric four-render order-balanced highres comparison.

ABBA and BAAB quadruplets cancel a linear within-quadruplet timing drift
in each mode's two-observation mean. This does not control non-linear drift,
GPU contention or clock changes. Correctness is required on every rendering.
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
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--rounds", type=int, default=4)
    p.add_argument("--gaussians", type=int, default=700000)
    args = p.parse_args()
    if args.out.exists():
        p.error("output exists: " + str(args.out))
    if args.rounds < 2 or args.gaussians < 1:
        p.error("requires rounds >=2 and gaussians >=1")
    _renderer_class()
    gs = scene(args.gaussians)
    engines = {
        "256": PersistentCppCompatRenderer(gs, duplicate_sort_mode="guarded",
            tile_range_mode="auto", tile_sort_impl="selective_final3"),
        "512": PersistentCppCompatRenderer(gs, duplicate_sort_mode="guarded",
            tile_range_mode="auto", tile_sort_impl="selective_final3_scatter512"),
    }
    w, h = 2560, 1440
    maxcam = fast_camera(pose(w, h, 40))
    for mode in ("256", "512"):
        for _ in range(2):
            frame = engines[mode]._renderer.render(maxcam)
            mx.eval(frame["image"], frame["alpha"])
    rows = []
    for round_id in range(args.rounds):
        for step, pos in enumerate((3, 14, 29, 40)):
            cam = fast_camera(pose(w, h, pos))
            order = ("256", "512", "512", "256") if (round_id + step) % 2 == 0 else ("512", "256", "256", "512")
            samples = {"256": [], "512": []}
            frames = {}
            counts = {}
            for mode in order:
                start = time.perf_counter()
                frame = engines[mode]._renderer.render(cam)
                mx.eval(frame["image"], frame["alpha"])
                samples[mode].append((time.perf_counter() - start) * 1000.0)
                if frames:
                    assert_render_equal({"radix4": frames["256"], "final3": frame} if mode == "512" and "256" in frames else
                                        {"radix4": frame, "final3": frames["512"]} if mode == "256" and "512" in frames else
                                        {"radix4": frame, "final3": frame}, ("phase116", round_id, step, mode))
                frames[mode] = frame
                counts[mode] = dict(engines[mode]._renderer.last_frame_counts)
                if len(counts) == 2 and counts["256"] != counts["512"]:
                    raise AssertionError(("workload mismatch", round_id, pos, counts))
            slots = counts["256"].get("sorted")
            if not isinstance(slots, int) or slots < 8000000:
                raise AssertionError(("ineligible sort capacity", slots))
            means = {mode: statistics.mean(samples[mode]) for mode in samples}
            row = {"round": round_id, "pose": pos, "order": list(order),
                   "samples_ms": samples, "mean_ms": means,
                   "delta_512_minus_256_ms": means["512"] - means["256"], "slots": slots}
            rows.append(row)
            print("PASS", round_id, pos, "".join("A" if m == "256" else "B" for m in order),
                  "delta_ms", round(row["delta_512_minus_256_ms"], 3), flush=True)
    warm = [row for row in rows if row["round"] > 0]
    groups = {}
    for label in ("ABBA", "BAAB"):
        vals = [row["delta_512_minus_256_ms"] for row in warm
                if ("".join("A" if m == "256" else "B" for m in row["order"]) == label)]
        if len(vals) != len(warm) // 2:
            raise AssertionError(("imbalanced order", label, len(vals)))
        groups[label] = {"quadruplets": len(vals), "median_delta_ms": statistics.median(vals),
                         "wins_512": sum(v < 0 for v in vals)}
    deltas = [row["delta_512_minus_256_ms"] for row in warm]
    report = {"phase": 116, "status": "PASS correctness; timing observational",
              "rounds": args.rounds, "warmed_quadruplets": len(warm),
              "warmed_render_calls": len(warm) * 4,
              "median_delta_ms": statistics.median(deltas),
              "groups": groups, "peak_mlx_memory_bytes": int(mx.get_peak_memory()),
              "rows": rows, "notes": [
                  "Within-quadruplet symmetric ordering cancels linear time trend, not nonlinear drift.",
                  "Round zero excluded from timing summary; all rounds tested exact parity.",
                  "No production renderer, default or source digest changes."]}
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print("RESULT phase116 PASS", json.dumps({k: v for k, v in report.items() if k != "rows"}), flush=True)


if __name__ == "__main__":
    main()
