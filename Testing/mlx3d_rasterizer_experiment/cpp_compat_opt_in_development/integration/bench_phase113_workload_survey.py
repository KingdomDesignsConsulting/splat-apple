#!/usr/bin/env python3
"""Phase113: resolution/tile-domain and memory-sensitivity survey on Apple Silicon.

Compares two independent real renderers. Exact RGB/alpha + workload parity is
mandatory, while per-case speed is measured rather than assumed. Each case uses
its own fresh instances and counterbalanced pair order. No global hooks or
production changes. Run on an idle GPU; avoid comparing cold frame totals.
"""
import argparse
import gc
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import mlx.core as mx
import numpy as np
from mlx_gs.renderer.mlx3d_cpp_compat import PersistentCppCompatRenderer, _renderer_class
from bench_phase49_breakdown import scene
from bench_phase57_active_sort import pose
from bench_phase85_final3 import fast_camera, assert_render_equal

CASES = {
    "subthreshold": (1280, 720),  # 3600 tiles: must not claim 512 eligibility
    "fullhd": (1920, 1080),       # 8160 tiles: conditional on sorted slots
    "highres": (2560, 1440),      # 14400 tiles: conditional on sorted slots
}
MODES = {"256": "selective_final3", "512": "selective_final3_scatter512"}


def run_case(label, n, rounds, poses):
    w, h = CASES[label]
    tile_count = ((w + 15) // 16) * ((h + 15) // 16)
    gs = scene(n)
    engines = {
        key: PersistentCppCompatRenderer(
            gs, duplicate_sort_mode="guarded", tile_range_mode="auto",
            tile_sort_impl=mode,
        ) for key, mode in MODES.items()
    }
    for key, expected in MODES.items():
        if engines[key]._renderer.tile_sort_impl != expected:
            raise AssertionError(("mode was not stored", label, key))

    records = []
    for round_id in range(rounds):
        for step, pose_id in enumerate(poses):
            cam = fast_camera(pose(w, h, pose_id))
            results = {}
            timings = {}
            counts = {}
            # Reverse execution order on alternate frame pairs to reduce order bias.
            order = ("256", "512") if (round_id + step) % 2 == 0 else ("512", "256")
            for key in order:
                start = time.perf_counter()
                result = engines[key]._renderer.render(cam)
                mx.eval(result["image"], result["alpha"])
                timings[key] = (time.perf_counter() - start) * 1000
                results[key] = result
                counts[key] = dict(engines[key]._renderer.last_frame_counts)
            assert_render_equal(
                {"radix4": results["256"], "final3": results["512"]},
                ("phase113", label, round_id, step, pose_id),
            )
            if counts["256"] != counts["512"]:
                raise AssertionError(("workload divergence", label, round_id, step, counts))
            slots = counts["256"].get("sorted")
            if not isinstance(slots, int):
                raise AssertionError(("missing integer sorted count", label, counts))
            eligible = tile_count >= 8000 and slots >= 8_000_000
            records.append({
                "round": round_id, "step": step, "pose": pose_id,
                "ms": timings, "counts": counts["256"], "eligible_by_threshold": eligible,
                "execution_order": list(order),
            })
            print("PASS", label, "round", round_id, "pose", pose_id,
                  "slots", slots, "eligible", eligible,
                  "256_ms", round(timings["256"], 3),
                  "512_ms", round(timings["512"], 3), flush=True)

    # Round zero is compilation/allocation warmup, not a controlled steady measurement.
    warm = [r for r in records if r["round"] > 0]
    median = {key: float(np.median([r["ms"][key] for r in warm])) for key in MODES}
    pair_deltas = [r["ms"]["512"] - r["ms"]["256"] for r in warm]
    peak_bytes = int(mx.get_peak_memory())
    summary = {
        "label": label, "width": w, "height": h, "gaussians": n,
        "tiles": tile_count, "rounds": rounds, "poses_per_round": len(poses),
        "warmed_pairs": len(warm), "median_ms": median,
        "median_pair_delta_ms": float(np.median(pair_deltas)),
        "median_relative_percent": 100.0 * (median["512"] / median["256"] - 1),
        "win_rate_512": float(np.mean([d < 0 for d in pair_deltas])),
        "eligible_warmed_pairs": sum(r["eligible_by_threshold"] for r in warm),
        "peak_mlx_memory_bytes_cumulative": peak_bytes,
        "parity": "PASS exact RGB and alpha, equal workload",
        "records": records,
    }
    del engines, gs
    gc.collect()
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--cases", nargs="+", choices=tuple(CASES),
                        default=list(CASES))
    parser.add_argument("--gaussians", type=int, default=700000)
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("output already exists: " + str(args.out))
    if args.rounds < 2 or args.gaussians < 1:
        parser.error("rounds must be >=2 and gaussians must be positive")
    if len(args.cases) != len(set(args.cases)):
        parser.error("duplicate case names")
    _renderer_class()  # enforce source guard before allocating test scenes

    # Fixed poses, same sequence for both paths; no cross-resolution time comparisons.
    pose_ids = [3, 14, 29, 40]
    results = [run_case(label, args.gaussians, args.rounds, pose_ids)
               for label in args.cases]
    report = {
        "phase": 113, "status": "PASS correctness (performance descriptive)",
        "source_guard": "PASS validated source loaded",
        "cases": results,
        "notes": [
            "No default promotion or performance threshold; per-case gains may reverse.",
            "Peak MLX memory is a cumulative process high-water mark, NOT per-case allocation.",
            "Eligibility is inferred from thresholds, not dispatch-instrumented (Phase110 covers fallback).",
            "Round 0 excluded from warmed timing; sequence/cold-start effects remain.",
            "No production renderer edits; RGB/alpha equality is required for every pair.",
        ],
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print("RESULT phase113 PASS", json.dumps({
        "case_summary": [{k: v for k, v in case.items() if k != "records"}
                         for case in results]
    }), flush=True)


if __name__ == "__main__":
    main()
