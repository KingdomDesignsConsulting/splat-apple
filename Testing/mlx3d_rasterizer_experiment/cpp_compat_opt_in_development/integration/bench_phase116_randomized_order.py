#!/usr/bin/env python3
"""Phase116: deterministic randomized, balanced within-pose order experiment.

Test persistent 256 vs 512 scatter sorting at equal workload with paired,
counterbalanced measurement. Wall-clock render+mx.eval timings, NOT GPU timestamps.
No renderer/default changes. Each pose/round has both orders and shuffled block
position; first round excluded from primary stats. Reports first-order bias,
pair effect, and repeated pose-round contrasts.
"""
import argparse
import json
import random
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

def median(xs):
    return statistics.median(xs) if xs else None

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--rounds", type=int, default=8)
    p.add_argument("--gaussians", type=int, default=700000)
    p.add_argument("--seed", type=int, default=1162026)
    args = p.parse_args()
    if args.out.exists():
        p.error("output exists; refusing overwrite: " + str(args.out))
    if args.rounds < 4 or args.gaussians < 1:
        p.error("requires --rounds >= 4 and --gaussians >= 1")
    _renderer_class()
    gs = scene(args.gaussians)
    engines = {
        "256": PersistentCppCompatRenderer(gs, duplicate_sort_mode="guarded",
            tile_range_mode="auto", tile_sort_impl="selective_final3"),
        "512": PersistentCppCompatRenderer(gs, duplicate_sort_mode="guarded",
            tile_range_mode="auto", tile_sort_impl="selective_final3_scatter512"),
    }
    width, height = 2560, 1440
    poses = (3, 14, 29, 40)
    camera = fast_camera(pose(width, height, 40))
    # Common capacity and pipeline warmup.
    for mode in ("256", "512"):
        for _ in range(3):
            frame = engines[mode]._renderer.render(camera)
            mx.eval(frame["image"], frame["alpha"])
    rng = random.Random(args.seed)
    records = []
    for round_id in range(args.rounds):
        positions = list(poses)
        rng.shuffle(positions)
        for pos in positions:
            camera = fast_camera(pose(width, height, pos))
            blocks = [("256", "512"), ("512", "256")]
            rng.shuffle(blocks)
            for block_id, order in enumerate(blocks):
                frames, ms, counts = {}, {}, {}
                for mode in order:
                    t0 = time.perf_counter()
                    frame = engines[mode]._renderer.render(camera)
                    mx.eval(frame["image"], frame["alpha"])
                    ms[mode] = (time.perf_counter() - t0) * 1000.0
                    frames[mode] = frame
                    counts[mode] = dict(engines[mode]._renderer.last_frame_counts)
                assert_render_equal({"radix4": frames["256"], "final3": frames["512"]},
                    ("phase116", round_id, pos, block_id))
                if counts["256"] != counts["512"]:
                    raise AssertionError(("workload mismatch", round_id, pos, counts))
                slots = counts["256"].get("sorted")
                if not isinstance(slots, int) or slots < 8000000:
                    raise AssertionError(("capacity unexpectedly ineligible", slots))
                record = {"round": round_id, "pose": pos, "block": block_id,
                    "first": order[0], "order": list(order), "ms": ms,
                    "delta_512_minus_256_ms": ms["512"]-ms["256"], "slots": slots}
                records.append(record)
                print("PASS", round_id, pos, "first", order[0],
                      "delta_ms", round(record["delta_512_minus_256_ms"], 3), flush=True)
    warm = [r for r in records if r["round"] != 0]
    by_first = {x: [r["delta_512_minus_256_ms"] for r in warm if r["first"] == x]
                for x in ("256", "512")}
    paired = []
    for rnd in range(1, args.rounds):
        for pos in poses:
            subset = [r for r in warm if r["round"] == rnd and r["pose"] == pos]
            if len(subset) != 2 or {r["first"] for r in subset} != {"256", "512"}:
                raise AssertionError(("order imbalance", rnd, pos))
            d256 = next(r["delta_512_minus_256_ms"] for r in subset if r["first"] == "256")
            d512 = next(r["delta_512_minus_256_ms"] for r in subset if r["first"] == "512")
            paired.append({"round": rnd, "pose": pos,
                           "order_averaged_delta_ms": (d256+d512)/2,
                           "first_order_bias_ms": d256-d512,
                           "d256": d256, "d512": d512})
    order_avg = [x["order_averaged_delta_ms"] for x in paired]
    order_bias = [x["first_order_bias_ms"] for x in paired]
    # Round-cluster bootstrap: keep the four poses and both execution orders
    # together when resampling, accounting for temporal dependence within a round.
    groups = {r: [x for x in paired if x["round"] == r] for r in range(1,args.rounds)}
    ids = sorted(groups)
    bootstrap_rng = random.Random(args.seed+1)
    boots = []
    for _ in range(5000):
        draws = [bootstrap_rng.choice(ids) for _ in ids]
        values = [x["order_averaged_delta_ms"] for rnd in draws for x in groups[rnd]]
        boots.append(statistics.mean(values))
    boots.sort()
    lo, hi = boots[int(.025*len(boots))], boots[int(.975*len(boots))]
    report = {"phase": 116, "status": "PASS correctness; wall-clock performance exploratory",
        "rounds": args.rounds, "seed": args.seed,
        "gaussians": args.gaussians, "resolution": [width,height],
        "warm_pairs": len(warm),
        "by_first": {k: {"n": len(v), "median_delta_ms": median(v),
                         "mean_delta_ms": statistics.mean(v),
                         "wins_512": sum(d < 0 for d in v)} for k,v in by_first.items()},
        "balanced_pose_round_pairs": len(paired),
        "order_averaged_mean_delta_ms": statistics.mean(order_avg),
        "order_averaged_median_delta_ms": median(order_avg),
        "first_order_bias_mean_ms": statistics.mean(order_bias),
        "first_order_bias_median_ms": median(order_bias),
        "round_cluster_bootstrap95_mean_delta_ms": [lo,hi],
        "peak_mlx_memory_bytes": int(mx.get_peak_memory()),
        "records": records, "paired": paired,
        "notes": ["Within-pose randomized order of two AB and BA blocks; pose order shuffled per round.",
                  "Round zero excluded from primary performance analysis.",
                  "Timings include renderer call and mx.eval synchronization; not GPU event timings.",
                  "Round-cluster bootstrap is descriptive with few rounds, not proof of repeatable speedup.",
                  "No renderer or defaults modified; do not promote 512 based on this alone."]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents overwrites even if a concurrent run creates output.
    with args.out.open("x") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    print("RESULT phase116 PASS", json.dumps({k:v for k,v in report.items()
          if k not in ("records","paired")}), flush=True)

if __name__ == "__main__":
    main()
