#!/usr/bin/env python3
"""Phase139: CPU-only evidence audit and mode-isolated Phase138 memory comparison.

Reads the complete, original Phase138 v1 report. Never re-runs GPU workers,
modifies input evidence, or claims kernel-owned memory allocations.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

WORKLOADS = ("small", "fullhd", "highres")
MODES = ("auto", "selective_final3", "scatter512", "active", "adaptive")
BLOCKS = range(4)
MEMORY_KEYS = (
    "mlx_active_bytes", "mlx_peak_bytes", "mlx_cache_bytes",
    "process_peak_rss_bytes",
)
METRICS = {
    "first_frame_mlx_peak_bytes": ("timeline", "first_frame", "mlx_peak_bytes"),
    "final_mlx_peak_bytes": ("timeline", "after_release", "mlx_peak_bytes"),
    "final_mlx_cache_bytes": ("timeline", "after_release", "mlx_cache_bytes"),
    "final_mlx_active_bytes": ("timeline", "after_release", "mlx_active_bytes"),
    "final_process_rss_hwm_bytes": ("timeline", "after_release", "process_peak_rss_bytes"),
    "first_frame_ms": ("frame", "first_frame", "elapsed_ms"),
    "warm_frame_median_ms": ("frame_warm", "", "elapsed_ms"),
}
HEX64 = re.compile(r"[a-f0-9]{64}\Z")
EXPECTED_FAST_SHA = "56a2b0351f05eede765098267e2f9e114f40916136bcdd71a5fd4bbb91e31710"
EXPECTED_ADAPTER_SHA = "2554ff81b3ae12a40a1b7cc07b6cb79769dc0c4036216cf3bf6e8e91ada6425e"


class EvidenceError(ValueError):
    """Fail closed rather than including invalid or cherry-picked workers."""


def require(ok: bool, message: str):
    if not ok:
        raise EvidenceError(message)


def finite_nonnegative(value, label: str, *, positive: bool = False):
    require(isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and (value > 0 if positive else value >= 0),
            f"Invalid {label}: {value!r}")
    return value


def get_timeline(worker, label):
    matches = [r for r in worker["timeline"] if r.get("label") == label]
    require(len(matches) == 1, f"Missing/duplicate memory timeline label {label}")
    return matches[0]


def get_frame(worker, label):
    matches = [r for r in worker["frames"] if r.get("label") == label]
    require(len(matches) == 1, f"Missing/duplicate frame label {label}")
    return matches[0]


def metric(worker, key):
    category, label, field = METRICS[key]
    if category == "timeline":
        return get_timeline(worker, label)[field]
    if category == "frame":
        return get_frame(worker, label)[field]
    return statistics.median(get_frame(worker, f"warm_{i}")[field] for i in range(4))


def audit_worker(row, invariants):
    workload, block, mode = (row.get("workload"), row.get("block"), row.get("mode"))
    label = f"{workload}_{block}_{mode}"
    require(row.get("returncode") == 0, f"{label}: nonzero worker returncode")
    free = finite_nonnegative(row.get("free_memory_pct"), f"{label} free-memory percent")
    require(20 <= free <= 100, f"{label}: free-memory gate not >=20%")
    gate = row.get("gpu_gate")
    require(isinstance(gate, dict), f"{label}: no GPU admission record")
    util = finite_nonnegative(gate.get("utilization_pct"), f"{label} GPU utilization")
    require(util <= 30 and isinstance(gate.get("utc"), str) and gate["utc"],
            f"{label}: GPU admission exceeded 30% or lacks timestamp")
    worker = row.get("report")
    require(isinstance(worker, dict), f"{label}: missing nested report")
    require(worker.get("phase") == 138 and worker.get("status") == "PASS"
            and worker.get("workload") == workload and worker.get("block") == block
            and worker.get("mode") == mode and worker.get("invariants") == invariants,
            f"{label}: worker source/status/identity mismatch")
    require(isinstance(worker.get("timeline"), list)
            and isinstance(worker.get("frames"), list), f"{label}: missing raw timeline")

    expected_frames = ["first_frame", *(f"warm_{i}" for i in range(4))]
    expected_timeline = ["before_scene", "after_scene", "after_constructor",
                         *expected_frames, "before_release", "after_release"]
    if workload != "small":
        expected_frames.append("restored_frame")
        expected_timeline.extend(("after_small_scene_update", "small_scene_render",
                                  "after_original_scene_restore", "restored_frame"))
    require(set(f.get("label") for f in worker["frames"]) == set(expected_frames)
            and len(worker["frames"]) == len(expected_frames),
            f"{label}: missing/extra/duplicate output frame")
    labels = [x.get("label") for x in worker["timeline"]]
    require(set(labels) == set(expected_timeline) and len(labels) == len(expected_timeline),
            f"{label}: missing/extra/duplicate memory snapshot")

    peak = rss = 0
    for rec in worker["timeline"]:
        for key in MEMORY_KEYS:
            v = rec.get(key)
            finite_nonnegative(v, f"{label}/{rec.get('label')}/{key}")
        require(rec["mlx_active_bytes"] + rec["mlx_cache_bytes"] <= 13 * 2**30,
                f"{label}: allocator exceeds 13 GiB experiment budget")
        require(rec["mlx_peak_bytes"] >= peak and rec["process_peak_rss_bytes"] >= rss,
                f"{label}: cumulative peak or RSS high-water decreased")
        peak, rss = rec["mlx_peak_bytes"], rec["process_peak_rss_bytes"]
        require(rec["mlx_peak_bytes"] >= rec["mlx_active_bytes"],
                f"{label}: active MLX exceeds cumulative peak")

    for frame in worker["frames"]:
        require(isinstance(frame.get("sha256_rgb"), str)
                and bool(HEX64.fullmatch(frame["sha256_rgb"])),
                f"{label}: missing or malformed full-frame digest")
        finite_nonnegative(frame.get("elapsed_ms"), f"{label}/{frame['label']} time", positive=True)
        suffix = frame["label"].removeprefix("warm_")
        expected_camera = block + 3 + (int(suffix) if frame["label"].startswith("warm_") else 0)
        require(frame.get("camera_index") == expected_camera,
                f"{label}: camera sequence differs at {frame['label']}")
        require(isinstance(frame.get("last_frame_counts"), dict),
                f"{label}: missing frame-count metadata")
    return worker


def analyze(report):
    require(isinstance(report, dict) and report.get("phase") == 138,
            "Not a Phase138 report")
    require(report.get("status") == "PASS", "Phase138 did not finish PASS")
    expected = {(w, b, m) for w in WORKLOADS for b in BLOCKS for m in MODES}
    rows = report.get("workers")
    require(isinstance(rows, list) and len(rows) == len(expected)
            and report.get("matrix_size") == len(expected),
            "Phase138 must contain the complete 60-worker, four-block matrix")
    inv = report.get("invariants")
    require(isinstance(inv, dict) and inv.get("defaults") == {
        "native_fast": "mlx", "one_shot_adapter": "auto", "persistent_adapter": "auto"},
        "Protected sorting defaults changed or missing")
    require(all(isinstance(inv.get(k), str) and HEX64.fullmatch(inv[k])
                for k in ("fast_sha256", "adapter_sha256")),
            "Missing source SHA-256 evidence")
    require(inv["fast_sha256"] == EXPECTED_FAST_SHA and
            inv["adapter_sha256"] == EXPECTED_ADAPTER_SHA,
            "Phase138 v1 source digests differ from the M4 host-camera candidate")
    for key in ("experiment_sha256", "adaptive_sha256"):
        require(isinstance(report.get(key), str) and HEX64.fullmatch(report[key]),
                f"Missing {key} provenance")
    by_key = {}
    for row in rows:
        key = (row.get("workload"), row.get("block"), row.get("mode"))
        require(key in expected and key not in by_key,
                f"Unknown or duplicate worker {key}")
        by_key[key] = audit_worker(row, inv)
    require(set(by_key) == expected, "Missing required mode/workload/block worker")

    # All mode comparisons must be the identical camera and scene within a block.
    for w in WORKLOADS:
        for b in BLOCKS:
            auto = {f["label"]: f["sha256_rgb"]
                    for f in by_key[(w, b, "auto")]["frames"]}
            for m in MODES:
                candidate = {f["label"]: f["sha256_rgb"]
                             for f in by_key[(w, b, m)]["frames"]}
                require(auto == candidate, f"Bitwise RGB mismatch {w} block {b}: {m}")

    gate_history = report.get("gpu_gate_attempts")
    require(isinstance(gate_history, list) and len(gate_history) == len(rows),
            "Missing original GPU admission attempt history")
    busy = 0
    for history in gate_history:
        require(isinstance(history, list) and history, "Empty GPU gate history")
        require(history[-1].get("status") == "PASS", "GPU gate did not end PASS")
        busy += sum(x.get("status") == "BUSY" for x in history)
    accepted = [row["gpu_gate"]["utilization_pct"] for row in rows]
    min_free = min(row["free_memory_pct"] for row in rows)

    workload_summary = {}
    for w in WORKLOADS:
        modes = {}
        comparisons = {}
        for m in MODES:
            metrics = {}
            for k in METRICS:
                values = [metric(by_key[(w, b, m)], k) for b in BLOCKS]
                metrics[k] = {"median": statistics.median(values), "values_by_block": values}
            modes[m] = metrics
            if m == "auto":
                continue
            diffs = {}
            for k in METRICS:
                deltas = [metric(by_key[(w, b, m)], k)
                          - metric(by_key[(w, b, "auto")], k) for b in BLOCKS]
                control = statistics.median(metric(by_key[(w, b, "auto")], k)
                                            for b in BLOCKS)
                med = statistics.median(deltas)
                diffs[k] = {"median_mode_minus_auto": med,
                            "percent_of_auto_median": med / control * 100 if control else None,
                            "lower_blocks": sum(v < 0 for v in deltas),
                            "block_deltas": deltas}
            comparisons[m] = diffs
        decisions = Counter()
        for b in BLOCKS:
            for f in by_key[(w, b, "adaptive")]["frames"]:
                for row in f.get("adaptive_decisions", []):
                    if isinstance(row, dict) and isinstance(row.get("mode"), str):
                        decisions[row["mode"]] += 1
        workload_summary[w] = {"modes": modes, "comparisons_to_auto": comparisons,
                               "reported_adaptive_choices": dict(decisions)}

    return {
        "phase": 139, "status": "PASS", "source_phase": 138,
        "analysis_utc": datetime.now(timezone.utc).isoformat(),
        "validation": {"complete_workers": len(rows), "expected_workers": 60,
                       "bitwise_parity": "PASS", "source_invariants": inv,
                       "source_experiment_sha256": report["experiment_sha256"],
                       "source_adaptive_sha256": report["adaptive_sha256"],
                       "gpu_entry_max_pct": max(accepted),
                       "memory_free_min_pct": min_free,
                       "blocked_gpu_samples": busy},
        "workloads": workload_summary,
        "limitations": [
            "Observational comparisons across different fresh processes, not in-process paired allocation",
            "One fixed synthetic scene seed; no multiple-density or long-running stress qualification",
            "MLX peak and Darwin RSS are process high-water marks; cache is reusable and not an additive allocation",
            "Runner frame times are not counterbalanced or GPU-exclusive timing comparisons",
            "Successful worker mode selections are recorded, not independent proof of kernel dispatch",
            "Results do not authorize production default promotion or merging PR #1",
        ],
    }


def format_size(n):
    return f"{n / 2**20:,.1f} MiB"


def render_markdown(result):
    v = result["validation"]
    lines = ["# Phase 139 — CPU-only Phase 138 memory evidence analysis", "",
             f"Phase 138 full matrix: **{v['complete_workers']}/60 PASS**; "
             f"RGB bitwise parity **{v['bitwise_parity']}**.",
             f"Maximum admitted GPU utilization: **{v['gpu_entry_max_pct']}%**; "
             f"minimum recorded free system memory: **{v['memory_free_min_pct']}%**; "
             f"busy gate readings: **{v['blocked_gpu_samples']}**.", "",
             "Negative differences favor a candidate over the separate-process auto baseline. "
             "Values are medians of four independent process-block differences, **not kernel-owned allocation**.", ""]
    fields = (("final_mlx_peak_bytes", "MLX peak"),
              ("final_mlx_cache_bytes", "MLX cache after release"),
              ("final_process_rss_hwm_bytes", "Process RSS high-water"),
              ("warm_frame_median_ms", "Warmed frame"))
    for w in WORKLOADS:
        lines += [f"## {w}", "",
                  "| Mode | MLX peak Δ | MLX cache Δ | RSS HWM Δ | Warm Δ | Memory peak lower in blocks |",
                  "|---|---:|---:|---:|---:|---:|"]
        base = result["workloads"][w]["modes"]["auto"]
        lines.append("| auto (control) | — | — | — | — | — |")
        for m in MODES[1:]:
            diffs = result["workloads"][w]["comparisons_to_auto"][m]
            parts = []
            for k, _ in fields:
                val = diffs[k]["median_mode_minus_auto"]
                parts.append(f"{val:+.3f} ms" if k.endswith("_ms")
                             else f"{val / 2**20:+,.1f} MiB")
            count = diffs["final_mlx_peak_bytes"]["lower_blocks"]
            lines.append(f"| {m} | {' | '.join(parts)} | {count}/4 |")
        auto_peak = base["final_mlx_peak_bytes"]["median"]
        auto_cache = base["final_mlx_cache_bytes"]["median"]
        auto_rss = base["final_process_rss_hwm_bytes"]["median"]
        auto_ms = base["warm_frame_median_ms"]["median"]
        lines += ["", f"Auto median MLX peak {format_size(auto_peak)}, "
                  f"cache-after-release {format_size(auto_cache)}, "
                  f"RSS high-water {format_size(auto_rss)}, "
                  f"warmed frame {auto_ms:.3f} ms.",
                  f"Adaptive selection records: "
                  f"`{result['workloads'][w]['reported_adaptive_choices']}`.", ""]
    lines += ["## Interpretation limits", ""]
    lines += [f"- {limit}" for limit in result["limitations"]]
    lines += ["", "**Next decision:** Inspect the mode-isolated differences and their "
              "four-block consistency before selecting any additional GPU test. "
              "Do not automatically promote camera validation, active sizing, or the adaptive selector.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True,
                        help="Original completed Phase138 v1 JSON report")
    parser.add_argument("--out", type=Path, required=True,
                        help="New Phase139 JSON report, must not exist")
    parser.add_argument("--markdown", type=Path,
                        help="New .md summary; defaults to --out with .md extension")
    args = parser.parse_args()
    summary_path = args.markdown or args.out.with_suffix(".md")
    for target in (args.out, summary_path):
        if target.exists() or target.resolve() == args.input.resolve():
            parser.error(f"Refusing to replace an existing file/evidence: {target}")
    if args.out.resolve() == summary_path.resolve():
        parser.error("JSON and Markdown output paths must differ")
    raw = args.input.read_bytes()
    try:
        result = analyze(json.loads(raw))
    except (ValueError, TypeError, KeyError) as exc:
        parser.exit(1, f"PHASE139 INCOMPLETE: evidence rejected: {exc}\n")
    result["source_report_sha256"] = hashlib.sha256(raw).hexdigest()
    result["analyzer_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with args.out.open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    with summary_path.open("x", encoding="utf-8") as f:
        f.write(render_markdown(result))
    print(f"RESULT PHASE139 MEMORY_ANALYSIS PASS {args.out}")
    print(f"REPORT {summary_path}")
    print(f"WORKERS {result['validation']['complete_workers']}, GPU ENTRY MAX "
          f"{result['validation']['gpu_entry_max_pct']}%, "
          f"BITWISE {result['validation']['bitwise_parity']}")
    for w in WORKLOADS:
        print(w + ":")
        for m in MODES[1:]:
            x = result["workloads"][w]["comparisons_to_auto"][m]
            peak = x["final_mlx_peak_bytes"]
            print(f"  {m:16s}: median MLX peak Δ "
                  f"{peak['median_mode_minus_auto'] / 2**20:+.1f} MiB "
                  f"({peak['lower_blocks']}/4 lower)")


if __name__ == "__main__":
    main()
