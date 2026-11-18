"""Summarize executed overnight reports without pooling partial replicate sets."""
import argparse
import json
from pathlib import Path
from statistics import median

B = Path(__file__).resolve().parent
WORK = ("small", "fullhd", "highres")
EXPECTED = {128:1,129:4,130:1,132:1,133:4,137:4}

def selected(reports, phase, workload):
    attempts = [
        [row for row in doc["workers"] if row["phase"] == phase and row["workload"] == workload]
        for doc in reports
    ]
    for rows in reversed(attempts):
        passing = [row for row in rows if row.get("report", {}).get("status") == "PASS"]
        if {r["block"] for r in passing} == set(range(EXPECTED[phase])):
            return passing
    return []

def summary(phase, workload, rows):
    result = [r["report"]["result"] for r in rows]
    if phase == 128:
        row = result[0]["medians_ms"]
        return {**row, "matrix_probe_saved_ms": row["matrix_finite_gpu"]-row["camera_host_candidate"]}
    if phase == 129:
        return {key: median(r[key] for r in result) for key in
                ("source_verification_and_load_ms","constructor_ms","first_render_ms",
                 "post_warm_one_shot_ms")} | {"warm_ms": median(median(r["warmed_ms"]) for r in result)}
    if phase == 130:
        timeline = result[0]["timeline"]
        original = [row for row in timeline if row["label"].endswith("after_original_replacement")]
        return {"timeline": timeline, "original_cycle_active_bytes": [row["active_bytes"] for row in original],
                "original_cycle_cache_bytes": [row["cache_bytes"] for row in original]}
    if phase == 132:
        return result[0]
    if phase == 133:
        deltas = [median(r["paired_delta_ms"]) for r in result]
        return {"block_delta_ms": deltas, "median_block_delta_ms":median(deltas),
                "block_delta_range_ms":[min(deltas),max(deltas)],
                "median_baseline_ms":median(r["median_ms"]["auto"] for r in result),
                "first_ms_by_process":[r["cold"] for r in result],
                "dispatch":[r["dispatch"] for r in result]}
    deltas = [median(r["paired_adaptive_minus_auto_ms"]) for r in result]
    return {"block_delta_ms":deltas, "median_block_delta_ms":median(deltas),
            "block_delta_range_ms":[min(deltas),max(deltas)],
            "median_ms":{mode:median(r["median_ms"][mode] for r in result) for mode in result[0]["median_ms"]},
            "max_sample_ms":{mode:max(s["ms"][mode] for r in result for s in r["samples"])
                             for mode in result[0]["median_ms"]},
            "first_ms_by_process":[r["cold"] for r in result], "routing":[r["routing"] for r in result],
            "decisions":[r["adaptive_decisions"] for r in result], "transitions":result[0]["transitions"]}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("reports",type=Path,nargs="+")
    args = parser.parse_args()
    reports = [json.loads(path.read_text()) for path in args.reports]
    names = {128:"phase128_adapter_overhead_1.json",129:"phase129_first_render_1.json",
             130:"phase130_memory_lifecycle_1.json",132:"phase132_renderer_stages_1.json",
             133:"phase134_comparative_benchmark_matrix_1.json",
             137:"phase137_adaptive_qualification_1.json"}
    args.output_dir.mkdir(exist_ok=True)
    for phase, blocks in EXPECTED.items():
        workloads = {}
        for w in WORK:
            rows = selected(reports,phase,w)
            if rows:
                workloads[w] = dict(measurements=summary(phase,w,rows),
                                    memory_by_process=[r["report"]["memory"] for r in rows],
                                    source_workers=[r["artifact"] for r in rows], count=blocks)
        failed = any(row.get("report",{}).get("status") == "FAIL" or row.get("returncode", 0)
                     for row in reports[-1]["workers"] if row["phase"] == phase)
        status = "COMPLETE" if len(workloads)==3 else "FAIL" if failed else "INCOMPLETE"
        with (args.output_dir/names[phase]).open("x") as stream:
            json.dump(dict(phase=phase, execution=status, workloads=workloads,
                           source_reports=[path.name for path in args.reports],
                           note="Execution completeness does not award optimization acceptance."),
                      stream, indent=2)
            stream.write("\n")
        print(phase, status, ",".join(workloads))

if __name__ == "__main__":
    main()
