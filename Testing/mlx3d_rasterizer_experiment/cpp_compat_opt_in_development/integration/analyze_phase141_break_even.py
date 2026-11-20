"""Phase 141 analysis: require both seeds and at least two densities per resolution."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import median

from qualification_common import digest, write_report
from run_phase140_143 import summarize

MIB = 2 ** 20
CACHE_BUDGET_DELTA = 256 * MIB

def analyze(report):
    groups = defaultdict(list)
    matrix = report.get("matrix")
    if matrix is None:
        matrix = summarize(report["workers"], "140")
    for row in matrix:
        if row["mode"] == "active":
            groups[(row["workload"], row["density"])].append(row)
    table = []
    for (workload, density), rows in sorted(groups.items()):
        changes = [r["warm_delta_ms"] for r in rows]
        cache = [r["memory"]["cache"] - r["baseline_memory"]["cache"] for r in rows]
        peaks = [r["memory"]["peak"] - r["baseline_memory"]["peak"] for r in rows]
        qualifies = len(rows) == 2 and len({r["seed"] for r in rows}) == 2 and all(
            r["warm_delta_ms"] < -max(0.5, (r["warm_ms"]-r["warm_delta_ms"])*0.01)
            and delta_cache <= CACHE_BUDGET_DELTA and delta_peak <= 0
            for r, delta_cache, delta_peak in zip(rows, cache, peaks))
        table.append(dict(workload=workload,density=density,seeds=[r["seed"] for r in rows],
                          counts=[r["base_count"] for r in rows],
                          unused=[r["unused_fraction"] for r in rows],
                          gaussians=rows[0]["gaussians"], warm_deltas_ms=changes,
                          median_warm_delta_ms=median(changes), peak_deltas_mib=[x/MIB for x in peaks],
                          cache_deltas_mib=[x/MIB for x in cache], passed_screen=qualifies,
                          observed_active=[r["counts"]["active"] for r in rows]))
    # A positive result in two seeds of one synthetic density remains exploratory.
    multiple_densities = {w for w in ("fullhd","highres")
        if sum(r["passed_screen"] and r["workload"] == w for r in table) >= 2}
    bands = []
    if report["status"] == "PASS" and len(table) == 9:
        for row in table:
            if row["passed_screen"] and row["workload"] in multiple_densities:
                dims = (1920,1080) if row["workload"] == "fullhd" else (2560,1440)
                bands.append([*dims,row["gaussians"],min(row["observed_active"]),max(row["observed_active"])])
    return dict(status="PASS" if report["status"]=="PASS" and len(table)==9 else "INCOMPLETE",
                matrix=table, trial_bands=bands,
                threshold_scope="previous observed active count for the same scene and camera resolution",
                screen="Both seeds >1% and >0.5ms gain, peak did not grow, cache growth <=256MiB, at least two densities per resolution",
                gpu_memory_pressure_cutoff_policy="Admission guard only; not used by selector",
                selection="test scoped bands" if bands else "reject automatic active sizing; qualify adaptive fallback")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report",type=Path)
    parser.add_argument("--out",type=Path,required=True)
    args = parser.parse_args()
    result = analyze(json.loads(args.report.read_text()))
    result["input_sha256"] = digest(args.report)
    write_report(args.out,result)
    print(json.dumps(result,indent=2))

if __name__ == "__main__":
    main()
