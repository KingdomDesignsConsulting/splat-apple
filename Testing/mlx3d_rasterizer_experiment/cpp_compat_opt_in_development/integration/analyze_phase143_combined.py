"""CPU-only comparison of the mode-isolated Phase 143 evidence."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import median

from qualification_common import write_report
from run_phase140_143 import summarize

def analyze(report):
    groups = defaultdict(list)
    matrix = report.get("matrix")
    if matrix is None:
        matrix = summarize(report["workers"], "143")
    for row in matrix:
        if row["mode"] == "combined":
            decision = row["selection"][-1]
            actual = row["actual"][-1]
            selected = decision["mode"]
            expected = ("final3" if selected == "selective_final3_scatter512" else
                        "radix4" if decision["capacity"] >= 2**22 else "mlx")
            if expected != actual["route"] or (expected == "final3" and actual["scatter_group"] != 512):
                raise ValueError("Phase 143 selected and traced routes diverged")
        groups[(row["workload"],row["mode"])].append(row)
    groups_out = []
    for (workload, mode), rows in sorted(groups.items()):
        warm = [r["warm_delta_ms"] for r in rows]
        policy = [f["policy_ms"] for row in report["workers"]
                  if row["result"]["job"]["workload"] == workload and row["result"]["job"]["mode"] == mode
                  for f in row["result"].get("frames",[]) if f["policy_ms"] is not None]
        movement = {key:median(r["memory"][key]-r["baseline_memory"][key] for r in rows)
                    for key in ("active","cache","peak","rss")}
        groups_out.append(dict(workload=workload,mode=mode,blocks=len(rows),warm_median_ms=median(warm),
            warm_range_ms=[min(warm),max(warm)],warmed_wins=sum(d<0 for d in warm),
            first_median_ms=median(r["first_delta_ms"] for r in rows),memory_median_delta_bytes=movement,
            median_captured_policy_ms=median(policy) if policy else None,
            trace_by_process=[r["actual"] for r in rows]))
    contracts = next((row["result"] for row in report["workers"]
                      if row["result"]["job"]["kind"] == "contracts"), None)
    passing_contracts = sum(r["status"]=="PASS" for r in contracts["contracts"]["checks"]) if contracts else 0
    complete = (report["status"] == "PASS" and len(groups_out) == 15
                and all(row["blocks"] == 4 for row in groups_out) and passing_contracts == 65
                and len(contracts["boundaries"]) == 5)
    return dict(correctness_and_matrix_status="PASS" if complete else report["status"],
                performance_acceptance="separate workload decision",contracts_passed=passing_contracts,rows=groups_out)

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("report",type=Path)
    p.add_argument("--out",required=True,type=Path)
    a = p.parse_args()
    write_report(a.out,analyze(json.loads(a.report.read_text())))
