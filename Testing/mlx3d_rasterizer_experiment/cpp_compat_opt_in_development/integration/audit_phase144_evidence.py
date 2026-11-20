"""Read and verify all Phase 143 worker evidence before new GPU work."""
import argparse
from datetime import datetime
import json
from pathlib import Path
from statistics import median, quantiles
from phase140_gpu_lease import verify
from qualification_common import HERE, digest, invariants, write_report
from run_phase140_143 import label

def audit():
    folder = HERE/"qualification_evidence/phase140/phase143_completed"
    source = folder/"phase143_artifacts"
    manifest = json.loads((source/"manifest.json").read_text())
    parent = json.loads((folder/"phase143.json").read_text())
    comparison = json.loads((folder/"phase143_comparison.json").read_text())
    assert manifest["invariants"] == invariants()
    assert parent["status"] == comparison["correctness_and_matrix_status"] == "PASS"
    assert len(manifest["jobs"]) == 61
    rows = {}
    gates = []
    for job in manifest["jobs"]:
        check, result = verify(source,label(job),job,manifest)
        accepted = check["gate"]["confirmations"]
        assert (datetime.fromisoformat(accepted[1]["utc"])-datetime.fromisoformat(accepted[0]["utc"])).total_seconds() >= 2
        assert result["invariants"] == invariants()
        gates.append(max(r["utilization_pct"] for r in accepted))
        if job["kind"] == "contracts":
            assert len(result["contracts"]["checks"]) == 65
            assert all(r["status"] == "PASS" for r in result["contracts"]["checks"])
            assert len(result["boundaries"]) == 5
            continue
        rows[(job["workload"],job["mode"],job["block"])] = result
    summaries = {}
    process_rows = []
    for (workload, mode, block), result in rows.items():
        baseline = rows[workload,"auto",block]
        assert result["output_checksums"] == baseline["output_checksums"]
        process_rows.append(dict(workload=workload,mode=mode,block=block,
            first_ms=result["frames"][0]["ms"], frames=result["frames"],
            after_release=result["after_release"]))
    for workload in sorted({w for w,_,_ in rows}):
        summaries[workload] = {}
        for mode in sorted({m for _,m,_ in rows}):
            paired = []
            for block in range(4):
                result, baseline = rows[workload,mode,block], rows[workload,"auto",block]
                a, b = [{f["label"]:f for f in r["frames"]} for r in (result,baseline)]
                warm = lambda f: median(row["ms"] for name,row in f.items() if name.startswith("warm"))
                transition = a["replacement"]
                paired.append(dict(first_delta_ms=a["first"]["ms"]-b["first"]["ms"],
                    warm_delta_ms=warm(a)-warm(b), first_warm_ms=a["warm_0"]["ms"],
                    replacement_ms=transition["ms"], restore_ms=a["restore"]["ms"],
                    replacement_slots=transition["counts"]["sorted"],
                    replacement_active_slots=transition["counts"]["active"],
                    replacement_route=transition["routes"][-1]["route"],
                    replacement_cache_bytes=transition["memory"]["cache"],
                    after_release=result["after_release"]))
            first = [r["first_delta_ms"] for r in paired]
            q = quantiles(first,method="inclusive")
            summaries[workload][mode] = dict(blocks=paired,first_median_delta_ms=median(first),
                first_delta_iqr_ms=[q[0],q[2]],first_blocks_faster=sum(v<0 for v in first),
                replaced_median_ms=median(r["replacement_ms"] for r in paired))
    return dict(status="PASS", verified_workers=len(rows)+1, max_admitted_pct=max(gates),
                prior_first_frame=digest(HERE/"overnight_results/phase129_first_render_1.json"),
                prior_lifecycle=digest(HERE/"overnight_results/phase130_memory_lifecycle_1.json"),
                evidence_sha256={n:digest(folder/n) for n in ("phase143.json","phase143_comparison.json")},
                provenance=manifest, summary=summaries, processes=process_rows,
                limitations=["Process-first wall time includes populated driver caches",
                             "Phase 143 observed one shrink and restore per worker",
                             "No independent thermal or system memory pressure time series"])

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",required=True,type=Path)
    a=p.parse_args()
    report=audit()
    write_report(a.out,report)
    print("Verified",report["verified_workers"],"workers")
    for workload, modes in report["summary"].items():
        row=modes["combined"]
        print(workload, "first delta",round(row["first_median_delta_ms"],3),
              "wins",row["first_blocks_faster"],"replacement ms",round(row["replaced_median_ms"],3),
              "slots",[(x["replacement_slots"],x["replacement_active_slots"],x["replacement_route"]) for x in row["blocks"]])
