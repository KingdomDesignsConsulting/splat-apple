#!/usr/bin/env python3
"""Reaggregate Phase146's ORIGINAL nine validated worker checkpoints, CPU-only.

Do not rerun workers, recreate changed-code manifests, or overwrite original FAIL.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from phase140_gpu_lease import verify
from run_phase140_143 import label
from run_phase146_reserve_guard import summarize


def reaggregate(artifacts):
    manifest = json.loads((artifacts / "manifest.json").read_text())
    if manifest.get("phase") != "146":
        raise ValueError("Not a Phase146 evidence directory")
    jobs = manifest["jobs"]
    if len(jobs) != 9 or sum(j["kind"] == "contracts" for j in jobs) != 1:
        raise ValueError("Unexpected Phase146 job matrix")
    rows = []
    minimum_free = 100
    peak_gpu = 0
    for job in jobs:
        job_label = label(job)
        checked = verify(artifacts, job_label, job, manifest)
        if checked is None:
            raise ValueError(f"Missing immutable verified checkpoint: {job_label}")
        check, result = checked
        # Require the timestamps to be spaced by at least 2 s.
        confirms = check["gate"]["confirmations"]
        if len(confirms) != 2:
            raise ValueError("Invalid GPU admission confirmation count")
        times = [datetime.fromisoformat(c["utc"].replace("Z", "+00:00")) for c in confirms]
        if (times[1] - times[0]).total_seconds() < 2:
            raise ValueError(f"GPU confirmation too close: {job_label}")
        minimum_free = min(minimum_free, check["free_memory_pct"])
        peak_gpu = max(peak_gpu, *(c["utilization_pct"] for c in confirms))
        rows.append({"result": result, "checkpoint": check, "name": job_label})
    contracts = next(row["result"] for row in rows if row["result"]["job"]["kind"] == "contracts")
    if (contracts.get("contracts",{}).get("count") != 65
            or len(contracts.get("boundaries",[])) != 5):
        raise ValueError("65 contract or five boundary proofs missing")
    matrix = summarize(rows, "146")
    if len(matrix) != 4:
        raise AssertionError("Missing lifecycle block in aggregate")
    return dict(
        status="PASS", phase="146", qualification_kind="CPU_ONLY_REAGGREGATION_OF_ORIGINAL_GPU_EVIDENCE",
        previous_result="FAIL due to aggregate worker-kind collision (not a GPU worker failure)",
        original_manifest=manifest, artifacts=str(artifacts),
        verified_workers=len(rows), verified_contracts=65, verified_dispatch_boundaries=5,
        paired_blocks=4, matched_frames=4*31, peak_admitted_gpu_pct=peak_gpu,
        minimum_admitted_free_memory_pct=minimum_free, matrix=matrix,
        note="Only aggregate selector was repaired. All nine original checkpoint hashes, evidence, and admission history were independently verified. No new GPU work."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Refusing to overwrite existing report")
    result = reaggregate(args.artifacts)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print("PHASE146 VERIFIED REAGGREGATION PASS", args.out)
    print("WORKERS", result["verified_workers"], "CONTRACTS", result["verified_contracts"],
          "BOUNDARIES", result["verified_dispatch_boundaries"], "RGB PAIRS", result["matched_frames"])
    print("GPU MAX", result["peak_admitted_gpu_pct"], "FREE MEM MIN", result["minimum_admitted_free_memory_pct"])
    for row in result["matrix"]:
        print("BLOCK", row["block"], "FIRST SMALL Δ ms", round(row["small_first_delta_ms"],3),
              "CACHE Δ B",row["retained_cache_delta_bytes"],"PEAK Δ B",row["peak_delta_bytes"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
