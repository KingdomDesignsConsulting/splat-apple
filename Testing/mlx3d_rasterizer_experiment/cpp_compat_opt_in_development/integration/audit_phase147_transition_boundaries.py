#!/usr/bin/env python3
"""Phase147 CPU-only transition/boundary audit of verified Phase146 evidence.

No MLX import, worker execution, GPU access or original evidence modification.
"""
import argparse
import json
from pathlib import Path
from statistics import median

from phase140_gpu_lease import verify
from run_phase140_143 import label

SCENE_COUNTS = (1024,4096,4097,1024,1024,4096)
FRAME_TAGS = ("initial",) + tuple(
    f"{stage}_{cycle}"
    for cycle in range(6)
    for stage in ("small_first","small_warm","fullhd_restore",
                  "highres_return","highres_warm")
)


def audit(directory):
    manifest=json.loads((directory/"manifest.json").read_text())
    if manifest.get("phase") != "146" or len(manifest.get("jobs",[])) != 9:
        raise ValueError("Expected original nine-worker Phase146 manifest")
    reports={}
    for job in manifest["jobs"]:
        result=verify(directory,label(job),job,manifest)
        if result is None:
            raise ValueError(f"Missing verified checkpoint {job}")
        reports[(job["kind"],job["block"],job["mode"])]=result[1]
    if len(reports)!=9:
        raise ValueError("Duplicate worker job identity")
    contracts=reports[("contracts",0,"guard")]
    if contracts["contracts"].get("count")!=65 or len(contracts.get("boundaries",[]))!=5:
        raise ValueError("Contract/dispatch evidence incomplete")
    observations=[]
    for block in range(4):
        keep=reports[("lifecycle",block,"keep")]
        guarded=reports[("lifecycle",block,"guard")]
        ref={x["tag"]:x for x in keep["frames"]}
        trial={x["tag"]:x for x in guarded["frames"]}
        if set(ref)!=set(FRAME_TAGS) or set(trial)!=set(FRAME_TAGS):
            raise ValueError(f"Unexpected frame sequence block {block}")
        if keep["output_checksums"] != guarded["output_checksums"]:
            raise AssertionError(f"Bitwise mismatch block {block}")
        for cycle,count in enumerate(SCENE_COUNTS):
            tag=f"small_first_{cycle}"
            a,z=ref[tag],trial[tag]
            width=641 if cycle==4 else 640
            if a["camera"]!=z["camera"] or z["camera"][0]!=width or z["camera"][1]!=360:
                raise AssertionError(f"Camera mismatch: {block}/{cycle}")
            decision=z.get("guard_decision")
            if not isinstance(decision,dict):
                raise AssertionError(f"Missing guard decision {block}/{cycle}")
            should=count<=4096 and width==640 and z["inherited"]>=2**22
            if decision["applied"] is not should:
                raise AssertionError(f"Unexpected decision {block}/{cycle}: {decision}")
            if should:
                route=z["routes"]
                if (len(route)!=1 or route[0]["route"]!="mlx"
                    or z["counts"]["sorted"]>=2**22):
                    raise AssertionError(f"Oversized guarded sort at {block}/{cycle}")
            observations.append(dict(
                block=block,cycle=cycle,gaussians=count,width=width,
                applied=should,first_small_delta_ms=z["wall_ms"]-a["wall_ms"],
                keep_first_ms=a["wall_ms"],guard_first_ms=z["wall_ms"],
                keep_inherited=a["inherited"],guard_inherited=z["inherited"],
                keep_route=a["routes"],guard_route=z["routes"],
                guard_sorted=z["counts"]["sorted"],rgb_equal=a["rgb"]==z["rgb"]))
    summary=[]
    for cycle,count in enumerate(SCENE_COUNTS):
        rows=[z for z in observations if z["cycle"]==cycle]
        delta=median(z["first_small_delta_ms"] for z in rows)
        summary.append(dict(cycle=cycle,gaussians=count,width=rows[0]["width"],
                            applied_blocks=sum(x["applied"] for x in rows),
                            favorable_blocks=sum(x["first_small_delta_ms"]<0 for x in rows),
                            median_delta_ms=delta,
                            min_delta_ms=min(x["first_small_delta_ms"] for x in rows),
                            max_delta_ms=max(x["first_small_delta_ms"] for x in rows)))
    return dict(status="PASS",kind="CPU_ONLY_PHASE147_EXISTING_EVIDENCE_AUDIT",
                verified_worker_checkpoints=9,verified_contracts=65,
                verified_dispatch_boundaries=5,paired_frames=4*len(FRAME_TAGS),
                original_phase146_source_manifest=manifest,
                by_transition=summary,individual_observations=observations,
                note="All values derive from original Phase146 archived GPU workers. Does not prove general scene portability.")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifacts",required=True,type=Path)
    p.add_argument("--out",required=True,type=Path)
    args=p.parse_args()
    if args.out.exists():
        p.error("Refusing overwrite of report")
    result=audit(args.artifacts)
    with args.out.open("x") as fh:
        json.dump(result,fh,indent=2,sort_keys=True)
        fh.write("\n")
    print("PHASE147 BOUNDARY AUDIT PASS",args.out)
    print("ORIGINAL GPU CHECKPOINTS",result["verified_worker_checkpoints"],
          "RGB PAIRS",result["paired_frames"])
    for x in result["by_transition"]:
        print("CYCLE",x["cycle"],"N",x["gaussians"],"WIDTH",x["width"],
              "APPLIED",x["applied_blocks"],"/4",
              "MEDIAN DELTA ms",round(x["median_delta_ms"],3),
              "WINS",x["favorable_blocks"],"/4")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
