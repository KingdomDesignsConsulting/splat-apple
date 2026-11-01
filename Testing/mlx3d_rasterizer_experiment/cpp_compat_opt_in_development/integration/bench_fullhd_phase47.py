#!/usr/bin/env python3
"""Phase 47: full-HD scale comparison, with the Phase 45 parity harness.

Uses paired moving-camera timings and checks every rendered frame bitwise.
Synthetic scenes are useful for regression, not representative of trained splats.
Run from the separate feature worktree with gs-mps.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--counts",default="200000,700000")
    ap.add_argument("--resolutions",default="1280x720,1920x1080")
    ap.add_argument("--rounds",type=int,default=6)
    ap.add_argument("--out-dir",type=Path,default=Path("phase47_fullhd"))
    args=ap.parse_args()
    try: counts=[int(v) for v in args.counts.split(",")]
    except ValueError: ap.error("invalid Gaussian counts")
    if not counts or any(n<=0 or n>800000 for n in counts): ap.error("counts must be 1..800000")
    if args.rounds<1: ap.error("rounds must be positive")
    resolutions=[]
    for r in args.resolutions.split(","):
        try: w,h=map(int,r.lower().split("x"))
        except ValueError: ap.error("invalid resolution, use WIDTHxHEIGHT")
        if w<=0 or h<=0: ap.error("resolution must be positive")
        resolutions.append((w,h))
    if args.out_dir.exists(): ap.error(f"output directory already exists: {args.out_dir}")
    args.out_dir.mkdir(parents=True)
    harness=Path(__file__).with_name("bench_persistent_density.py")
    summary={"phase":47,"counts":counts,"rounds":args.rounds,"cases":[]}
    for w,h in resolutions:
        name=f"{w}x{h}"
        result_path=args.out_dir/f"{name}.json"
        cmd=[sys.executable,str(harness),"--counts",args.counts,
             "--width",str(w),"--height",str(h),"--rounds",str(args.rounds),
             "--out",str(result_path)]
        print("RUN",name,flush=True)
        with (args.out_dir/f"{name}.log").open("w") as log:
            proc=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
        if proc.returncode:
            raise RuntimeError(f"Test {name} failed with exit code {proc.returncode}; inspect {args.out_dir / (name+'.log')}")
        data=json.loads(result_path.read_text())
        for case in data["cases"]:
            med=case["median_ms"]
            ratio=med["persistent"]/med["one_shot"]
            record={"resolution":name,"n":case["n"],
                    "median_ms":med,"ratio_of_medians":ratio,
                    "persistent_fps_from_median":1000.0/med["persistent"],
                    "one_shot_fps_from_median":1000.0/med["one_shot"],
                    "paired_ratio_median":case["paired_ratio_median"],
                    "paired_ratio_p10":case["paired_ratio_p10"],
                    "paired_ratio_p90":case["paired_ratio_p90"],
                    "parity_frames":case["timed_parity_frames"]}
            summary["cases"].append(record)
            print(f"PASS {name} n={case['n']}: one-shot={med['one_shot']:.3f}ms "
                  f"persistent={med['persistent']:.3f}ms "
                  f"persistent_fps={record['persistent_fps_from_median']:.2f} "
                  f"paired_ratio={case['paired_ratio_median']:.4f}",flush=True)
        (args.out_dir/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print("Saved:",args.out_dir/"summary.json")

if __name__=="__main__": main()
