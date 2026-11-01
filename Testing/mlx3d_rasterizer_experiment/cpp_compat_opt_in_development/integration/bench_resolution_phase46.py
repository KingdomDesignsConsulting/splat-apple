#!/usr/bin/env python3
"""Phase 46: resolution-scaling study with per-frame bitwise parity checks.

Runs the Phase 45 harness independently for each resolution, retaining the
same camera-motion and interleaved timing protocol. Does not touch site-packages.
Run from splat-apple-cpp-compat worktree in the gs-mps environment.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--counts",default="200000,700000")
    p.add_argument("--resolutions",default="640x480,1280x720")
    p.add_argument("--rounds",type=int,default=8)
    p.add_argument("--out-dir",type=Path,default=Path("phase46_resolution"))
    a=p.parse_args()
    if a.rounds<1:p.error("rounds must be positive")
    resolutions=[]
    for item in a.resolutions.split(","):
        try:
            w,h=map(int,item.lower().split("x"))
        except ValueError:p.error("resolutions must be e.g. 640x480,1280x720")
        if min(w,h)<1:p.error("resolution dimensions must be positive")
        resolutions.append((w,h))
    if a.out_dir.exists():p.error("Refusing existing output directory: "+str(a.out_dir))
    a.out_dir.mkdir(parents=True)
    source=Path(__file__).with_name("bench_persistent_density.py")
    summary={"phase":46,"counts":a.counts,"rounds":a.rounds,"cases":[]}
    for w,h in resolutions:
        name=f"{w}x{h}"
        output=a.out_dir/f"{name}.json"
        cmd=[sys.executable,str(source),"--counts",a.counts,"--width",str(w),
             "--height",str(h),"--rounds",str(a.rounds),"--out",str(output)]
        print("RUN",name,flush=True)
        with (a.out_dir/f"{name}.log").open("w") as log:
            result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"{name}: exit {result.returncode}; see {a.out_dir/name}.log")
        data=json.loads(output.read_text())
        for item in data["cases"]:
            summary["cases"].append({
                "resolution":name,"n":item["n"],
                "median_ms":item["median_ms"],
                "paired_ratio_median":item["paired_ratio_median"],
                "paired_ratio_p10":item["paired_ratio_p10"],
                "paired_ratio_p90":item["paired_ratio_p90"],
                "timed_parity_frames":item["timed_parity_frames"],
            })
            med=item["median_ms"]
            print(f"PASS {name} n={item['n']}: one-shot={med['one_shot']:.3f}ms "
                  f"persistent={med['persistent']:.3f}ms "
                  f"paired_ratio={item['paired_ratio_median']:.4f}",flush=True)
        (a.out_dir/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print("Saved:",a.out_dir/"summary.json")

if __name__=="__main__":main()
