#!/usr/bin/env python3
"""Phase 48: repeat the anomalous 700K Full-HD result with raw paired samples.

Independent subprocess repetitions of the validated Phase 45 moving-camera
harness. Every frame must still pass bitwise parity. No production changes.
"""
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n",type=int,default=700000)
    ap.add_argument("--width",type=int,default=1920)
    ap.add_argument("--height",type=int,default=1080)
    ap.add_argument("--rounds",type=int,default=16)
    ap.add_argument("--repeats",type=int,default=3)
    ap.add_argument("--out-dir",type=Path,default=Path("phase48_fullhd_repeat"))
    a=ap.parse_args()
    if a.n<1 or a.n>800000 or min(a.width,a.height,a.rounds,a.repeats)<1:
        ap.error("invalid dimensions/rounds/repeats")
    if a.out_dir.exists():ap.error("output directory exists: "+str(a.out_dir))
    a.out_dir.mkdir(parents=True)
    harness=Path(__file__).with_name("bench_persistent_density.py")
    records=[]
    for i in range(a.repeats):
        outfile=a.out_dir/f"repeat_{i+1}.json"
        cmd=[sys.executable,str(harness),"--counts",str(a.n),
             "--width",str(a.width),"--height",str(a.height),
             "--rounds",str(a.rounds),"--out",str(outfile)]
        print(f"RUN {i+1}/{a.repeats}",flush=True)
        with (a.out_dir/f"repeat_{i+1}.log").open("w") as log:
            p=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
        if p.returncode:
            raise RuntimeError(f"Repeat {i+1} exit {p.returncode}; inspect log")
        case=json.loads(outfile.read_text())["cases"][0]
        t=case["times_ms"]
        median=case["median_ms"]
        rec={"repeat":i+1,"n":a.n,"resolution":f"{a.width}x{a.height}",
             "medians_ms":median,"paired_ratio_median":case["paired_ratio_median"],
             "paired_ratio_p10":case["paired_ratio_p10"],
             "paired_ratio_p90":case["paired_ratio_p90"],
             "timing_samples_ms":t,"correctness_frames":case["timed_parity_frames"]}
        records.append(rec)
        (a.out_dir/"summary.json").write_text(json.dumps({"phase":48,"repeats":records},indent=2)+"\n")
        print(f"PASS repeat {i+1}: one-shot {median['one_shot']:.3f}ms, "
              f"persistent {median['persistent']:.3f}ms, "
              f"paired median {case['paired_ratio_median']:.4f}",flush=True)
    print("Saved:",a.out_dir/"summary.json")

if __name__=="__main__":main()
