#!/usr/bin/env python3
"""Phase 52: separate capacity sizing from buffer reuse in a controlled ablation.

Three independent persistent instances share identical Gaussian values:
cached capacity; capacity-reset each frame; capacity-reset then pre-size to
observed stable capacity (avoids first-frame sizing sync). Parity every frame.
This tests the private allocator state, not GPU kernel-level attribution.
"""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
import numpy as np
import mlx.core as mx
from mlx_gs.renderer.mlx3d_cpp_compat import PersistentCppCompatRenderer
from bench_phase49_breakdown import scene,camera

def run(fn):
    t=time.perf_counter();im=fn();mx.eval(im)
    return (time.perf_counter()-t)*1000,np.asarray(im)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n",type=int,default=700000)
    p.add_argument("--width",type=int,default=1920)
    p.add_argument("--height",type=int,default=1080)
    p.add_argument("--rounds",type=int,default=12)
    p.add_argument("--out",type=Path,default=Path("phase52_capacity_presize.json"))
    a=p.parse_args()
    if a.out.exists():p.error("Output exists")
    if a.n<1 or a.n>800000 or min(a.width,a.height,a.rounds)<1:p.error("Invalid args")
    gs=scene(a.n)
    names=("cached","reset","presized")
    renderers={k:PersistentCppCompatRenderer(gs) for k in names}
    base=camera(a.width,a.height,0)
    for renderer in renderers.values():run(lambda r=renderer:r.render(base))
    # Use the largest observed warm capacity, with 2x margin for camera changes.
    capacity=max(r._renderer._capacity for r in renderers.values())*2
    records=[];timings={k:[] for k in names}
    for i in range(a.rounds+3):
        c=camera(a.width,a.height,i+1)
        outputs={}
        # Rotate order to reduce thermal/allocator scheduling bias.
        order=names[i%3:]+names[:i%3]
        for mode in order:
            r=renderers[mode]
            if mode=="reset":r._renderer._capacity=0
            elif mode=="presized":r._renderer._capacity=capacity
            ms,img=run(lambda:r.render(c))
            outputs[mode]=img
            if i>=3:
                timings[mode].append(ms)
                records.append({"frame":i-3,"mode":mode,"ms":ms,
                                "capacity_after":r._renderer._capacity})
        if not (np.array_equal(outputs["cached"],outputs["reset"]) and
                np.array_equal(outputs["cached"],outputs["presized"])):
            raise AssertionError(f"Bitwise parity failure frame {i}")
        if i>=3:
            print(f"PASS {i-2}/{a.rounds} cached={timings['cached'][-1]:.2f}ms "
                  f"reset={timings['reset'][-1]:.2f}ms presized={timings['presized'][-1]:.2f}ms",
                  flush=True)
    med={k:float(np.median(v)) for k,v in timings.items()}
    summary={"phase":52,"n":a.n,"resolution":[a.width,a.height],
             "rounds":a.rounds,"presized_capacity":capacity,
             "medians_ms":med,"ratios_to_cached":{k:med[k]/med["cached"] for k in names},
             "samples":records,
             "notes":["Private capacity override is test-only.",
                      "Pre-sized condition avoids initial sizing sync but uses an oversized buffer; sorting extra capacity may affect speed.",
                      "Results do not separately time Metal allocation or command-buffer submission."]}
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(summary,indent=2)+"\n")
    print("Medians:",med,"Saved:",a.out)

if __name__=="__main__":main()
