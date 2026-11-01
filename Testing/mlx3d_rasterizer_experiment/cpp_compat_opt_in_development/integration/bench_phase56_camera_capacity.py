#!/usr/bin/env python3
"""Phase 56: active tile-duplicate capacity requirement over varied cameras.

Diagnoses the largest active duplicate count, default capacity, and utilization
across camera translations and zooms; preserves validated source unchanged.
This is a correctness/capacity survey, not a timing benchmark.
"""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
import mlx.core as mx
import numpy as np
from types import SimpleNamespace
from mlx_gs.renderer.mlx3d_cpp_compat import PersistentCppCompatRenderer
from bench_phase49_breakdown import scene

def camera(w,h,tx,ty,zoom):
    w2c=np.eye(4,dtype=np.float32)
    w2c[0,3]=tx;w2c[1,3]=ty
    f=float(min(w,h))*1.2*zoom
    return SimpleNamespace(W=w,H=h,fx=f,fy=f,cx=w/2,cy=h/2,W2C=mx.array(w2c))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--n',type=int,default=700000)
    p.add_argument('--width',type=int,default=1920)
    p.add_argument('--height',type=int,default=1080)
    p.add_argument('--out',type=Path,default=Path('phase56_camera_capacity.json'))
    a=p.parse_args()
    if a.out.exists():p.error('output exists')
    if a.n<1 or a.n>800000 or min(a.width,a.height)<1:p.error('invalid scene')
    gauss=scene(a.n)
    engine=PersistentCppCompatRenderer(gauss)
    original=engine._renderer._composite
    seen={}
    def capture(*args,**kwargs):
        # _composite(xy, conic_opac, rgbh, order, offsets, counts,
        #            bbox, background, total, w, h, ...)
        seen['active']=int(args[8].item())
        return original(*args,**kwargs)
    engine._renderer._composite=capture
    records=[]
    # 3 zooms x 5 horizontal positions x 3 vertical positions.
    for zoom in (0.75,1.0,1.4):
        for tx in (-.7,-.35,0.,.35,.7):
            for ty in (-.35,0.,.35):
                seen.clear()
                c=camera(a.width,a.height,tx,ty,zoom)
                im=engine.render(c)
                mx.eval(im)
                img=np.asarray(im)
                if not np.isfinite(img).all():raise AssertionError('nonfinite output')
                count=seen.get('active')
                if count is None:raise RuntimeError('active count unavailable')
                cap=engine._renderer._capacity
                if cap<count:raise RuntimeError('overflow in observed capacity')
                rec={'zoom':zoom,'tx':tx,'ty':ty,'active_duplicates':count,
                     'capacity':cap,'utilization':count/cap}
                records.append(rec)
                print(f"PASS zoom={zoom:.2f} tx={tx:+.2f} ty={ty:+.2f} "
                      f"active={count} capacity={cap} utilization={count/cap:.3f}",flush=True)
    mx_counts=[r['active_duplicates'] for r in records]
    report={'phase':56,'n':a.n,'resolution':[a.width,a.height],
            'poses':len(records),'min_active':min(mx_counts),
            'max_active':max(mx_counts),'recommended_not_less_than':max(mx_counts),
            'records':records,
            'notes':['Sync instrumentation is intentional; times are not measured.',
                     'Max is an observation across 45 poses, not a globally safe capacity.',
                     'Renderer may grow capacity under high duplicate counts.']}
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,indent=2)+'\n')
    print('PASS:',len(records),'poses; active range',report['min_active'],report['max_active'])
    print('Saved:',a.out)

if __name__=='__main__':main()
