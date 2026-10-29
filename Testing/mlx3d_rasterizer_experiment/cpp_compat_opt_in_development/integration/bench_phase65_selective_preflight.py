#!/usr/bin/env python3
"""Phase 65: test selective preflight on camera changes, not every frame.

Uses one persistent hybrid renderer; toggles its existing duplicate sizing
policy between auto and active before each render. The baseline stays auto.
Preflight is selected on substantial changes of focal length, rotation,
translation, or high prior duplicate-buffer occupancy. Retains radix4 and
automatic tile ranges in both paths. Pixel parity is mandatory.
Experimental: policy is not enabled in production.
"""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import mlx.core as mx
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
from mlx_gs.renderer.mlx3d_cpp_compat import PersistentCppCompatRenderer
from bench_phase49_breakdown import scene,camera,memory
from bench_phase57_active_sort import pose

def descriptor(cam):
    return {'f':float(cam.fx),'w2c':np.asarray(cam.W2C)}

def trigger(previous,current,prior_active,prior_capacity):
    if previous is None:return False,['initial']
    a=descriptor(previous);b=descriptor(current);reasons=[]
    if abs(b['f']/a['f']-1)>0.15:reasons.append('zoom')
    if np.linalg.norm(b['w2c'][:3,:3]-a['w2c'][:3,:3])>0.16:reasons.append('rotation')
    if np.linalg.norm(b['w2c'][:3,3]-a['w2c'][:3,3])>0.4:reasons.append('translation')
    if prior_capacity and prior_active/prior_capacity>0.9:reasons.append('occupancy')
    return bool(reasons),reasons

def measure(e,cam):
    r=e._renderer;before=int(r._capacity);policy=r.duplicate_sort_mode
    start=time.perf_counter();image=e.render(cam);mx.eval(image)
    elapsed=(time.perf_counter()-start)*1000
    c=dict(r.last_frame_counts)
    return image,{'ms':elapsed,'policy':policy,'active':int(c['active']),
                  'before':before,'used':int(c['capacity']),
                  'after':int(r._capacity),'overflow_inferred':bool(before and int(c['active'])>before)}

def summary(rows,mode):
    vals=np.array([r[mode]['ms'] for r in rows])
    return {'p50':float(np.percentile(vals,50)),
            'p95':float(np.percentile(vals,95)),
            'p99':float(np.percentile(vals,99)),
            'maximum':float(vals.max()),'total':float(vals.sum())}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--n',type=int,default=700000)
    p.add_argument('--width',type=int,default=1920)
    p.add_argument('--height',type=int,default=1080)
    p.add_argument('--out',type=Path,default=Path('phase65_selective_preflight.json'))
    a=p.parse_args()
    if a.out.exists():p.error('output exists')
    if not 1<=a.n<=800000 or min(a.width,a.height)<1:p.error('invalid arguments')
    gs=scene(a.n)
    engines={m:PersistentCppCompatRenderer(gs,duplicate_sort_mode='auto',
             tile_range_mode='auto',tile_sort_impl='radix4') for m in ('auto','selective')}
    cam0=camera(a.width,a.height,0)
    for e in engines.values():measure(e,cam0)
    sequence=[0,1,14,15,16,17,18,19,20,29,30,31,32,33,34,35,44,0,1,44,0,1]
    rng=np.random.default_rng(65)
    prior=None;rows=[]
    for k,index in enumerate(sequence):
        cam=pose(a.width,a.height,index)
        hc=engines['selective']._renderer.last_frame_counts
        prior_active=int(hc.get('active',0));prior_capacity=int(engines['selective']._renderer._capacity)
        do_preflight,reasons=trigger(prior,cam,prior_active,prior_capacity)
        engines['selective']._renderer.duplicate_sort_mode='active' if do_preflight else 'auto'
        imgs={};data={}
        for mode in rng.permutation(['auto','selective']):
            imgs[mode],data[mode]=measure(engines[mode],cam)
        if not np.array_equal(np.asarray(imgs['auto']),np.asarray(imgs['selective'])):
            raise AssertionError(f'parity failure step={k} pose={index}')
        row={'step':k,'pose':index,'preflight':do_preflight,'reasons':reasons,**data}
        rows.append(row)
        print(f"PASS {k+1}/{len(sequence)} pose={index} preflight={do_preflight} "
              f"auto={data['auto']['ms']:.2f} selective={data['selective']['ms']:.2f} ms "
              f"active={data['selective']['active']} reasons={','.join(reasons)}",flush=True)
        prior=cam
    result={'phase':65,'n':a.n,'resolution':[a.width,a.height],
            'stats_ms':{m:summary(rows,m) for m in engines},
            'preflight_count':sum(x['preflight'] for x in rows),'rows':rows,
            'process_memory_after':memory(),
            'notes':['Selection is camera heuristic, not proof overflow will occur.',
                     'Unpredicted overflow still uses existing rerender safety path.',
                     'Bitwise parity enforced on all 22 camera transitions.',
                     'Prototype does not modify production defaults.']}
    a.out.write_text(json.dumps(result,indent=2)+'\n')
    print('RESULT',result['stats_ms'],'preflight_count',result['preflight_count'],
          'saved',a.out,flush=True)

if __name__=='__main__':main()
