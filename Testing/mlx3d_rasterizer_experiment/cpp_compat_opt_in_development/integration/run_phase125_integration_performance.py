#!/usr/bin/env python3
"""Phase125: real fallback boundaries and counterbalanced integration regression."""
import argparse
from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import time
from qualification_common import (ROOT,HERE,ADAPTER,command,invariants,inventory,
    isolated_runtime,output_paths,write_report,gpu_gate,metadata,digest)


def worker(kind,workload,block,reference,out,rounds):
    sys.path.insert(0,str(ROOT))
    import mlx.core as mx
    import numpy as np
    import random,resource
    from mlx_gs.renderer import mlx3d_cpp_compat as api
    from bench_phase49_breakdown import scene,camera
    from bench_phase85_final3 import fast_camera
    cls=api._renderer_class();module=sys.modules[cls.__module__]
    def mem():
        return {'process_peak_rss_bytes':int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
                'mlx_active_bytes':int(mx.get_active_memory()),'mlx_peak_bytes':int(mx.get_peak_memory()),
                'mlx_cache_bytes':int(mx.get_cache_memory()),
                'scope':'process-wide, shared renderers/cache; not per-mode allocation; Darwin ru_maxrss is bytes'}
    def equal(a,b):
        mx.eval(a,b)
        aa=np.asarray(a);bb=np.asarray(b)
        assert np.isfinite(aa).all() and np.isfinite(bb).all()
        assert aa.shape==bb.shape and aa.dtype==bb.dtype==np.float32
        assert np.array_equal(aa.view(np.uint32),bb.view(np.uint32))
    @contextmanager
    def trace():
        recorded={'radix4':0,'final3':[],'scatter_groups':[],'final_groups':[]}
        originals={n:getattr(module,n) for n in ('_radix_tile_argsort','_selective_final3_argsort','_radix_scatter_kernel','_FINAL_INDICES')}
        def radix(*a,**k):recorded['radix4']+=1;return originals['_radix_tile_argsort'](*a,**k)
        def final(*a,**k):recorded['final3'].append(k.get('scatter_group',256));return originals['_selective_final3_argsort'](*a,**k)
        def scatter(*a,**k):recorded['scatter_groups'].append(list(k['threadgroup']));return originals['_radix_scatter_kernel'](*a,**k)
        def last(*a,**k):recorded['final_groups'].append(list(k['threadgroup']));return originals['_FINAL_INDICES'](*a,**k)
        try:
            module._radix_tile_argsort=radix;module._selective_final3_argsort=final
            module._radix_scatter_kernel=scatter;module._FINAL_INDICES=last
            yield recorded
        finally:
            for n,f in originals.items():setattr(module,n,f)
    def validate_trace(rec,eligible,group=512):
        if eligible:
            assert rec['radix4']==0 and rec['final3']==[group],rec
            assert rec['scatter_groups'] and all(g==[group,1,1] for g in rec['scatter_groups']),rec
            assert rec['final_groups']==[[256,1,1]],rec
        else:
            assert rec['radix4']==1 and rec['final3']==[] and rec['final_groups']==[],rec
            assert rec['scatter_groups'] and all(g==[256,1,1] for g in rec['scatter_groups']),rec
    def engine(gs,mode):
        return api.PersistentCppCompatRenderer(gs,duplicate_sort_mode='capacity',tile_range_mode='auto',tile_sort_impl=mode)
    if kind=='boundaries':
        gs=scene(64);rows=[]
        configs=[('capacity_'+str(c),1920,1080,c) for c in (7_999_999,8_000_000,8_000_001)]
        configs += [('tiles7999',304,6736,8_000_001),('tiles8000',1600,1280,8_000_001),('tiles8001',1008,2032,8_000_001)]
        for label,w,h,capacity in configs:
            tiles=((w+15)//16)*((h+15)//16)
            expected=capacity>=8_000_000 and tiles>=8_000 and tiles<=8192
            control=engine(gs,'selective_final3');test=engine(gs,'selective_final3_scatter512')
            control._renderer._capacity=capacity;test._renderer._capacity=capacity
            cam=fast_camera(camera(w,h,3))
            with trace() as rec:
                observed=test._renderer.render(cam);mx.eval(observed)
            ref=control._renderer.render(cam);mx.eval(ref)
            equal(observed['image'],ref['image']);equal(observed['alpha'],ref['alpha'])
            assert test._renderer.last_frame_counts==control._renderer.last_frame_counts
            assert test._renderer.last_frame_counts['sorted']==capacity
            validate_trace(rec,expected)
            rows.append({'name':label,'resolution':[w,h],'tiles':tiles,'capacity':capacity,
                         'eligible_expected':expected,'observed':rec,'rgb_alpha_bitwise':'PASS'})
            print('PASS actual dispatch',label,rec,flush=True)
        result={'phase':125,'kind':kind,'status':'PASS','cases':rows,'memory':mem()}
    else:
        configs={'small':(4096,640,360),'fullhd':(700000,1920,1080),'highres':(700000,2560,1440)}
        n,w,h=configs[workload];tiles=((w+15)//16)*((h+15)//16)
        gs=scene(n);cam=camera(w,h,3)
        modes={'control':'selective_final3','scatter512':'selective_final3_scatter512'}
        first=['control','scatter512'] if block in (0,3) else ['scatter512','control']
        engines={};cold={};first_images={}
        for name in first:
            start=time.perf_counter();engines[name]=engine(gs,modes[name]);constructor=(time.perf_counter()-start)*1000
            start=time.perf_counter();first_images[name]=engines[name].render(cam);mx.eval(first_images[name])
            cold[name]={'constructor_ms':constructor,'process_first_render_ms':(time.perf_counter()-start)*1000}
        equal(first_images['control'],first_images['scatter512'])
        active=max(e._renderer.last_frame_counts['active'] for e in engines.values())
        capacity=active*105//100+1024
        for e in engines.values():e._renderer._capacity=capacity
        for _ in range(3):
            for name in first:mx.eval(engines[name].render(cam))
        eligible=capacity>=8_000_000 and tiles>=8000 and module.final_pass_plan(tiles)['eligible']
        with trace() as rec:mx.eval(engines['scatter512'].render(cam))
        validate_trace(rec,eligible)
        rng=random.Random(12500+block)
        rows=[]
        def paired(order):
            images={};times={}
            for name in order:
                start=time.perf_counter();images[name]=engines[name].render(cam);mx.eval(images[name])
                times[name]=(time.perf_counter()-start)*1000
            equal(images['control'],images['scatter512'])
            assert engines['control']._renderer.last_frame_counts==engines['scatter512']._renderer.last_frame_counts
            return times
        for r in range(rounds):
            orders=[['control','scatter512'],['scatter512','control']];rng.shuffle(orders)
            for order in orders:rows.append({'round':r,'first':order[0],'ms':paired(order)})
        one=[]
        for r in range(3):
            order=first if r%2==0 else list(reversed(first));images={};times={}
            for name in order:
                start=time.perf_counter()
                images[name]=api.render_cpp_compat(gs,cam,duplicate_sort_mode='capacity',tile_range_mode='auto',tile_sort_impl=modes[name])
                mx.eval(images[name]);times[name]=(time.perf_counter()-start)*1000
            equal(images['control'],images['scatter512']);equal(images['control'],first_images['control'])
            one.append(times)
        # Compare the newly added public validation cost with the frozen adapter.
        spec=importlib.util.spec_from_file_location('_phase125_reference',reference)
        old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
        old.__file__=str(ADAPTER);old._MODULE_NAME='mlx3d.splatting._phase125_old_fast'
        prior=old.PersistentCppCompatRenderer(gs,duplicate_sort_mode='capacity',tile_range_mode='auto',tile_sort_impl='selective_final3')
        prior._renderer._capacity=capacity
        for _ in range(3):mx.eval(prior.render(cam))
        overhead=[]
        for r in range(4):
            for order in (['prior','current'],['current','prior']):
                imgs={};times={}
                for name in order:
                    e=prior if name=='prior' else engines['control']
                    start=time.perf_counter();imgs[name]=e.render(cam);mx.eval(imgs[name]);times[name]=(time.perf_counter()-start)*1000
                equal(imgs['prior'],imgs['current']);overhead.append(times)
        deltas=[r['ms']['scatter512']-r['ms']['control'] for r in rows]
        order_averaged=[statistics.mean([x for x,r in zip(deltas,rows) if r['round']==i]) for i in range(rounds)]
        result={'phase':125,'kind':kind,'status':'PASS','workload':workload,'block':block,'resolution':[w,h],
                'gaussians':n,'tiles':tiles,'capacity':capacity,'eligible':eligible,'actual_dispatch':rec,
                'cold_order':first,'cold':cold,'warm_pairs':rows,'warm_median_ms':{name:statistics.median(r['ms'][name] for r in rows) for name in modes},
                'order_averaged_mean_delta_512_ms':statistics.mean(order_averaged),
                'order_averaged_median_delta_512_ms':statistics.median(order_averaged),
                'oneshot_warm_new_renderer_samples_ms':one,
                'oneshot_median_ms':{name:statistics.median(r[name] for r in one) for name in modes},
                'adapter_overhead_samples_ms':overhead,
                'adapter_overhead_median_delta_ms':statistics.median(r['current']-r['prior'] for r in overhead),
                'memory':mem(),
                'notes':['process-first means first use in this process, not system/compiler-cache cold',
                         'one-shot costs include new scene preparation and initial capacity; not isolated cache cost',
                         'matched capacities for warmed persistent comparisons; both orders in every round',
                         'all RGB comparisons bitwise; boundary worker additionally checks alpha',
                         'no default promotion and no requirement that 512 win a noisy timing comparison']}
        print('PASS workload',workload,block,result['warm_median_ms'],'adapter_delta',result['adapter_overhead_median_delta_ms'],flush=True)
    result['invariants']=invariants();write_report(out,result)
    return 0


def wait_for_gpu_gate(report, *, attempts=12, delay_seconds=10):
    """Allow short-lived post-worker GPU activity to settle, without relaxing 30%."""
    readings=[]
    for attempt in range(attempts):
        if attempt:
            time.sleep(delay_seconds)
        try:
            reading=gpu_gate()
            readings.append({'attempt':attempt+1,'status':'PASS',**reading})
            report['gpu_gate_attempts'].append(readings)
            return reading
        except RuntimeError as exc:
            if not str(exc).startswith('GPU test blocked: utilization '):
                raise
            readings.append({'attempt':attempt+1,'status':'BUSY','message':str(exc)})
            print('GPU_GATE_BUSY',attempt+1,'of',attempts,str(exc),flush=True)
    report['gpu_gate_attempts'].append(readings)
    raise RuntimeError('GPU remained above 30% or unreadable after '
                       +str(attempts)+' checks; no new GPU worker started')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--worker',choices=['boundaries','workload']);p.add_argument('--workload',choices=['small','fullhd','highres'])
    p.add_argument('--block',type=int,default=0);p.add_argument('--reference',type=Path)
    p.add_argument('--rounds',type=int,default=8);a=p.parse_args()
    if a.worker:return worker(a.worker,a.workload,a.block,a.reference,a.out,a.rounds)
    if a.rounds<4:p.error('rounds >=4 required')
    out,logs=output_paths(a.out);before=invariants();packages=inventory()
    report={'phase':125,'status':'INCOMPLETE','environment':metadata(),'gpu_gates':[], 'gpu_gate_attempts':[], 'workloads':[]}
    try:
        rc,text,err=command(['git','show','6487258:mlx_gs/renderer/mlx3d_cpp_compat.py'])
        if rc:raise RuntimeError('Cannot load pre-Phase124 adapter')
        reference=logs/'reference_adapter.py';reference.write_text(text)
        with isolated_runtime(logs,with_mlx3d=True) as (py,tmp,env):
            if command([py,'-m','pip','check'],logs/'dependency_check.log',env=env):raise RuntimeError('Dependency check failed')
            def child(label,args):
                time.sleep(1)
                report['gpu_gates'].append(wait_for_gpu_gate(report))
                path=logs/(label+'.json')
                rc=command([py,Path(__file__).resolve(),*args,'--out',path],logs/(label+'.log'),env=env,timeout=1200)
                if rc or not path.exists():raise RuntimeError(label+' failed; inspect log')
                data=json.loads(path.read_text())
                if data.get('status')!='PASS' or data.get('invariants')!=before:raise AssertionError(label+' invalid evidence')
                return data
            report['boundaries']=child('boundaries',['--worker','boundaries'])
            for name in ('small','fullhd','highres'):
                blocks=[]
                for i in range(4):
                    blocks.append(child(name+'_'+str(i),['--worker','workload','--workload',name,'--block',str(i),'--reference',reference,'--rounds',str(a.rounds)]))
                    print('PASS Phase125',name,i,flush=True)
                means=[b['order_averaged_mean_delta_512_ms'] for b in blocks]
                report['workloads'].append({'name':name,'blocks':blocks,'mean_process_delta_512_ms':statistics.mean(means),
                    'median_process_delta_512_ms':statistics.median(means),'blocks_favor_512':sum(x<0 for x in means),
                    'median_adapter_overhead_ms':statistics.median(b['adapter_overhead_median_delta_ms'] for b in blocks)})
            command([py,'-m','pip','freeze'],logs/'package_inventory.log',env=env)
            report['status']='PASS'
    except Exception as exc:report['error']={'type':type(exc).__name__,'message':str(exc)}
    report['invariants']=invariants();assert before==report['invariants'] and packages==inventory()
    report['caller_environment_unchanged']=True;report['artifacts']=str(logs)
    report['acceptance']='Correctness and actual fallback dispatch required; wall-time effects observational; no default promotion'
    write_report(out,report);print('RESULT PHASE125',report['status'],out,flush=True)
    return 0 if report['status']=='PASS' else 1


if __name__=='__main__':sys.exit(main())
