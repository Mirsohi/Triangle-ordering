"""Local Numba adaptive-forward counting diagnostic; not a production benchmark."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
import itertools,json,time,csv
from pathlib import Path
import numpy as np
from numba import njit
from methods import ROOT
from dove.graph import load_edgelist
from dove.orderings import candidate_work
from validate import graph,independent_work

HERE=Path(__file__).resolve().parent
@njit(cache=True)
def prepare(edges,order):
    n=len(order);rank=np.empty(n,np.int64);degree=np.zeros(n,np.int64)
    for i in range(n):rank[order[i]]=i
    for u,v in edges:degree[u if rank[u]<rank[v] else v]+=1
    offsets=np.empty(n+1,np.int64);offsets[0]=0
    for i in range(n):offsets[i+1]=offsets[i]+degree[i]
    positions=offsets[:-1].copy();out=np.empty(len(edges),np.int32)
    for u,v in edges:
        a,b=(u,v) if rank[u]<rank[v] else (v,u)
        out[positions[a]]=b;positions[a]+=1
    for i in range(n):out[offsets[i]:offsets[i+1]].sort()
    return offsets,out

@njit(cache=True)
def count(edges,offsets,out):
    triangles=0;probes=0
    for u,v in edges:
        if offsets[u+1]-offsets[u]>offsets[v+1]-offsets[v]:u,v=v,u
        for i in range(offsets[u],offsets[u+1]):
            target=out[i];lo=offsets[v];hi=offsets[v+1];probes+=1
            while lo<hi:
                mid=(lo+hi)//2
                if out[mid]<target:lo=mid+1
                else:hi=mid
            if lo<offsets[v+1] and out[lo]==target:triangles+=1
    return triangles,probes

def validate():
    tests=0
    for n in range(5):
        possible=list(itertools.combinations(range(n),2))
        for mask in range(2**len(possible)):
            g=graph(n,[e for k,e in enumerate(possible) if mask>>k&1]);es=set(map(tuple,g.edges))
            truth=sum((u,v) in es and (u,w) in es and (v,w) in es for u,v,w in itertools.combinations(range(n),3))
            for p in itertools.permutations(range(n)):
                order=np.asarray(p,dtype=np.int64);offsets,out=prepare(g.edges,order)
                tri,probes=count(g.edges,offsets,out)
                assert tri==truth and probes==independent_work(g,order);tests+=1
    return tests

def main():
    dest=HERE/'matched_time_20260923';assert (dest/'COMPLETE.json').exists()
    assert not (dest/'kernel_rows.jsonl').exists(),'Preserve earlier results; use a new run directory'
    started=time.perf_counter();checks=validate();warm=time.perf_counter()-started
    rows=[json.loads(s) for s in (dest/'rows.jsonl').read_text().splitlines()]
    selected=[r for r in rows if r['group']=='public' and r['budget']==4.]
    outputs=[];rng=np.random.default_rng(20260923)
    for r in csv.DictReader((ROOT/'data/fresh20260914/manifest.csv').open()):
        g=load_edgelist(HERE.parent/'audit_results/fresh_graphs'/r['path'],r['name'])
        cases=[dict(method='reference',seed=0,order_file=g.name+'.reference.npz',elapsed=0.,work=None)]+[x for x in selected if x['graph']==g.name]
        orders={x['order_file']:np.load(dest/'orders'/x['order_file'])['order'] for x in cases}
        # All final orders, all five search seeds; no best-seed selection.
        jobs=[(x,repeat) for repeat in range(7) for x in cases];rng.shuffle(jobs)
        expected=None
        for x,repeat in jobs:
            p=orders[x['order_file']];begin=time.perf_counter();offsets,out=prepare(g.edges,p);prep=time.perf_counter()-begin
            begin=time.perf_counter();tri,probes=count(g.edges,offsets,out);seconds=time.perf_counter()-begin
            assert probes==candidate_work(g,p)
            if expected is None:expected=tri
            assert tri==expected
            row=dict(graph=g.name,method=x['method'],seed=x['seed'],repeat=repeat,triangles=int(tri),probes=int(probes),
                preparation_seconds=prep,kernel_seconds=seconds,optimization_seconds=x['elapsed'])
            outputs.append(row)
            with (dest/'kernel_rows.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print('Kernel verified:',g.name,'triangles',expected,flush=True)
    (dest/'kernel_complete.json').write_text(json.dumps(dict(small_exhaustive_checks=checks,validation_and_warmup_seconds=warm,
        rows=len(outputs),repetitions=7,scope='Single-process local Numba binary-search membership implementation; no production AOT speed claim'),indent=2))

if __name__=='__main__':main()
