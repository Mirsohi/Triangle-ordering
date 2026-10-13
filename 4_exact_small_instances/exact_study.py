"""Enumerate global optima on small graphs; not a time-matched comparison."""
import itertools,json,time,hashlib
from pathlib import Path
import numpy as np
from methods import *
from validate import graph,independent_work
from dove.orderings import best_baseline,candidate_work
from dove.relaxation import optimize_order

HERE=Path(__file__).resolve().parent
def main():
    rng=np.random.default_rng(20260920)
    cycle=[(i,(i+1)%5) for i in range(5)]
    cases=[graph(5,cycle,'C5'),graph(6,[(i,j) for i in range(3) for j in range(3,6)],'K3_3'),
      graph(6,list(itertools.combinations(range(6),2)),'K6'),
      graph(6,[(0,1),(1,2),(0,2),(3,4),(4,5),(3,5),(0,3),(1,4),(2,5)],'triangular_prism'),
      graph(6,[(i,j) for i in range(3) for j in range(3,6)]+[(0,1)],'near_bipartite'),
      graph(7,[(0,1),(0,2),(1,2),(0,3),(0,4),(1,5),(2,6)],'core_periphery')]
    for j in range(20):
        cases.append(graph(7,[e for e in itertools.combinations(range(7),2) if rng.random()<.15+.03*j],f'ER7_{j:02}'))
    rows=[]
    for g in cases:
        orders=np.asarray(list(itertools.permutations(range(g.n))),dtype=np.int64)
        works=all_order_works(g.edges,orders);opt=int(works.min())
        _,ref,rw=best_baseline(g,seed=7);tp,trace=tangent(g,ref)
        dove=optimize_order(g,initial_order=ref,guard_with_builtins=False)
        dp,dt=tangent(g,dove.order)
        offsets,inc=csr_edges(g);state=insertion_state(g,ref);irng=np.random.default_rng(7)
        choices=irng.integers(g.m,size=10000,dtype=np.int64)
        sides=irng.integers(2,size=10000,dtype=np.int64);afters=irng.integers(2,size=10000,dtype=np.int64)
        delta,accepted=insertion_block(g.edges,offsets,inc,state,choices,sides,afters,1)
        iw=independent_work(g,state[0]);assert iw==rw+delta
        row=dict(graph=g.name,n=g.n,m=g.m,edges=g.edges.tolist(),orders_enumerated=len(orders),optimum=opt,
            optimum_order=orders[np.argmin(works)].tolist(),reference=rw,tangent=trace[-1],dove=dove.work,
            dove_tangent=dt[-1],insertion=iw,insertion_accepted=int(accepted))
        if not g.name.startswith('ER7'):
            finals=[];witness=None
            for p in orders:
                q,tr=tangent(g,p);finals.append(tr[-1])
                if witness is None and len(tr)==1 and tr[-1]>opt:
                    witness=dict(order=p.tolist(),work=tr[-1],optimum=opt,
                        twice_supergradient=twice_supergradient(g,degrees(g,rank_from_order(p))).tolist())
            row['all_start_tangent_optimal_fraction']=float(np.mean(np.array(finals)==opt))
            row['all_start_tangent_worst_work']=max(finals);row['nonglobal_fixed_point']=witness
        rows.append(row)
        (HERE/'exact_results.json').write_text(json.dumps({'scope':'Development, exact optima; fixed iterations/proposals, NOT matched time. Insertion: 10,000 neighbor proposals, seed 7.',
            'results':rows},indent=2))
        print(g.name,'opt',opt,'ref/tangent/dove/dove+tangent/insert',rw,trace[-1],dove.work,dt[-1],iw,flush=True)
    summary={method:dict(optimal=sum(r[method]==r['optimum'] for r in rows),total=len(rows),
        total_additive_gap=sum(r[method]-r['optimum'] for r in rows)) for method in ['reference','tangent','dove','dove_tangent','insertion']}
    (HERE/'exact_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
