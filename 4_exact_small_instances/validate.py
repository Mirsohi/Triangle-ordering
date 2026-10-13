"""Independent mathematical and incremental-update checks. Run on local CPU."""
import itertools,json,time
from pathlib import Path
import numpy as np
from methods import *
from dove.graph import GraphData
from dove.orderings import candidate_work
from dove.global_relaxation import neural_objective_and_gradient
from sparse_neural import objective

def graph(n,edges,name='validation'):
    es=np.asarray(sorted(tuple(sorted(e)) for e in edges),dtype=np.int32).reshape(-1,2)
    rows=[[] for _ in range(n)]
    for u,v in es:rows[u].append(v);rows[v].append(u)
    return GraphData(name,es,tuple(np.array(r,dtype=np.int32) for r in rows),np.arange(n),'small')

def independent_work(g,order):
    rank={int(v):i for i,v in enumerate(order)};d=[0]*g.n
    for u,v in g.edges:d[u if rank[int(u)]<rank[int(v)] else v]+=1
    return sum(min(d[u],d[v]) for u,v in g.edges)

def main():
    start=time.perf_counter();rng=np.random.default_rng(20260920)
    counts=dict(labelled_graphs=0,orders=0,fractional_certificates=0,concave_quadratic=0,insertion_proposals=0,sparse_dense=0,finite_difference_coordinates=0)
    maxgrad=0.;maxfd=0.
    for n in range(5):
        possible=list(itertools.combinations(range(n),2))
        orders=np.asarray(list(itertools.permutations(range(n))),dtype=np.int64).reshape(-1,n) if n else np.empty((1,0),dtype=np.int64)
        for mask in range(2**len(possible)):
            g=graph(n,[e for k,e in enumerate(possible) if mask>>k&1]);counts['labelled_graphs']+=1
            ws=all_order_works(g.edges,orders)
            for order,w0 in zip(orders,ws):
                assert independent_work(g,order)==w0
                d=degrees(g,rank_from_order(order));a=twice_supergradient(g,d)
                q=order_from_scores(a);w1=independent_work(g,q)
                u,v=g.edges.T
                assert a@d==2*w0
                assert 2*w1<=np.minimum(a[u],a[v]).sum()<=2*w0
                result,trace=tangent(g,order)
                assert independent_work(g,result)==trace[-1]<=w0
                counts['orders']+=1
            for p in [np.zeros(g.m),np.ones(g.m),np.full(g.m,.5),rng.random(g.m)]:
                for eps in [.001,.2,3.]:
                    q,U,B,a,x=soft_certificate(g,p,eps);u,v=g.edges.T
                    direct=float(((x[u]+x[v]-np.hypot(x[u]-x[v],eps)+eps)/2).sum())
                    assert np.isclose(U,direct,atol=1e-11)
                    assert independent_work(g,q)<=B+1e-9<=U+2e-9
                    counts['fractional_certificates']+=1
    for trial in range(300):
        n=int(rng.integers(3,25));g=graph(n,[e for e in itertools.combinations(range(n),2) if rng.random()<.25])
        p=rng.random(g.m);q,U,B,a,x=soft_certificate(g,p,.03)
        assert independent_work(g,q)<=B+1e-8<=U+2e-8
        counts['fractional_certificates']+=1
        # The general argument also works for a nonmonotone concave function.
        a=-2*x;q=order_from_scores(a);d=degrees(g,rank_from_order(q));u,v=g.edges.T
        assert np.isclose(a@d,np.minimum(a[u],a[v]).sum())
        assert -d@d<=-x@x+a@(d-x)+1e-8<=-x@x+2e-8
        counts['concave_quadratic']+=1
        offsets,inc=csr_edges(g);state=insertion_state(g,rng.permutation(n))
        for k in range(100):
            u,v=rng.choice(n,2,replace=False);after=bool(rng.integers(2));accept=bool(k%3)
            before=state[0].copy();old=independent_work(g,before);proposed=literal_move(before,u,v,after)
            delta=independent_work(g,proposed)-old
            got=insertion_step(g.edges,offsets,inc,state,int(u),int(v),after,k+1,accept)
            assert got==delta,(g.name,u,v,after,got,delta)
            expected=proposed if accept and delta<0 else before
            assert np.array_equal(state[0],expected)
            assert np.array_equal(state[1],rank_from_order(expected))
            assert np.array_equal(state[2],degrees(g,state[1]))
            assert not state[3].any()
            counts['insertion_proposals']+=1
    for n in [3,5,9,16]:
        g=graph(n,[e for e in itertools.combinations(range(n),2) if rng.random()<.4])
        for tau in [.05,.3,1.]:
            for tie in [False,True]:
                s=rng.normal(size=n)*.3
                if tie:s[1]=s[0]
                f,grad=objective(g,s,tau,.07);f0,grad0=neural_objective_and_gradient(g,s,tau,.07)
                assert np.isclose(f,f0,atol=1e-10)
                err=float(np.max(np.abs(grad-grad0)));maxgrad=max(maxgrad,err)
                assert np.allclose(grad,grad0,atol=2e-9,rtol=2e-8)
                counts['sparse_dense']+=1
                if not tie:
                    for j in range(n):
                        z=np.zeros(n);z[j]=1e-6
                        fd=(objective(g,s+z,tau,.07)[0]-objective(g,s-z,tau,.07)[0])/2e-6
                        maxfd=max(maxfd,abs(fd-grad[j]))
                        assert np.isclose(fd,grad[j],atol=3e-6,rtol=3e-5),(n,tau,j,fd,grad[j])
                        counts['finite_difference_coordinates']+=1
    # Warm up the exact proposal-block signature used in the timing study.
    if g.m:
        offsets,inc=csr_edges(g);state=insertion_state(g,np.arange(g.n))
        insertion_block(g.edges,offsets,inc,state,np.zeros(2,dtype=np.int64),np.zeros(2,dtype=np.int64),np.zeros(2,dtype=np.int64),1)
        assert independent_work(g,state[0])==work_degrees(g,state[2])
    result=dict(passed=True,counts=counts,max_dense_sparse_gradient_error=maxgrad,max_finite_difference_error=maxfd,seconds=time.perf_counter()-start)
    Path(__file__).with_name('validation.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
