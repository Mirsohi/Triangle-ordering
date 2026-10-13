"""Numerical implementation checks for SCORE_GEOMETRY.md; analytic proofs stand separately."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
import itertools,json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from methods import soft_certificate
from validate import graph
from dove.relaxation import smooth_objective_and_gradient

def main():
    rng=np.random.default_rng(20260923);residuals=[];gradient_errors=[];bounds=[]
    for trial in range(80):
        n=int(rng.integers(3,15));g=graph(n,[e for e in itertools.combinations(range(n),2) if rng.random()<.25]);u,v=g.edges.T
        adjacency=csr_matrix((np.ones(2*g.m),(np.r_[u,v],np.r_[v,u])),shape=(n,n))
        count,labels=connected_components(adjacency)
        anchors={int(np.flatnonzero(labels==c)[0]) for c in range(count)}
        free=np.array([i for i in range(n) if i not in anchors],dtype=int)
        p=rng.uniform(.05,.95,g.m);x=np.bincount(u,p,minlength=n)+np.bincount(v,1-p,minlength=n)
        def potential(z):
            r=np.zeros(n);r[free]=z;probs=np.exp(-np.logaddexp(0.,r[v]-r[u]))
            mean=np.bincount(u,probs,minlength=n)+np.bincount(v,1-probs,minlength=n)
            return float(np.logaddexp(r[u],r[v]).sum()-r@x),(mean-x)[free]
        if len(free):
            sol=minimize(potential,np.zeros(len(free)),jac=True,method='BFGS',options={'gtol':1e-10,'maxiter':1000})
            r=np.zeros(n);r[free]=sol.x
        else:r=np.zeros(n)
        probs=np.exp(-np.logaddexp(0.,r[v]-r[u]));mean=np.bincount(u,probs,minlength=n)+np.bincount(v,1-probs,minlength=n)
        residual=float(np.max(np.abs(mean-x)));assert residual<2e-7;residuals.append(residual)
        s=rng.normal(size=n);tau=.7;epsilon=.1
        probs=np.exp(-np.logaddexp(0.,-(s[v]-s[u])/tau));q,U,B,a,x=soft_certificate(g,probs,epsilon)
        incidence=np.zeros((n,g.m));incidence[u,np.arange(g.m)]=1;incidence[v,np.arange(g.m)]=-1
        lap=(incidence*(probs*(1-probs)))@incidence.T
        _,grad=smooth_objective_and_gradient(g,s,tau,epsilon)
        err=float(np.max(np.abs(grad+lap@a/tau)));assert err<1e-10;gradient_errors.append(err)
        if g.m:
            eigen=np.linalg.eigvalsh(lap);gap=eigen[count]
            rank=np.empty(n,dtype=int);rank[q]=np.arange(n);d=np.bincount(np.where(rank[u]<rank[v],u,v),minlength=n)
            bound=float(tau*np.linalg.norm(x-d)*np.linalg.norm(grad)/gap);delta=U-B
            assert delta<=bound+1e-8;bounds.append(dict(delta=delta,bound=bound))
    output=dict(inversion_cases=len(residuals),max_inversion_residual=max(residuals),
        laplacian_gradient_checks=len(gradient_errors),max_gradient_error=max(gradient_errors),spectral_bound_checks=len(bounds))
    Path(__file__).with_name('geometry_checks.json').write_text(json.dumps(output,indent=2));print(json.dumps(output,indent=2))

if __name__=='__main__':main()
