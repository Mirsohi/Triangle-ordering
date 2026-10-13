"""Equivalent NeuralSort adapter with edge-only forward/reverse contraction."""
import numpy as np
from scipy.sparse import csr_matrix
from methods import ROOT
from dove.global_relaxation import position_probabilities,upper_half,lower_half

def objective(g,s,temperature,epsilon):
    if epsilon<=0:raise ValueError('positive smoothing required')
    h,p,normalizer,coefficients,signs=position_probabilities(s,temperature)
    kh=upper_half(h);u,v=g.edges.T
    ep=np.empty(g.m)
    for start in range(0,g.m,128):
        ix=slice(start,start+128)
        ep[ix]=np.einsum('ij,ij->j',h[:,u[ix]],kh[:,v[ix]])
    d=np.bincount(u,ep,minlength=g.n)+np.bincount(v,1-ep,minlength=g.n)
    gap=d[u]-d[v];root=np.hypot(gap,epsilon)
    value=float((.5*(d[u]+d[v]-root+epsilon)).sum())
    gd=np.bincount(u,.5*(1-gap/root),minlength=g.n)+np.bincount(v,.5*(1+gap/root),minlength=g.n)
    w=csr_matrix((gd[u]-gd[v],(u,v)),shape=(g.n,g.n))
    gh=(w@kh.T).T+(w.T@lower_half(h).T).T
    gp=(gh-(gh*h).sum(axis=0,keepdims=True))/normalizer
    gl=p*(gp-(gp*p).sum(axis=1,keepdims=True))/temperature
    ga=-gl.sum(axis=0)
    gradient=-(coefficients[:,None]*gl).sum(axis=0)+ga*signs.sum(axis=1)-ga@signs
    return value,gradient
