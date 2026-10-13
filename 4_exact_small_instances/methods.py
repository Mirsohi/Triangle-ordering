"""Sparse tangent rounds and literal integer-rank insertion. Local CPU only."""
from pathlib import Path
import sys
import numpy as np
from numba import njit
ROOT=Path(__file__).resolve().parents[1]/'audit_workspace/DOVE_AISTATS_2027_code_and_data'
sys.path.insert(0,str(ROOT/'src'))
from dove.orderings import rank_from_order,order_from_scores

def degrees(g,keys):
    u,v=g.edges.T
    before=(keys[u]<keys[v])|((keys[u]==keys[v])&(u<v))
    return np.bincount(np.where(before,u,v),minlength=g.n)

def work_degrees(g,d):
    u,v=g.edges.T
    return int(np.minimum(d[u],d[v]).sum(dtype=np.int64))

def twice_supergradient(g,d):
    u,v=g.edges.T
    low=d[u]<d[v];high=d[u]>d[v];tie=~(low|high)
    return (2*np.bincount(u[low],minlength=g.n)+2*np.bincount(v[high],minlength=g.n)
            +np.bincount(u[tie],minlength=g.n)+np.bincount(v[tie],minlength=g.n))

def soft_certificate(g,p,epsilon):
    if epsilon<=0 or len(p)!=g.m or not np.all(np.isfinite(p)) or np.any((p<0)|(p>1)):
        raise ValueError('finite complementary edge probabilities and epsilon>0 required')
    u,v=g.edges.T
    x=np.bincount(u,p,minlength=g.n)+np.bincount(v,1-p,minlength=g.n)
    gap=x[u]-x[v];root=np.hypot(gap,epsilon)
    # Stable expression for the smooth minimum.
    value=float((np.minimum(x[u],x[v])+.5*(epsilon-epsilon**2/(root+np.abs(gap)))).sum())
    w=.5*(1-gap/root)
    a=np.bincount(u,w,minlength=g.n)+np.bincount(v,1-w,minlength=g.n)
    delta=float(np.dot(a,x)-np.minimum(a[u],a[v]).sum())
    order=order_from_scores(a)
    return order,value,value-delta,a,x

def tangent(g,order,max_steps=100):
    """O(m+n) per iteration; sort once on final export. Exact integer adjoints."""
    keys=rank_from_order(order);d=degrees(g,keys);best=work_degrees(g,d);trace=[best]
    for _ in range(max_steps):
        a=twice_supergradient(g,d);nd=degrees(g,a);w=work_degrees(g,nd)
        if w>best:raise AssertionError('monotonicity violated')
        if w==best:break
        keys,d,best=a,nd,w;trace.append(best)
    return order_from_scores(keys),trace

def csr_edges(g):
    u,v=g.edges.T
    vertices=np.concatenate([u,v]);edgeids=np.tile(np.arange(g.m,dtype=np.int64),2)
    ix=np.argsort(vertices,kind='stable')
    offsets=np.r_[0,np.cumsum(np.bincount(vertices,minlength=g.n))].astype(np.int64)
    return offsets,edgeids[ix]

def insertion_state(g,order):
    order=np.asarray(order,dtype=np.int64).copy();rank=rank_from_order(order)
    return (order,rank,degrees(g,rank),np.zeros(g.n,dtype=np.int64),np.zeros(g.m,dtype=np.int64),
            np.empty(g.n,dtype=np.int64),np.empty(g.m,dtype=np.int64))

@njit(cache=True)
def insertion_step(edges,offsets,inc,state,u,v,after,stamp,accept=True):
    order,rank,d,change,seen,affected,terms=state
    i=rank[u];rv=rank[v]
    affected[0]=u;na=1;nt=0
    for k in range(offsets[u],offsets[u+1]):
        e=inc[k];w=edges[e,0]+edges[e,1]-u
        old=i<rank[w]
        now=rank[w]>rv if after else rank[w]>=rv
        if old!=now:
            change[u]+=1 if now else -1
            change[w]+=-1 if now else 1
            affected[na]=w;na+=1
    for k in range(na):
        x=affected[k]
        for j in range(offsets[x],offsets[x+1]):
            e=inc[j]
            if seen[e]!=stamp:
                seen[e]=stamp;terms[nt]=e;nt+=1
    delta=0
    for k in range(nt):
        e=terms[k];a=edges[e,0];b=edges[e,1]
        delta+=min(d[a]+change[a],d[b]+change[b])-min(d[a],d[b])
    if accept and delta<0:
        dest=rv-(1 if i<rv else 0)+(1 if after else 0)
        if dest>i:
            for k in range(i,dest):order[k]=order[k+1];rank[order[k]]=k
        elif dest<i:
            for k in range(i,dest,-1):order[k]=order[k-1];rank[order[k]]=k
        order[dest]=u;rank[u]=dest
        for k in range(na):x=affected[k];d[x]+=change[x]
    for k in range(na):change[affected[k]]=0
    return delta

@njit(cache=True)
def insertion_block(edges,offsets,inc,state,edgechoices,sides,afters,stamp):
    total=0;accepted=0
    for k in range(len(edgechoices)):
        e=edgechoices[k];u=edges[e,sides[k]];v=edges[e,1-sides[k]]
        delta=insertion_step(edges,offsets,inc,state,u,v,afters[k],stamp+k)
        if delta<0:total+=delta;accepted+=1
    return total,accepted

def literal_move(order,u,v,after):
    seq=list(map(int,order));seq.remove(int(u));seq.insert(seq.index(int(v))+int(after),int(u))
    return np.array(seq,dtype=np.int64)

@njit(cache=True)
def all_order_works(edges,orders):
    out=np.empty(len(orders),dtype=np.int64);n=orders.shape[1]
    for j in range(len(orders)):
        rank=np.empty(n,dtype=np.int64);d=np.zeros(n,dtype=np.int64)
        for k in range(n):rank[orders[j,k]]=k
        for e in range(len(edges)):
            u,v=edges[e];d[u if rank[u]<rank[v] else v]+=1
        total=0
        for e in range(len(edges)):u,v=edges[e];total+=min(d[u],d[v])
        out[j]=total
    return out
