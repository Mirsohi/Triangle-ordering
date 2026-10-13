"""Compiled exact-work refinements, independently checked on return."""
import ctypes as ct
import os
from pathlib import Path
import shutil
import subprocess
import sys
from time import perf_counter
import numpy as np
from .orderings import normalized_scores, candidate_work, order_from_scores
ROOT=Path(__file__).resolve().parents[2]
def library():
    source=ROOT/'cpp'/'exact_refine.cpp'
    dest=ROOT/'build'/('exact_refine.dll' if os.name=='nt' else 'libexact_refine.so')
    dest.parent.mkdir(exist_ok=True)
    if not dest.exists() or dest.stat().st_mtime<source.stat().st_mtime:
        compiler=os.environ.get('DOVE_CXX') or shutil.which('g++') or shutil.which('clang++')
        if compiler is None and os.name=='nt':
            compiler=shutil.which('cl')
        if compiler is None:
            raise RuntimeError('No C++ compiler found. Install g++/clang++, or set DOVE_CXX.')
        if Path(compiler).name.lower() in {'cl','cl.exe'}:
            subprocess.run([compiler,'/nologo','/std:c++17','/EHsc','/LD','/O2',str(source),f'/Fe:{dest}',
                            '/link','/EXPORT:dove_refine','/EXPORT:dove_adjacent'],check=True)
        else:
            cmd=[compiler,'-O3','-std=c++17','-shared']
            if os.name!='nt': cmd.append('-fPIC')
            subprocess.run([*cmd,str(source),'-o',str(dest)],check=True)
    lib=ct.CDLL(str(dest));i64=ct.c_int64;ptr=ct.c_void_p
    lib.dove_refine.argtypes=[i64,i64,ptr,ptr,ct.c_uint64,ct.c_double,i64,ptr]
    lib.dove_adjacent.argtypes=[i64,i64,ptr,ptr,i64,ptr]
    lib.dove_refine.restype=lib.dove_adjacent.restype=i64
    return lib
def refine(graph,order,seed=1,seconds=5.0,max_proposals=200000,adjacent=False):
    lib=library();edges=np.ascontiguousarray(graph.edges,dtype=np.int32)
    info=np.zeros(2,dtype=np.int64);initial=candidate_work(graph,order)
    start=perf_counter()
    if adjacent:
        out=np.ascontiguousarray(order,dtype=np.int64).copy()
        work=lib.dove_adjacent(graph.n,graph.m,edges.ctypes.data,out.ctypes.data,10,info.ctypes.data)
    else:
        scores=normalized_scores(order)
        work=lib.dove_refine(graph.n,graph.m,edges.ctypes.data,scores.ctypes.data,seed,seconds,max_proposals,info.ctypes.data)
        out=order_from_scores(scores)
    elapsed=perf_counter()-start
    assert candidate_work(graph,out)==work and work<=initial
    return out,dict(work=int(work),seconds=elapsed,proposals=int(info[0]) if not adjacent else 0,
                    accepted=int(info[0] if adjacent else info[1]),improving_adjacent=int(info[1]) if adjacent else None)
