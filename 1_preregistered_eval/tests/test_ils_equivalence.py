"""Equivalence of the numba ILS (order-maintenance keys) with the tested C++ insert_ils.cpp (integer ranks).

Both are driven by the SAME SplitMix64 proposal stream (same seed), so equivalence means: identical work after
every proposal, identical orders (checked after every proposal in phase 1 and after every chunk/kick later), and
identical Algorithm-1 trajectories (incumbent, kicks, final order) when the C++ primitives are driven by a Python
implementation of Algorithm 1 lines 7-9.

The C++ source is compiled read-only from /home/claude/ideas/work/orientation/insert_ils.cpp into a scratch dir.
Run: python tests/test_ils_equivalence.py [--quick]   -> tests/results/ils_equivalence.json
"""
import sys, os, json, time, ctypes as ct, subprocess, tempfile, glob
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import core  # noqa: E402

CPP = Path('/home/claude/ideas/work/orientation/insert_ils.cpp')
CASES = Path('/home/claude/ideas/ref/research_20260927/run_20260927/cases')
PTR = ct.POINTER(ct.c_int64)


def P(a):
    assert a.dtype == np.int64 and a.flags.c_contiguous
    return a.ctypes.data_as(PTR)


def cpp_lib():
    build = Path(tempfile.gettempdir()) / 'pc_eval_cpp'
    build.mkdir(exist_ok=True)
    so = build / 'insert_ils.so'
    subprocess.run(['g++', '-std=c++17', '-O3', '-DNDEBUG', '-shared', '-fPIC', str(CPP), '-o', str(so)], check=True)
    L = ct.CDLL(str(so))
    L.ils_create.argtypes = [ct.c_int64, ct.c_int64, PTR, PTR, PTR, PTR, ct.c_int64]; L.ils_create.restype = ct.c_void_p
    L.ils_destroy.argtypes = [ct.c_void_p]; L.ils_destroy.restype = None
    L.ils_work.argtypes = [ct.c_void_p]; L.ils_work.restype = ct.c_int64
    L.ils_run.argtypes = [ct.c_void_p, ct.c_int64, ct.c_double, PTR]; L.ils_run.restype = ct.c_int64
    L.ils_kick.argtypes = [ct.c_void_p, ct.c_int64]; L.ils_kick.restype = None
    L.ils_reset.argtypes = [ct.c_void_p, PTR]; L.ils_reset.restype = None
    L.ils_export.argtypes = [ct.c_void_p, PTR]; L.ils_export.restype = None
    return L


class Cpp:
    def __init__(self, L, n, off, inc, E, order, seed):
        self.L, self.n = L, n
        self.keep = (E, off, inc, np.ascontiguousarray(order, dtype=np.int64))
        self.raw = L.ils_create(n, E.shape[0] // 2, *(P(a) for a in self.keep), int(seed))
        self.acc = ct.c_int64()

    def run(self, count, p):
        self.L.ils_run(self.raw, int(count), float(p), ct.byref(self.acc))
        return self.acc.value

    def kick(self, k): self.L.ils_kick(self.raw, int(k))
    def reset(self, order):
        o = np.ascontiguousarray(order, dtype=np.int64); self.L.ils_reset(self.raw, P(o))
    @property
    def work(self): return self.L.ils_work(self.raw)
    def export(self):
        o = np.empty(self.n, np.int64); self.L.ils_export(self.raw, P(o)); return o
    def close(self): self.L.ils_destroy(self.raw)


def load(name):
    z = np.load(CASES / f'{name}.npz')
    return int(z['n']), z['edges'].astype(np.int64), z['reference'].astype(np.int64)


def check_graph(L, name, quick):
    n, edges, ref = load(name)
    off, inc, E = core.csr(n, edges)
    rng = np.random.default_rng(12345)
    starts = {'reference': ref, 'random': rng.permutation(n).astype(np.int64)}
    rep = dict(graph=name, n=n, m=len(edges), per_proposal_checks=0, order_checks=0, chunk_checks=0, kick_checks=0,
               accepted_total=0, alg1_runs=0, alg1_kicks=0, alg1_incumbent_updates=0, mismatches=0, key_checks=0)
    n_single = 400 if quick else 2000
    n_chunks = 20 if quick else 120
    for sname, start in starts.items():
        for seed in (1, 2, 3):
            for p in (1.0, 0.0, 0.5):
                c = Cpp(L, n, off, inc, E, start, seed)
                s = core.ILSState(n, off, inc, E, start, seed, 1)
                try:
                    assert c.work == s.work
                    # phase 1: single proposals
                    every = 1 if n <= 3000 else 25
                    for t in range(n_single):
                        a = c.run(1, p); s.run(1, p)
                        rep['per_proposal_checks'] += 1
                        if c.work != s.work: rep['mismatches'] += 1
                        if t % every == 0:
                            rep['order_checks'] += 1
                            if not np.array_equal(c.export(), s.export()): rep['mismatches'] += 1
                    # phase 2: chunks and kicks
                    for t in range(n_chunks):
                        c.run(256, p); s.run(256, p)
                        rep['chunk_checks'] += 1
                        if c.work != s.work or not np.array_equal(c.export(), s.export()): rep['mismatches'] += 1
                        if t % 10 == 9:
                            c.kick(10); s.kick(10)
                            rep['kick_checks'] += 1
                            if c.work != s.work or not np.array_equal(c.export(), s.export()): rep['mismatches'] += 1
                    rep['key_checks'] += 1
                    if not s.check_keys(): rep['mismatches'] += 1
                    # independent recomputation of the maintained work
                    if core.work_of_order(n, E, s.export()) != s.work: rep['mismatches'] += 1
                    rep['accepted_total'] += int(s.stats[core.ST_ACCEPTED])
                finally:
                    c.close()
                # phase 3: Algorithm 1 (lines 7-9) with small S to force many restarts
                for S in (1, 3, 50):
                    k = 10
                    chunk = 256
                    nch = (30 if quick else 150) if S <= 3 else (60 if quick else 300)
                    c = Cpp(L, n, off, inc, E, start, seed)
                    s = core.ILSState(n, off, inc, E, start, seed, S * chunk + k + chunk)
                    try:
                        inc_w = c.work; inc_o = c.export(); cnt = 0; kicks = 0; upd = 0
                        s.meta[core.INCUMBENT] = s.work; s.meta[core.COUNTER] = 0
                        for t in range(nch):
                            c.run(chunk, p)
                            if c.work < inc_w:
                                inc_w = c.work; inc_o = c.export(); cnt = 0; upd += 1
                            else:
                                cnt += 1
                            if cnt == S:
                                c.reset(inc_o); c.kick(k); cnt = 0; kicks += 1
                        # numba in irregular batches (batching must not matter)
                        done = 0; b = 1
                        while done < nch:
                            q = min(b, nch - done); s.chunks(q, chunk, p, k, S); done += q; b = b * 2 % 7 + 1
                        rep['alg1_runs'] += 1; rep['alg1_kicks'] += kicks; rep['alg1_incumbent_updates'] += upd
                        ok = (c.work == s.work and np.array_equal(c.export(), s.export())
                              and int(s.meta[core.INCUMBENT]) == inc_w and int(s.meta[core.COUNTER]) == cnt
                              and int(s.stats[core.ST_KICKS]) == kicks and int(s.stats[core.ST_INCUMBENT_UPDATES]) == upd)
                        # restore the incumbent: undo log (numba) vs reset(order) (C++)
                        s.undo_all(); c.reset(inc_o)
                        ok = ok and s.work == inc_w and np.array_equal(s.export(), inc_o) and np.array_equal(c.export(), inc_o)
                        ok = ok and s.check_keys() and core.work_of_order(n, E, s.export()) == inc_w
                        if not ok:
                            rep['mismatches'] += 1
                            print('  ALG1 MISMATCH', name, sname, seed, p, S, flush=True)
                    finally:
                        c.close()
    return rep


def timing(L, name, seconds=3.0):
    """Proposals per second, same stream, plateau 1, from the reference order."""
    n, edges, ref = load(name)
    off, inc, E = core.csr(n, edges)
    out = {}
    for impl in ('cpp', 'numba'):
        st = Cpp(L, n, off, inc, E, ref, 1) if impl == 'cpp' else core.ILSState(n, off, inc, E, ref, 1, 1)
        cnt = 0; t0 = time.perf_counter()
        while time.perf_counter() - t0 < seconds:
            st.run(2560, 1.0); cnt += 2560
        dt = time.perf_counter() - t0
        out[impl] = dict(proposals=cnt, seconds=dt, per_second=cnt / dt, work=st.work)
        if impl == 'cpp': st.close()
    return out


def timing_fixed(L, name, count):
    """Same number of proposals (same stream) in both: equal final orders + wall time."""
    n, edges, ref = load(name)
    off, inc, E = core.csr(n, edges)
    c = Cpp(L, n, off, inc, E, ref, 7); s = core.ILSState(n, off, inc, E, ref, 7, 1)
    t = time.perf_counter(); c.run(count, 1.0); tc = time.perf_counter() - t
    t = time.perf_counter(); s.run(count, 1.0); tn = time.perf_counter() - t
    same = c.work == s.work and np.array_equal(c.export(), s.export())
    c.close()
    return dict(graph=name, proposals=count, cpp_s=tc, numba_s=tn, identical=bool(same), work=s.work)


def main():
    quick = '--quick' in sys.argv
    core.warmup()
    L = cpp_lib()
    names = sorted(Path(f).stem for f in glob.glob(str(CASES / '*.npz')))
    reps = []
    t0 = time.time()
    for name in names:
        r = check_graph(L, name, quick)
        reps.append(r)
        print(name, {k: r[k] for k in ('per_proposal_checks', 'order_checks', 'chunk_checks', 'kick_checks', 'alg1_runs', 'alg1_kicks', 'mismatches')}, f'{time.time()-t0:.0f}s', flush=True)
    tim = {name: timing(L, name) for name in ('Bitcoin-Alpha', 'P2P-Gnutella05', 'Loc-Brightkite', 'GitHub')}
    fixed = [timing_fixed(L, name, 500_000) for name in ('Bitcoin-Alpha', 'P2P-Gnutella05', 'Loc-Brightkite', 'GitHub')]
    for k, v in tim.items(): print('timing', k, {i: round(x['per_second']) for i, x in v.items()})
    for f in fixed: print('fixed', f)
    tot = {k: sum(r[k] for r in reps) for k in reps[0] if k not in ('graph', 'n', 'm')}
    out = dict(quick=quick, graphs=len(reps), totals=tot, per_graph=reps, timing_3s=tim, timing_fixed_500k=fixed,
               cpp_source=str(CPP))
    (HERE / 'results').mkdir(exist_ok=True)
    (HERE / 'results' / ('ils_equivalence_quick.json' if quick else 'ils_equivalence.json')).write_text(json.dumps(out, indent=1))
    print('TOTAL', tot)
    assert tot['mismatches'] == 0




def compressed_keys_test(L, names, proposals=3000):
    """Stress the order-maintenance relabelling: keys set to 1..n (no free gap anywhere), so every accepted move
    triggers a relabel. Orders/works must still equal the C++ ones after every chunk, kicks and Alg.-1 restarts."""
    out = dict(graphs=0, chunks=0, mismatches=0, relabels=0)
    for name in names:
        n, edges, ref = load(name)
        off, inc, E = core.csr(n, edges)
        for seed in (1, 2):
            c = Cpp(L, n, off, inc, E, ref, seed)
            s = core.ILSState(n, off, inc, E, ref, seed, 50 * 256 + 10 + 256)
            s.key[ref] = np.arange(1, n + 1)
            for t in range(max(1, proposals // 256)):
                s.key[s.export()] = np.arange(1, n + 1)   # re-compress before every chunk
                c.run(256, 1.0); s.run(256, 1.0)
                out['chunks'] += 1
                if c.work != s.work or not np.array_equal(c.export(), s.export()) or not s.check_keys():
                    out['mismatches'] += 1
                if t % 3 == 2:
                    c.kick(10); s.kick(10)
            # Algorithm 1 from the compressed state, S = 1 (restart after every non-improving chunk)
            inc_w = c.work; inc_o = c.export(); cnt = 0
            s.meta[core.INCUMBENT] = s.work; s.meta[core.COUNTER] = 0; s.meta[core.LOGLEN] = 0
            s.key[s.export()] = np.arange(1, n + 1)
            for t in range(20):
                c.run(256, 1.0)
                if c.work < inc_w: inc_w = c.work; inc_o = c.export(); cnt = 0
                else: cnt += 1
                if cnt == 1: c.reset(inc_o); c.kick(10); cnt = 0
            s.chunks(20, 256, 1.0, 10, 1)
            if c.work != s.work or not np.array_equal(c.export(), s.export()) or not s.check_keys():
                out['mismatches'] += 1
            s.undo_all(); c.reset(inc_o)
            if not np.array_equal(s.export(), inc_o) or s.work != inc_w:
                out['mismatches'] += 1
            out['relabels'] += int(s.meta[core.RELABELS])
            c.close()
        out['graphs'] += 1
    return out


if __name__ == '__main__' and '--compressed' in sys.argv:
    core.warmup(); L = cpp_lib()
    names = sorted(Path(f).stem for f in glob.glob(str(CASES / '*.npz')))
    r = compressed_keys_test(L, names)
    print(r)
    (HERE / 'results').mkdir(exist_ok=True)
    (HERE / 'results' / 'ils_compressed_keys.json').write_text(json.dumps(r, indent=1))
    assert r['mismatches'] == 0


if __name__ == '__main__' and '--compressed' not in sys.argv:
    main()
