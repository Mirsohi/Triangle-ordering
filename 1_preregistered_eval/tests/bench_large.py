"""Scale benchmark: runs the per-graph pipeline of run_eval.py on a synthetic Chung-Lu graph of held-out size
(text file -> artifact parser -> preprocessing -> R* -> T -> DOVE -> T+ILS (1 seed) -> verification) and reports
the time and peak memory of each step. Also times the C++ literal-rank ILS against the numba ILS on it.
Usage: python tests/bench_large.py <scratch dir> <n> <m> <gamma> [--no-cpp]
"""
import sys, os, time, json, resource, gc
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, '/home/claude/ideas/ref/code_data/DOVE_AISTATS_2027_code_and_data/src')
import core, graphs, methods  # noqa: E402


def peak_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def main():
    scratch = Path(sys.argv[1]); scratch.mkdir(parents=True, exist_ok=True)
    n, m, gamma = int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4])
    rep = dict(n_target=n, m_target=m, gamma=gamma)
    core.warmup()
    rng = np.random.default_rng(0)
    w = (np.arange(1, n + 1) ** (-1.0 / (gamma - 1)))
    p = w / w.sum()
    src = rng.choice(n, size=int(m * 1.15), p=p); dst = rng.choice(n, size=int(m * 1.15), p=p)
    ids = rng.permutation(n) * 3 + 11  # sparse original ids
    f = scratch / f'bench_{n}_{m}.txt'
    t = time.perf_counter()
    with open(f, 'w') as fh:
        fh.write('# synthetic\n')
        np.savetxt(fh, np.stack([ids[src], ids[dst]], 1), fmt='%d', delimiter='\t')
    rep['write_s'] = time.perf_counter() - t
    del src, dst; gc.collect()
    t = time.perf_counter(); edges, oid = graphs.load_graph_file(f, 'bench'); rep['parse_preprocess_s'] = time.perf_counter() - t
    t = time.perf_counter(); g = graphs.make_graphdata('bench', edges, oid); rep['graphdata_s'] = time.perf_counter() - t
    rep['n'], rep['m'] = g.n, g.m
    rep['max_degree'] = int(g.degree.max())
    print('graph', g.n, g.m, rep, flush=True)
    prep = methods.Prepared(g)
    t = time.perf_counter(); ref = methods.build_reference(g, prep, {}); rep['reference_s'] = time.perf_counter() - t
    rep['builtin_s'] = ref['builtin_seconds']; rep['R_label'] = ref['label']; rep['R_work'] = ref['work']
    print('R*', ref['label'], ref['work'], rep['reference_s'], ref['builtin_seconds'], f'peak {peak_mb():.0f} MB', flush=True)
    To, Tw, Ti = methods.run_T(prep, ref['order']); rep['T'] = dict(work=Tw, **Ti)
    print('T', Tw, Ti, flush=True)
    Do, Dw, Di = methods.run_dove(g, ref['order']); rep['DOVE'] = dict(work=Dw, **Di)
    print('DOVE', Dw, Di, f'peak {peak_mb():.0f} MB', flush=True)
    o, wk, info = methods.run_tils(g, ref['order'], Di['seconds'], 1); rep['TILS'] = dict(work=wk, info=info)
    print('TILS', wk, json.dumps(info)[:800], f'peak {peak_mb():.0f} MB', flush=True)
    p = scratch / 'tils.npy'; np.save(p, o.astype(np.int32))
    t = time.perf_counter(); v = methods.verify_order_file(g, prep, p, wk); rep['verify_s'] = time.perf_counter() - t
    rep['verify'] = v
    print('verify', v, rep['verify_s'], flush=True)
    # numba ILS vs C++ literal ranks: same stream, fixed proposal count, from R*
    if '--no-cpp' not in sys.argv:
        sys.path.insert(0, str(HERE))
        from test_ils_equivalence import cpp_lib, Cpp
        off, inc, E = core.csr(g.n, prep.edges)
        L = cpp_lib()
        cnt = 20000
        c = Cpp(L, g.n, off, inc, E, ref['order'], 5); s = core.ILSState(g.n, off, inc, E, ref['order'], 5, 1)
        t = time.perf_counter(); c.run(cnt, 1.0); tc = time.perf_counter() - t
        t = time.perf_counter(); s.run(cnt, 1.0); tn = time.perf_counter() - t
        same = c.work == s.work and np.array_equal(c.export(), s.export())
        c.close()
        rep['cpp_vs_numba'] = dict(proposals=cnt, cpp_s=tc, numba_s=tn, identical=bool(same),
                                   accepted=int(s.stats[core.ST_ACCEPTED]))
        print('cpp vs numba', rep['cpp_vs_numba'], flush=True)
    rep['peak_mb'] = peak_mb()
    (HERE / 'results').mkdir(exist_ok=True)
    (HERE / 'results' / f'bench_{n}_{m}_{gamma}.json').write_text(json.dumps(rep, indent=1, default=str))
    f.unlink()


if __name__ == '__main__':
    main()
