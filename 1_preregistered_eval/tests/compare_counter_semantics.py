"""Informative (not part of the protocol): Algorithm 1 as printed in the paper (stall counter c reset only when the
incumbent improves) versus the historical driver used for tuning, orient.py::ils_phase (counter reset whenever the
run-local best since the last restart improves). Same move engine, same seeds, same time per run, no T phase, no
final pass (pure ILS phase from the strengthened reference). Run: python tests/compare_counter_semantics.py
"""
import sys, json, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import core  # noqa: E402

CASES = Path('/home/claude/ideas/ref/research_20260927/run_20260927/cases')
REFS = Path('/home/claude/ideas/work/fixes/refs')
GRAPHS = ['ba-0', 'core_periphery-0', 'er-0', 'powerlaw_cluster-0', 'sbm-0', 'ws-0',
          'ba-1', 'core_periphery-1', 'er-1', 'powerlaw_cluster-1', 'sbm-1', 'ws-1',
          'Bitcoin-Alpha', 'P2P-Gnutella05', 'Loc-Brightkite', 'GitHub']


def run(n, off, inc, E, start, seed, seconds, semantics, p=1.0, k=10, S=50, chunk=256):
    st = core.ILSState(n, off, inc, E, start, seed, 5_000_000)  # historical semantics has no log bound
    inc_w = st.work; run_best = st.work; c = 0
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        st.run(chunk, p, logging=True)
        w = st.work
        improved = w < inc_w
        if improved:
            inc_w = w; st.meta[core.LOGLEN] = 0
        if semantics == 'paper':
            c = 0 if improved else c + 1
        else:  # historical: stall resets on improvement of the best since the last restart
            if w < run_best:
                run_best = w; c = 0
            else:
                c += 1
        if c >= S:
            st.undo_all(); st.kick(k, logging=True); run_best = st.work; c = 0
    st.undo_all()
    assert st.work == inc_w
    return inc_w


def main():
    core.warmup()
    rows = []
    for name in GRAPHS:
        z = np.load(CASES / f'{name}.npz'); n = int(z['n'])
        off, inc, E = core.csr(n, z['edges'].astype(np.int64))
        start = np.load(REFS / f'{name}.npz')['order'].astype(np.int64)
        R = core.work_of_order(n, E, start)
        sec = 1.0 if name[0].islower() else 3.0
        for seed in (1, 2, 3):
            for sem in ('paper', 'historical'):
                w = run(n, off, inc, E, start, seed, sec, sem)
                rows.append(dict(graph=name, seed=seed, semantics=sem, seconds=sec, R=R, work=w,
                                 gain=100 * (R - w) / R if R else 0.0))
        g = [r for r in rows if r['graph'] == name]
        print(name, {s: round(np.mean([r['gain'] for r in g if r['semantics'] == s]), 3) for s in ('paper', 'historical')}, flush=True)
    summ = {}
    for grp, names in (('synthetic dev (12)', GRAPHS[:12]), ('public (4)', GRAPHS[12:])):
        d = {s: float(np.mean([r['gain'] for r in rows if r['graph'] in names and r['semantics'] == s])) for s in ('paper', 'historical')}
        wins = sum(1 for nm in names if np.mean([r['work'] for r in rows if r['graph'] == nm and r['semantics'] == 'paper'])
                   < np.mean([r['work'] for r in rows if r['graph'] == nm and r['semantics'] == 'historical']))
        ties = sum(1 for nm in names if np.mean([r['work'] for r in rows if r['graph'] == nm and r['semantics'] == 'paper'])
                   == np.mean([r['work'] for r in rows if r['graph'] == nm and r['semantics'] == 'historical']))
        summ[grp] = dict(mean_gain=d, paper_better_graphs=wins, ties=ties, graphs=len(names))
    print(json.dumps(summ, indent=1))
    (HERE / 'results').mkdir(exist_ok=True)
    (HERE / 'results' / 'counter_semantics.json').write_text(json.dumps(dict(summary=summ, rows=rows), indent=1))


if __name__ == '__main__':
    main()
