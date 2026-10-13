"""Post-hoc methods (NOT pre-registered): ILS-T and C+ILS-T.

ILS-T   = Algorithm 1 WITHOUT the initial tangent phase: ILS from the start order, then the final tangent pass.
          Identical ILS kernel, proposal stream, chunk/plateau/kick/stall parameters and batching as
          pc_eval methods.run_tils (steps 3-4 of PREREGISTRATION.md 3.8); the only differences are that step 1
          (initial T phase) is skipped and that the reserve r for the final pass is taken from the T+ILS run with
          the same seed on the same graph (result.json: TILS[seed].info.reserve), so that both variants stop ILS
          at the same B - r. If that value is unavailable, r = min(0.5 B, 3 t1) with t1 the measured time of one
          probe tangent round on the start order (charged, result discarded).
C+ILS-T = ILS-T whose start order is built inside the clock by `start_fn` (a colouring order), so the colouring
          construction (and, for the best-of-8 start, the construction and exact evaluation of all 8 orders) is
          charged to the same budget B. Unlike T+ILS, C+ILS-T is not guarded by R*: it can end above R*.
"""
from time import perf_counter

import numpy as np

import core      # pc_eval, unchanged
import methods   # pc_eval, unchanged


def run_ils_t(g, start_fn, budget, seed, reserve=None, p=methods.P_PLATEAU, k=methods.K_KICK, S=methods.S_STALL,
              chunk=methods.CHUNK):
    """start_fn() -> (start order, dict info) is called inside the clock. Returns (order, work, info)."""
    t0 = perf_counter()
    B = float(budget)
    n = g.n
    edges = np.ascontiguousarray(g.edges, dtype=np.int64)
    off, inc, E = core.csr(n, edges)
    start, sinfo = start_fn()
    start = np.ascontiguousarray(start, dtype=np.int64)
    start_work = core.work_of_order(n, E, start)
    t_start = perf_counter() - t0
    info = dict(budget=B, seed=seed, start_work=int(start_work), start=sinfo, start_seconds=t_start)
    if start_work == 0:
        info.update(status='known_zero_optimum', total_seconds=perf_counter() - t0, overshoot=0.0, reserve=0.0)
        return start, 0, info
    if reserve is None:
        tp = perf_counter()
        d = core.outdeg_from_rank(n, E, core.rank_of(start))
        core.tangent_round(n, E, d)
        t1 = perf_counter() - tp
        r = min(0.5 * B, 3.0 * t1)
        info['reserve_rule'] = 'probe: min(0.5B, 3 t1)'
        info['probe_round_s'] = t1
    else:
        r = float(reserve)
        info['reserve_rule'] = 'same as T+ILS (result.json)'
    info['reserve'] = r
    st = core.ILSState(n, off, inc, E, start, seed, S * chunk + k + chunk)
    assert st.work == start_work
    st.meta[core.INCUMBENT] = st.work
    st.meta[core.COUNTER] = 0
    stop_at = B - r
    batch = 1
    target = max(1e-4, 0.01 * B)
    calls = 0
    while True:   # same batching as methods.run_tils
        now = perf_counter() - t0
        if now >= stop_at:
            break
        tb = perf_counter()
        st.chunks(batch, chunk, p, k, S)
        dt = perf_counter() - tb
        calls += 1
        per = dt / batch
        remaining = stop_at - (perf_counter() - t0)
        nb = int(target / per) if per > 0 else batch * 2
        nb = max(1, min(nb, 2 * batch))
        if per > 0 and remaining > 0:
            nb = max(1, min(nb, int(remaining / per) if remaining / per >= 1 else 1))
        batch = nb
    ils_end = perf_counter() - t0
    undone = st.undo_all()
    if st.work != int(st.meta[core.INCUMBENT]):
        raise AssertionError('incumbent restoration failed')
    inc_order = st.export()
    inc_work = st.work
    stats = {name: int(st.stats[i]) for i, name in enumerate(
        ['proposals', 'accepted', 'improving', 'kicks', 'chunks', 'undone_moves', 'incumbent_updates'])}
    stats.update(relabels=int(st.meta[core.RELABELS]), relabel_work=int(st.meta[core.RELABEL_WORK]), calls=calls,
                 final_undo=undone)
    del st
    fo, fw, finfo = core.tangent_refine(n, E, inc_order, methods.MAX_T_ROUNDS)
    total = perf_counter() - t0
    info.update(status='finished', ils=stats, ils_end=ils_end, ils_incumbent_work=int(inc_work),
                final_T=dict(finfo), total_seconds=total, overshoot=max(0.0, total - B))
    return np.asarray(fo, dtype=np.int64), int(fw), info


def fixed_colour_start(g, label):
    """start_fn for C+ILS-T from one fixed colouring order (label e.g. 'DEG-desc')."""
    import ph_kernels as K

    def fn():
        kind, direction = label.split('-')[:2]
        edges = np.ascontiguousarray(g.edges, dtype=np.int64)
        noff, nbr = core.neighbor_csr(g.n, edges)
        o, ncol = K.colour_order(g.n, np.diff(noff), noff, nbr, kind, direction)
        if label.endswith('-rev'):
            o = o[::-1].copy()
        return o, dict(kind='fixed', label=label, colours=ncol)
    return fn


def best_colour_start(g):
    """start_fn for C+ILS-T from the lowest-work of the 8 colouring orders (ties: listing order); all 8
    constructions and exact evaluations are charged."""
    import ph_kernels as K

    def fn():
        edges = np.ascontiguousarray(g.edges, dtype=np.int64)
        E = edges.reshape(-1)
        cands = K.all_colour_orders(g.n, edges)
        works = {lab: core.work_of_order(g.n, E, o) for lab, (o, _, _) in cands.items()}
        best = min(works, key=lambda lab: (works[lab], list(works).index(lab)))
        return cands[best][0], dict(kind='best_of_8', label=best, works=works)
    return fn
