"""Methods of the pre-registered protocol: reference R*, T, DOVE, T+ILS, and order verification."""
import glob
import os
import re
import subprocess
import time
from pathlib import Path
from time import perf_counter

import numpy as np

import core

CHUNK, P_PLATEAU, K_KICK, S_STALL = 256, 1.0, 10, 50
T_FRACTION = 0.10
VOLT_TIMEOUT = 4 * 3600  # seconds per Neigh run; a timeout counts as a failed seed
MAX_T_ROUNDS = 100


class Prepared:
    """Graph arrays shared by the untimed parts (reference building, T, verification)."""

    def __init__(self, g):
        self.g = g
        self.n = g.n
        self.m = g.m
        self.edges = np.ascontiguousarray(g.edges, dtype=np.int64)
        self.E = self.edges.reshape(-1).copy()

    def work(self, order):
        return core.work_of_order(self.n, self.E, order)


# ------------------------------------------------------------------------------------------------ Neigh ranks
def stored_neigh_ranks(artifact, name):
    """All stored VOLT Neigh rank files of the artifact for this graph name: {label: path}."""
    art = Path(artifact) / 'results'
    out = {}
    for d in ('final_development', 'final_validation', 'final_heldout'):
        for f in sorted(glob.glob(str(art / d / 'ranks' / f'{glob.escape(name)}.neigh.seed-*.rank'))):
            s = re.search(r'seed-(\d+)\.rank$', f).group(1)
            out[f'NeighStored{s}'] = Path(f)
    for f in sorted(glob.glob(str(art / 'revision20260914' / f'{glob.escape(name)}.neigh.*.rank'))):
        mm = re.search(r'\.neigh\.(\d+)\.rank$', f)
        if mm:
            out[f'NeighStored{mm.group(1)}'] = Path(f)
    return dict(sorted(out.items(), key=lambda kv: int(kv[0][len('NeighStored'):])))


def order_from_rank_file(path, n):
    rank = np.loadtxt(path, dtype=np.int64).reshape(-1)
    if rank.shape != (n,) or (n and (rank.min() != 0 or rank.max() != n - 1 or not np.all(np.bincount(rank, minlength=n) == 1))):
        raise ValueError(f'{path}: not a rank permutation of length {n}')
    return np.argsort(rank, kind='stable').astype(np.int64)


def _volt_launcher(volt):
    """volt is a path to an executable, or 'wsl:<linux path>' for a VOLT binary built inside WSL (the artifact's
    supported build route); in that case file arguments are translated with `wslpath`."""
    v = str(volt)
    if v.startswith('wsl:'):
        prog = v[4:]
        def conv(p):
            cp = subprocess.run(['wsl.exe', 'wslpath', '-a', str(p)], text=True, capture_output=True, check=True)
            return cp.stdout.strip()
        return ['wsl.exe', '-e', prog], conv
    return [v], str


def run_volt(volt, g, workdir, seeds=(1, 2, 3, 4, 5), threshold=0.01, log=print):
    """Neigh via VOLT exactly as the artifact's run_neigh(). Returns ({label: order}, records)."""
    prefix, conv = _volt_launcher(volt)
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    edge_path = workdir / f'{g.name}.edgelist'
    tmp = edge_path.with_name(edge_path.name + '.writing')
    with open(tmp, 'w', newline='\n') as f:
        np.savetxt(f, g.edges, fmt='%d')
    back = np.loadtxt(tmp, dtype=np.int64).reshape(-1, 2)
    if not np.array_equal(back, g.edges):
        raise RuntimeError('inconsistent VOLT input export')
    os.replace(tmp, edge_path)
    orders, recs = {}, []
    for s in seeds:
        rank_path = workdir / f'{g.name}.neigh.seed-{s}.rank'
        if rank_path.exists():
            rank_path.unlink()
        t = perf_counter()
        try:
            cmd = prefix + [conv(edge_path), '-o', 'neigh', '-n', str(threshold), '--seed', str(s),
                            '--rank-output', conv(rank_path), '--order-only', '-p', '1']
            cp = subprocess.run(cmd, text=True, capture_output=True, check=False, timeout=VOLT_TIMEOUT)
            sec = perf_counter() - t
            (workdir / f'{g.name}.neigh.seed-{s}.log').write_text('$ ' + ' '.join(cmd) + '\n\nSTDOUT\n' + cp.stdout + '\nSTDERR\n' + cp.stderr, encoding='utf-8')
            if cp.returncode:
                raise RuntimeError(f'exit code {cp.returncode}')
            orders[f'Neigh{s}'] = order_from_rank_file(rank_path, g.n)
            recs.append(dict(seed=s, ok=True, seconds=sec))
        except Exception as ex:  # noqa: BLE001
            recs.append(dict(seed=s, ok=False, error=repr(ex), seconds=perf_counter() - t))
            log(f'  VOLT seed {s} failed: {ex!r}')
    return orders, recs


# ------------------------------------------------------------------------------------------------ reference R*
def builtin_candidates(g, prep):
    """Pre-registered built-in candidates in listing order."""
    from dove.orderings import natural_order, random_order, degree_order, degeneracy_order, split_order, check_order
    noff, nbr = core.neighbor_csr(g.n, g.edges)
    c = {}
    timing = {}
    for label, fn in [('Natural', lambda: natural_order(g)), ('Random', lambda: random_order(g, 7)),
                      ('Degree', lambda: degree_order(g)), ('Core', lambda: degeneracy_order(g)),
                      ('SmallestLast', lambda: core.smallest_last(g.n, noff, nbr)),
                      ('Split', lambda: split_order(g)), ('Check', lambda: check_order(g))]:
        t = perf_counter()
        c[label] = np.asarray(fn(), dtype=np.int64)
        timing[label] = perf_counter() - t
    return c, timing


def build_reference(g, prep, extra):
    """extra: ordered dict of additional candidates (Neigh). Returns dict with R* and all candidate works."""
    from dove.orderings import candidate_work
    base, timing = builtin_candidates(g, prep)
    cands = dict(base)
    cands.update(extra)
    works = {}
    fwd = {}
    listing = []
    for label, o in cands.items():
        if not core.is_permutation(o):
            raise ValueError(f'candidate {label} is not a permutation')
        for lab, oo in ((label, o), (label + '-rev', o[::-1].copy())):
            w = int(candidate_work(g, oo))
            w2 = prep.work(oo)
            if w != w2:
                raise AssertionError(f'work mismatch for candidate {lab}: {w} vs {w2}')
            works[lab] = w
            listing.append((lab, oo))
        fwd[label] = works[label]
    best_label, best_order = min(listing, key=lambda t: (works[t[0]], [x[0] for x in listing].index(t[0])))
    lam_fwd = min(fwd.values())
    return dict(order=best_order, label=best_label, work=works[best_label], works=works, forward_min=lam_fwd,
                forward_min_label=min(fwd, key=lambda k: (fwd[k], list(fwd).index(k))), builtin_seconds=timing)


# ------------------------------------------------------------------------------------------------ T
def run_T(prep, order):
    t = perf_counter()
    o, w, info = core.tangent_refine(prep.n, prep.E, order, MAX_T_ROUNDS)
    info['seconds'] = perf_counter() - t
    return o, int(w), info


# ------------------------------------------------------------------------------------------------ DOVE
def run_dove(g, order):
    from dove.relaxation import optimize_order
    t = perf_counter()
    res = optimize_order(g, steps=200, learning_rate=0.08, initial_temperature=0.5, final_temperature=0.04,
                         eval_every=5, seed=7, initial_order=np.asarray(order, dtype=np.int64), initial_name='Rstar',
                         guard_with_builtins=False, smoothing_floor=0.01)
    sec = perf_counter() - t
    return np.asarray(res.order, dtype=np.int64), int(res.work), dict(seconds=sec, inner_seconds=res.seconds,
                                                                       initial_work=int(res.initial_work))


# ------------------------------------------------------------------------------------------------ T+ILS
def run_tils(g, order, budget, seed, p=P_PLATEAU, k=K_KICK, S=S_STALL, chunk=CHUNK):
    """Algorithm 1 with the pre-registered implementation details (PREREGISTRATION.md 3.8).
    Everything from building the arrays to returning the final order is charged."""
    t0 = perf_counter()
    B = float(budget)
    n = g.n
    edges = np.ascontiguousarray(g.edges, dtype=np.int64)
    off, inc, E = core.csr(n, edges)
    start = np.ascontiguousarray(order, dtype=np.int64)
    start_work = core.work_of_order(n, E, start)
    info = dict(budget=B, seed=seed, start_work=start_work)
    if start_work == 0:
        info.update(status='known_zero_optimum', total_seconds=perf_counter() - t0)
        return start, 0, info

    # 1. initial T phase (first round always; next only if predicted to end within 10% of B)
    def may_start(r, last):
        return (perf_counter() - t0) + last <= T_FRACTION * B
    tT = perf_counter()
    pi, piw, tinfo = core.tangent_refine(n, E, start, MAX_T_ROUNDS, may_start)
    t_T = perf_counter() - tT
    t_phase_end = perf_counter() - t0
    info['T_phase'] = dict(tinfo, seconds=t_T, seconds_since_start=t_phase_end, work=int(piw))
    # 2. reserve for the final pass (PREREGISTRATION.md Amendment 1): min(0.5 B, max(3 t1, t_T))
    r = min(0.5 * B, max(3.0 * tinfo['first_round_s'], t_T))
    info['reserve'] = r
    # 3. ILS
    st = core.ILSState(n, off, inc, E, pi, seed, S * chunk + k + chunk)
    assert st.work == piw
    st.meta[core.INCUMBENT] = st.work
    st.meta[core.COUNTER] = 0
    stop_at = B - r
    batch = 1
    target = max(1e-4, 0.01 * B)
    calls = 0
    while True:
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
    # 4. final T pass (charged)
    fo, fw, finfo = core.tangent_refine(n, E, inc_order, MAX_T_ROUNDS)
    total = perf_counter() - t0
    info.update(status='finished', ils=stats, ils_end=ils_end, ils_incumbent_work=int(inc_work),
                final_T=dict(finfo), total_seconds=total, overshoot=max(0.0, total - B))
    return np.asarray(fo, dtype=np.int64), int(fw), info


# ------------------------------------------------------------------------------------------------ verification
def verify_order_file(g, prep, path, claimed_work):
    from dove.orderings import candidate_work
    o = np.load(path).astype(np.int64)
    ok_perm = bool(core.is_permutation(o)) and o.shape == (g.n,)
    w_dove = int(candidate_work(g, o)) if ok_perm else -1
    probes, tri = core.adaptive_forward_lister(prep.n, prep.E, o) if ok_perm else (-1, -1)
    return dict(file=str(path), permutation=ok_perm, claimed=int(claimed_work), dove_work=w_dove,
                lister_probes=int(probes), triangles=int(tri),
                ok=ok_perm and w_dove == int(claimed_work) == int(probes))
