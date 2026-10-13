"""Verifies the numba/numpy ports against the tested Python reference code on all 25 case graphs (+ random graphs):
  - smallest-last (core.smallest_last) vs /home/claude/ideas/work/fixes/reference.py::smallest_last
  - tangent refinement (core.tangent_refine) vs research_20260927/algorithms.py::tangent_phase (ties 'id'), which
    work/common/lib.py re-exports; single rounds vs work/common/lib.py::tangent_round
  - exact work (core.work_of_order) and the verification lister (probes, triangles) vs dove.orderings
  - the vectorised loader (graphs.preprocess / parse / make_graphdata) vs dove.graph.graph_from_edges/load_edgelist
Run: python tests/test_ports.py  -> tests/results/ports.json
"""
import sys, json, glob, time, tempfile, os
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, '/home/claude/ideas/work/common')
sys.path.insert(0, '/home/claude/ideas/work/fixes')
import lib  # noqa: E402  (sets up dove + algorithms imports read-only)
import reference as fixref  # noqa: E402
import core  # noqa: E402
import graphs  # noqa: E402
from dove.graph import graph_from_edges, load_edgelist  # noqa: E402
from dove.orderings import candidate_work, exact_triangle_count, degree_order, split_order, check_order  # noqa: E402
import networkx as nx  # noqa: E402

CASES = Path('/home/claude/ideas/ref/research_20260927/run_20260927/cases')


def tangent_old(g, order):
    """Frozen algorithms.tangent_phase with an unlimited Budget (ties 'id')."""
    w0 = candidate_work(g, order)
    b = lib.Budget(g, np.asarray(order, np.int64), w0, None)
    steps = lib.tangent_phase(g, b, {'ties': 'id'}, float('inf'))
    return np.asarray(b.order, np.int64), int(b.work), steps


def random_graphs():
    out = []
    rng = np.random.default_rng(5)
    for i in range(60):
        n = int(rng.integers(5, 400))
        kind = i % 4
        if kind == 0: G = nx.gnp_random_graph(n, float(rng.uniform(0.01, 0.3)), seed=i)
        elif kind == 1: G = nx.barabasi_albert_graph(n, int(rng.integers(1, 5)), seed=i)
        elif kind == 2: G = nx.powerlaw_cluster_graph(n, int(rng.integers(1, 5)), 0.3, seed=i)
        else: G = nx.random_regular_graph(4, n + (n % 2), seed=i)
        e = np.array(G.edges(), dtype=np.int64).reshape(-1, 2)
        if len(e) == 0: continue
        edges, ids = graphs.preprocess(e[:, 0], e[:, 1])
        out.append((f'rand{i}', graphs.make_graphdata(f'rand{i}', edges, ids), G))
    return out


def main():
    core.warmup()
    rep = dict(smallest_last=[0, 0], tangent_refine=[0, 0], tangent_round=[0, 0], work=[0, 0], lister=[0, 0],
               loader_arrays=[0, 0], loader_files=[0, 0], graphdata=[0, 0])  # [checks, mismatches]
    def chk(key, ok):
        rep[key][0] += 1
        if not ok:
            rep[key][1] += 1
            print('MISMATCH', key, flush=True)
    items = []
    for f in sorted(glob.glob(str(CASES / '*.npz'))):
        name = Path(f).stem
        g, ref = lib.load_case(name)
        items.append((name, g, ref))
    for name, g, G in random_graphs():
        items.append((name, g, None))
    t0 = time.time()
    for name, g, ref in items:
        edges = np.ascontiguousarray(g.edges, dtype=np.int64)
        E = edges.reshape(-1).copy()
        noff, nbr = core.neighbor_csr(g.n, g.edges)
        # smallest-last
        chk('smallest_last', np.array_equal(core.smallest_last(g.n, noff, nbr), fixref.smallest_last(g)))
        rng = np.random.default_rng(len(name))
        starts = [np.arange(g.n, dtype=np.int64), rng.permutation(g.n).astype(np.int64),
                  np.asarray(degree_order(g), np.int64), np.asarray(split_order(g), np.int64),
                  np.asarray(check_order(g), np.int64)]
        if ref is not None:
            starts.append(ref)
            rf = Path('/home/claude/ideas/work/fixes/refs') / f'{name}.npz'
            if rf.exists():
                starts.append(np.load(rf)['order'].astype(np.int64))
        for o in starts:
            # tangent refinement
            a1, w1, s1 = tangent_old(g, o)
            a2, w2, info = core.tangent_refine(g.n, E, o, 100)
            chk('tangent_refine', np.array_equal(a1, a2) and w1 == w2 and s1 == info['rounds'])
            # single round vs lib.tangent_round
            x = core.outdeg_from_rank(g.n, E, core.rank_of(o))
            a, nd, w = core.tangent_round(g.n, E, x)
            q = core.order_by_key_then_id(a)
            chk('tangent_round', np.array_equal(q, lib.tangent_round(g, x)) and int(w) == candidate_work(g, q))
            # work + lister
            wd = candidate_work(g, o)
            chk('work', core.work_of_order(g.n, E, o) == wd)
            probes, tri = core.adaptive_forward_lister(g.n, E, o)
            tri_ref = exact_triangle_count(g, o) if g.m <= 60000 else None
            if tri_ref is None:
                # networkx on the dense graph
                H = nx.Graph(); H.add_nodes_from(range(g.n)); H.add_edges_from(edges.tolist())
                tri_ref = sum(nx.triangles(H).values()) // 3
            chk('lister', int(probes) == wd and int(tri) == tri_ref)
        # loader on the same edge set, scrambled: sparse large ids, duplicates, reversals, self-loops
        ids = np.sort(rng.choice(10 ** 9, size=g.n, replace=False)) if g.n < 10 ** 6 else np.arange(g.n)
        perm_ids = rng.permutation(ids)
        u = perm_ids[edges[:, 0]]; v = perm_ids[edges[:, 1]]
        dup = rng.integers(0, len(u), size=max(1, len(u) // 5))
        U = np.concatenate([u, v[dup], u[:3], [7, 7, 123456789]])
        V = np.concatenate([v, u[dup], u[:3], [7, 7, 123456789]])  # reversals, dups, self-loops, isolated self-loop ids
        sh = rng.permutation(len(U)); U, V = U[sh], V[sh]
        e_new, id_new = graphs.preprocess(U, V)
        gd = graph_from_edges(zip(U.tolist(), V.tolist()), 'x')
        chk('loader_arrays', np.array_equal(e_new, gd.edges) and e_new.dtype == gd.edges.dtype
            and np.array_equal(id_new, gd.original_ids))
        gm = graphs.make_graphdata('x', e_new, id_new)
        chk('graphdata', gm.n == gd.n and all(np.array_equal(a, b) and a.dtype == b.dtype for a, b in zip(gm.adjacency, gd.adjacency)))
        # file-level: messy text formats
        if g.m <= 300000:
            with tempfile.TemporaryDirectory() as td:
                for style in range(4):
                    p = Path(td) / f's{style}.txt'
                    with open(p, 'w', encoding='utf-8') as fh:
                        if style == 0: fh.write('# comment\n% other comment\n\n')
                        if style == 1: fh.write('node_1,node_2\n')
                        if style == 3: fh.write('FromNodeId\tToNodeId\n# x\n')
                        for a_, b_ in zip(U.tolist(), V.tolist()):
                            if style == 0: fh.write(f'{a_} {b_}\n')
                            elif style == 1: fh.write(f'{a_},{b_}\n')
                            elif style == 2: fh.write(f'{a_}\t{b_}\t{int(rng.integers(10**9))}\n')
                            else: fh.write(f'  {a_}\t {b_} \n')
                        fh.write('bad line\n1\n')
                    e1, i1 = graphs.load_graph_file(p, 'x')
                    gl = load_edgelist(p)
                    chk('loader_files', np.array_equal(e1, gl.edges) and np.array_equal(i1, gl.original_ids))
        print(name, g.n, g.m, {k: v for k, v in rep.items()}, f'{time.time() - t0:.0f}s', flush=True)
    (HERE / 'results').mkdir(exist_ok=True)
    out = dict(graphs=len(items), case_graphs=25, checks={k: dict(checks=v[0], mismatches=v[1]) for k, v in rep.items()})
    (HERE / 'results' / 'ports.json').write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    assert all(v[1] == 0 for v in rep.values())


if __name__ == '__main__':
    main()
