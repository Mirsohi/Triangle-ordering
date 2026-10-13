"""Unit tests of ph_kernels against independent references (run in the sandbox):
 - greedy_colour == work/kclique/kc.py::greedy_colors (pure Python) for the same process order;
 - DEG orders == kc.color_order(g, 'degree', dir) exactly; SL process order == reverse of pc_eval smallest_last;
 - colourings are proper; reversal pairs have equal work;
 - Lister probes == Lambda (pc_eval core and dove.candidate_work) and triangles == pc_eval verification lister,
   on random graphs x random orders + the 25 case graphs.
python3 tests/test_kernels.py  -> tests/test_kernels.json
"""
import os, sys, json, glob
sys.dont_write_bytecode = True
os.environ.setdefault('NUMBA_CACHE_DIR', '/home/claude/review1/posthoc/.numba_cache')
sys.path.insert(0, '/home/claude/pc_eval'); sys.path.insert(0, '/home/claude/review1/posthoc')
sys.path.insert(0, '/home/claude/ideas/ref/code_data/DOVE_AISTATS_2027_code_and_data/src')
sys.path.insert(0, '/home/claude/ideas/work/kclique'); sys.path.insert(0, '/home/claude/ideas/work/common')
import numpy as np
import networkx as nx
import core, graphs
import ph_kernels as K
from dove.orderings import candidate_work
import kc

rng = np.random.default_rng(5)
stats = dict(graphs=0, colour_checks=0, lister_checks=0, mismatches=[])


def check_graph(name, g):
    stats['graphs'] += 1
    edges = np.ascontiguousarray(g.edges, dtype=np.int64); E = edges.reshape(-1).copy(); n = g.n
    noff, nbr = core.neighbor_csr(n, edges); deg = np.diff(noff)
    tri_ref = None
    for kind in ('SL', 'DEG'):
        proc = K.process_order(n, deg, noff, nbr, kind)
        c1 = K.greedy_colour(n, noff, nbr, proc)
        c2 = kc.greedy_colors(g, proc)
        stats['colour_checks'] += 1
        if not np.array_equal(c1, c2): stats['mismatches'].append((name, kind, 'colour'))
        if np.any(c1[edges[:, 0]] == c1[edges[:, 1]]): stats['mismatches'].append((name, kind, 'improper'))
        for d in ('asc', 'desc'):
            o, k = K.colour_order(n, deg, noff, nbr, kind, d)
            if kind == 'DEG':
                o2, k2 = kc.color_order(g, 'degree', d)
                if not (np.array_equal(o, o2) and k == k2): stats['mismatches'].append((name, kind, d, 'order'))
    cols = K.all_colour_orders(n, edges)
    W = {lab: core.work_of_order(n, E, o) for lab, (o, _, _) in cols.items()}
    for v in ('SL', 'DEG'):
        if not (W[f'{v}-desc'] == W[f'{v}-asc-rev'] and W[f'{v}-asc'] == W[f'{v}-desc-rev']):
            stats['mismatches'].append((name, v, 'reversal pair'))
    orders = [o for (o, _, _) in cols.values()] + [rng.permutation(n).astype(np.int64) for _ in range(3)]
    for o in orders:
        L = K.Lister(n, E, o); _, p, t = L.run()
        w = core.work_of_order(n, E, o); w2 = int(candidate_work(g, o)); p3, t3 = core.adaptive_forward_lister(n, E, o)
        stats['lister_checks'] += 1
        if not (p == w == w2 == p3 and t == t3): stats['mismatches'].append((name, 'lister', p, w, w2, p3, t, t3))
        if tri_ref is None: tri_ref = t
        elif t != tri_ref: stats['mismatches'].append((name, 'triangle count varies'))


for i in range(40):
    kind = i % 4
    nn = int(rng.integers(20, 400))
    G = [lambda: nx.gnm_random_graph(nn, int(nn * rng.uniform(1, 6)), seed=i),
         lambda: nx.barabasi_albert_graph(nn, int(rng.integers(1, 6)), seed=i),
         lambda: nx.powerlaw_cluster_graph(nn, 3, 0.3, seed=i),
         lambda: nx.complete_bipartite_graph(nn // 3, nn // 4)][kind]()
    e = np.array(G.edges(), dtype=np.int64)
    if len(e) == 0: continue
    ed, ids = graphs.preprocess(e[:, 0], e[:, 1])
    check_graph(f'rand{i}', graphs.make_graphdata(f'rand{i}', ed, ids))
for f in sorted(glob.glob('/home/claude/ideas/ref/research_20260927/run_20260927/cases/*.npz')):
    z = np.load(f); n = int(z['n']); e = z['edges'].astype(np.int64)
    a, b = np.minimum(e[:, 0], e[:, 1]), np.maximum(e[:, 0], e[:, 1])
    check_graph(os.path.basename(f), graphs.make_graphdata('x', np.stack([a, b], 1).astype(np.int32), np.arange(n)))
print(json.dumps({k: (v if k != 'mismatches' else v[:20]) for k, v in stats.items()}, indent=1))
stats['n_mismatches'] = len(stats['mismatches'])
json.dump(stats, open('/home/claude/review1/posthoc/tests/test_kernels.json', 'w'), indent=1, default=str)
