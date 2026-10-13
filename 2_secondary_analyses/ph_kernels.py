"""Post-hoc kernels (NOT pre-registered): greedy-colouring orders and a lean adaptive-forward triangle lister
for kernel timing. Everything else (exact work, tangent refinement, smallest-last, ILS) is imported unchanged
from the pc_eval package (core.py / methods.py).

Colouring orders (review concern C3; same construction as work/kclique/kc.py::color_order):
  process order  SL  = reverse of Matula-Beck smallest-last (pc_eval core.smallest_last)
                 DEG = descending degree, identifier ties
  greedy colouring in the process order (smallest colour not used by an already coloured neighbour)
  order          asc  = colour classes as blocks, colour 0 first;  desc = highest colour first
                 identifier ties inside a class; '-rev' = the reversed order.
A proper colouring has no edge inside a class, so '<v>-desc' and '<v>-asc-rev' induce the same orientation and
the same work (likewise '<v>-asc' and '<v>-desc-rev').

Timing lister: the loop of the artifact's cpp/exact_triangle_lister.cpp, ported to numba: forward and incoming
CSR lists sorted by (degree desc, id) (AOT local order), a marker array holding the forward set of u, and every
edge handled once at the endpoint with the larger forward set (ties by id), scanning the smaller forward set.
Probes = Lambda(order), triangles = |T(G)|. Only the listing loop is timed, as in the C++ program.
"""
import numpy as np
from numba import njit

import core  # pc_eval (imported unchanged)


# ------------------------------------------------------------------------------------------------ colouring
@njit(cache=True)
def greedy_colour(n, off, nbr, proc):
    col = np.full(n, -1, np.int64)
    maxdeg = 0
    for v in range(n):
        d = off[v + 1] - off[v]
        if d > maxdeg:
            maxdeg = d
    stamp = np.zeros(maxdeg + 2, np.int64)
    for i in range(proc.shape[0]):
        v = proc[i]
        for q in range(off[v], off[v + 1]):
            c = col[nbr[q]]
            if c >= 0:
                stamp[c] = i + 1
        c = 0
        while stamp[c] == i + 1:
            c += 1
        col[v] = c
    return col


def process_order(n, deg, noff, nbr, kind):
    if kind == 'SL':
        return core.smallest_last(n, noff, nbr)[::-1].copy()
    if kind == 'DEG':
        return np.lexsort((np.arange(n, dtype=np.int64), -deg)).astype(np.int64)
    raise ValueError(kind)


def colour_order(n, deg, noff, nbr, kind, direction):
    """Returns (order, number of colours)."""
    col = greedy_colour(n, noff, nbr, process_order(n, deg, noff, nbr, kind))
    k = int(col.max()) + 1 if n else 0
    key = col if direction == 'asc' else (k - 1) - col
    return core.order_by_key_then_id(np.ascontiguousarray(key, dtype=np.int64)), k


VARIANTS = [('SL', 'asc'), ('SL', 'desc'), ('DEG', 'asc'), ('DEG', 'desc')]
FIXED_VARIANT = 'DEG-desc'   # pre-declared in PREDECLARED_VARIANT.md (2026-09-28T08:04:00Z), before any PC result


def all_colour_orders(n, edges, noff=None, nbr=None):
    """The 8 colouring orders in listing order SL-asc, SL-asc-rev, SL-desc, SL-desc-rev, DEG-asc, ...
    Each variant is built from scratch and timed separately (construction time includes the neighbour CSR,
    the process order, the colouring and the sort; '-rev' adds the reversal copy).
    Returns {label: (order, seconds, colours)}."""
    from time import perf_counter
    out = {}
    for kind, direction in VARIANTS:
        t = perf_counter()
        no, nb = core.neighbor_csr(n, edges) if noff is None else (noff, nbr)
        deg = np.diff(no)
        o, k = colour_order(n, deg, no, nb, kind, direction)
        sec = perf_counter() - t
        lab = f'{kind}-{direction}'
        out[lab] = (o, sec, k)
        t = perf_counter()
        r = o[::-1].copy()
        out[lab + '-rev'] = (r, sec + (perf_counter() - t), k)
    return out


# ------------------------------------------------------------------------------------------------ lister
@njit(cache=True)
def build_lister(n, E, order):
    """Forward/incoming CSR under `order`, each list sorted by (degree desc, id)."""
    m = E.shape[0] // 2
    rank = np.empty(n, np.int64)
    for i in range(n):
        rank[order[i]] = i
    deg = np.zeros(n, np.int64)
    for e in range(2 * m):
        deg[E[e]] += 1
    fo = np.zeros(n + 1, np.int64)
    io = np.zeros(n + 1, np.int64)
    for e in range(m):
        a = E[2 * e]
        b = E[2 * e + 1]
        if rank[a] < rank[b]:
            fo[a + 1] += 1
            io[b + 1] += 1
        else:
            fo[b + 1] += 1
            io[a + 1] += 1
    for i in range(n):
        fo[i + 1] += fo[i]
        io[i + 1] += io[i]
    # adjacency CSR (both directions)
    ao = np.zeros(n + 1, np.int64)
    for i in range(n):
        ao[i + 1] = ao[i] + deg[i]
    fill = ao[:n].copy()
    adj = np.empty(2 * m, np.int64)
    for e in range(m):
        a = E[2 * e]
        b = E[2 * e + 1]
        adj[fill[a]] = b
        fill[a] += 1
        adj[fill[b]] = a
        fill[b] += 1
    # local order: degree desc, id asc -> visit targets x in that order and append x to its neighbours' lists
    key = np.empty(n, np.int64)
    maxd = 0
    for i in range(n):
        if deg[i] > maxd:
            maxd = deg[i]
    for i in range(n):
        key[i] = maxd - deg[i]
    loc = core.order_by_key_then_id(key)
    fw = np.empty(m, np.int32)
    iw = np.empty(m, np.int32)
    ff = fo[:n].copy()
    fi = io[:n].copy()
    for j in range(n):
        x = loc[j]
        for q in range(ao[x], ao[x + 1]):
            u = adj[q]
            if rank[u] < rank[x]:      # u -> x : x in forward[u]
                fw[ff[u]] = x
                ff[u] += 1
            else:                      # x -> u : x in incoming[u]
                iw[fi[u]] = x
                fi[u] += 1
    return fo, fw, io, iw


@njit(cache=True)
def list_triangles(n, fo, fw, io, iw, mark):
    probes = np.int64(0)
    tri = np.int64(0)
    for u in range(n):
        for q in range(fo[u], fo[u + 1]):
            mark[fw[q]] = u
        du = fo[u + 1] - fo[u]
        for q in range(fo[u], fo[u + 1]):
            v = fw[q]
            dv = fo[v + 1] - fo[v]
            if dv < du or (dv == du and v < u):
                probes += dv
                for r in range(fo[v], fo[v + 1]):
                    if mark[fw[r]] == u:
                        tri += 1
        for q in range(io[u], io[u + 1]):
            x = iw[q]
            dx = fo[x + 1] - fo[x]
            if dx < du or (dx == du and x < u):
                probes += dx
                for r in range(fo[x], fo[x + 1]):
                    if mark[fw[r]] == u:
                        tri += 1
    return probes, tri


class Lister:
    def __init__(self, n, E, order):
        self.n = n
        self.fo, self.fw, self.io, self.iw = build_lister(n, E, np.ascontiguousarray(order, dtype=np.int64))
        self.mark = np.full(max(n, 1), -1, np.int64)

    def run(self, inner=1):
        """Times `inner` consecutive listings; returns (seconds per listing, probes, triangles)."""
        from time import perf_counter
        t = perf_counter()
        for _ in range(inner):
            p, tr = list_triangles(self.n, self.fo, self.fw, self.io, self.iw, self.mark)
        return (perf_counter() - t) / inner, int(p), int(tr)


def warmup():
    edges = np.array([[0, 1], [0, 2], [1, 2], [2, 3], [3, 4], [1, 4], [0, 4], [2, 4]], dtype=np.int64)
    n = 5
    E = edges.reshape(-1).copy()
    all_colour_orders(n, edges)
    L = Lister(n, E, np.arange(n, dtype=np.int64))
    L.run()
