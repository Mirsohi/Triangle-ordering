"""Numba kernels for pc_eval: exact work, tangent refinement, smallest-last, ILS with order-maintenance keys,
and an independent adaptive-forward triangle lister used for verification.

Move semantics of the ILS are those of /home/claude/ideas/work/orientation/insert_ils.cpp (tested C++):
"remove u and reinsert it immediately before/after v", exact delta from the edges incident to u and to the
vertices whose forward degree changes. The C++ code shifts integer ranks (O(n) per accepted move); here the
order is a doubly linked list with int64 order-maintenance keys (local relabelling when a gap is exhausted),
so a move costs O(deg u + sum of degrees of vertices whose forward degree changes) plus amortised relabelling.
The random stream is the C++ SplitMix64 stream bit for bit, so identical seeds give identical proposals.
"""
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ.setdefault(_k, '1')
import numpy as np
from numba import njit

GOLD = np.uint64(0x9E3779B97F4A7C15)
MIX1 = np.uint64(0xBF58476D1CE4E5B9)
MIX2 = np.uint64(0x94D049BB133111EB)
U777 = np.uint64(777)
S30 = np.uint64(30)
S27 = np.uint64(27)
S31 = np.uint64(31)
S11 = np.uint64(11)
INV53 = 1.0 / 9007199254740992.0
TOP = np.int64(1) << np.int64(62)
MINGAP = np.int64(1) << np.int64(16)

# meta layout of the ILS state
HEAD, TAIL, WORK, STAMP, LOGLEN, RELABELS, RELABEL_WORK, COUNTER, INCUMBENT = range(9)
META_SIZE = 9
# stats layout
ST_PROPOSALS, ST_ACCEPTED, ST_IMPROVING, ST_KICKS, ST_CHUNKS, ST_UNDONE, ST_INCUMBENT_UPDATES = range(7)
STATS_SIZE = 7


# ------------------------------------------------------------------------------------------------ graph arrays
def csr(n, edges):
    """edges (m,2) -> (off (n+1) int64, inc (2m) int64 edge ids grouped by vertex, E flat (2m) int64)."""
    e64 = np.ascontiguousarray(edges, dtype=np.int64)
    m = e64.shape[0]
    verts = e64.T.reshape(-1)  # u's then v's
    eids = np.concatenate([np.arange(m, dtype=np.int64), np.arange(m, dtype=np.int64)])
    ix = np.argsort(verts, kind='stable')
    off = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(np.bincount(verts, minlength=n), out=off[1:])
    return off, eids[ix].copy(), e64.reshape(-1).copy()


@njit(cache=True)
def rank_of(order):
    n = order.shape[0]
    r = np.empty(n, np.int64)
    for i in range(n):
        r[order[i]] = i
    return r


@njit(cache=True)
def is_permutation(order):
    n = order.shape[0]
    seen = np.zeros(n, np.uint8)
    for i in range(n):
        x = order[i]
        if x < 0 or x >= n or seen[x]:
            return False
        seen[x] = 1
    return True


@njit(cache=True)
def outdeg_from_rank(n, E, rank):
    d = np.zeros(n, np.int64)
    m = E.shape[0] // 2
    for e in range(m):
        a = E[2 * e]
        b = E[2 * e + 1]
        if rank[a] < rank[b]:
            d[a] += 1
        else:
            d[b] += 1
    return d


@njit(cache=True)
def work_of_degrees(E, d):
    m = E.shape[0] // 2
    w = np.int64(0)
    for e in range(m):
        a = d[E[2 * e]]
        b = d[E[2 * e + 1]]
        w += a if a < b else b
    return w


def work_of_order(n, E, order):
    order = np.ascontiguousarray(order, dtype=np.int64)
    return int(work_of_degrees(E, outdeg_from_rank(n, E, rank_of(order))))


# ------------------------------------------------------------------------------------------------ tangent
@njit(cache=True)
def tangent_round(n, E, d):
    """One round: a = 2 x canonical F0 supergradient at d; orientation by ascending (a, id).
    Returns (a, new forward degrees, new work)."""
    m = E.shape[0] // 2
    a = np.zeros(n, np.int64)
    for e in range(m):
        u = E[2 * e]
        v = E[2 * e + 1]
        if d[u] < d[v]:
            a[u] += 2
        elif d[u] > d[v]:
            a[v] += 2
        else:
            a[u] += 1
            a[v] += 1
    nd = np.zeros(n, np.int64)
    for e in range(m):
        u = E[2 * e]
        v = E[2 * e + 1]
        if a[u] < a[v] or (a[u] == a[v] and u < v):
            nd[u] += 1
        else:
            nd[v] += 1
    w = work_of_degrees(E, nd)
    return a, nd, w


@njit(cache=True)
def order_by_key_then_id(a):
    """Stable counting sort of vertex ids by nonnegative integer key a: equals np.lexsort((arange(n), a))."""
    n = a.shape[0]
    if n == 0:
        return np.empty(0, np.int64)
    mx = 0
    for i in range(n):
        if a[i] > mx:
            mx = a[i]
    cnt = np.zeros(mx + 2, np.int64)
    for i in range(n):
        cnt[a[i] + 1] += 1
    for k in range(1, mx + 2):
        cnt[k] += cnt[k - 1]
    out = np.empty(n, np.int64)
    for i in range(n):
        out[cnt[a[i]]] = i
        cnt[a[i]] += 1
    return out


def tangent_refine(n, E, order, max_rounds=100, may_start=None):
    """Tangent refinement T (paper Sec. 3.1 / App. E): at most max_rounds rounds, identifier ties, stop at the
    first round without strict decrease. may_start(round_index, last_round_seconds) -> bool can veto starting a
    further round (used for the 10% rule of T+ILS; the first round is always run).
    Returns (order, work, info)."""
    from time import perf_counter
    order = np.ascontiguousarray(order, dtype=np.int64)
    d = outdeg_from_rank(n, E, rank_of(order))
    old = int(work_of_degrees(E, d))
    start_work = old
    best = order
    rounds = 0
    improving = 0
    first_round_s = None
    last_round_s = 0.0
    stopped = 'converged'
    for r in range(max_rounds):
        if r > 0 and may_start is not None and not may_start(r, last_round_s):
            stopped = 'time'
            break
        t = perf_counter()
        a, nd, w = tangent_round(n, E, d)
        rounds += 1
        w = int(w)
        if w > old:
            raise AssertionError('tangent refinement increased the work (violates Theorem 1)')
        if w == old:
            last_round_s = perf_counter() - t
            if first_round_s is None:
                first_round_s = last_round_s
            break
        best = order_by_key_then_id(a)
        improving += 1
        old = w
        d = nd
        last_round_s = perf_counter() - t
        if first_round_s is None:
            first_round_s = last_round_s
    else:
        stopped = 'max_rounds'
    return best, old, dict(rounds=rounds, improving_rounds=improving, start_work=start_work,
                           first_round_s=first_round_s or 0.0, stopped=stopped)


# ------------------------------------------------------------------------------------------------ smallest-last
@njit(cache=True)
def smallest_last(n, off, nbr):
    """Matula-Beck peeling: repeatedly remove the remaining vertex with minimum (current degree, id).
    Indexed binary heap keyed by cur[v]*n + v. Returns the removal order."""
    cur = np.empty(n, np.int64)
    for v in range(n):
        cur[v] = off[v + 1] - off[v]
    heap = np.empty(n, np.int64)
    pos = np.empty(n, np.int64)
    for v in range(n):
        heap[v] = v
        pos[v] = v
    size = n
    nn = np.int64(n)
    # heapify
    for i in range(size // 2 - 1, -1, -1):
        j = i
        while True:
            l = 2 * j + 1
            if l >= size:
                break
            c = l
            r = l + 1
            if r < size and cur[heap[r]] * nn + heap[r] < cur[heap[l]] * nn + heap[l]:
                c = r
            if cur[heap[c]] * nn + heap[c] < cur[heap[j]] * nn + heap[j]:
                t = heap[c]; heap[c] = heap[j]; heap[j] = t
                pos[heap[c]] = c; pos[heap[j]] = j
                j = c
            else:
                break
    removed = np.zeros(n, np.uint8)
    out = np.empty(n, np.int64)
    for k in range(n):
        u = heap[0]
        size -= 1
        if size > 0:
            heap[0] = heap[size]
            pos[heap[0]] = 0
            j = 0
            while True:
                l = 2 * j + 1
                if l >= size:
                    break
                c = l
                r = l + 1
                if r < size and cur[heap[r]] * nn + heap[r] < cur[heap[l]] * nn + heap[l]:
                    c = r
                if cur[heap[c]] * nn + heap[c] < cur[heap[j]] * nn + heap[j]:
                    t = heap[c]; heap[c] = heap[j]; heap[j] = t
                    pos[heap[c]] = c; pos[heap[j]] = j
                    j = c
                else:
                    break
        removed[u] = 1
        out[k] = u
        for q in range(off[u], off[u + 1]):
            w = nbr[q]
            if removed[w] == 0:
                cur[w] -= 1
                j = pos[w]
                while j > 0:
                    pj = (j - 1) // 2
                    if cur[heap[j]] * nn + heap[j] < cur[heap[pj]] * nn + heap[pj]:
                        t = heap[pj]; heap[pj] = heap[j]; heap[j] = t
                        pos[heap[pj]] = pj; pos[heap[j]] = j
                        j = pj
                    else:
                        break
    return out


def neighbor_csr(n, edges):
    """(off, nbr) with each row sorted ascending."""
    e = np.asarray(edges, dtype=np.int64)
    src = np.concatenate([e[:, 0], e[:, 1]])
    dst = np.concatenate([e[:, 1], e[:, 0]])
    ix = np.lexsort((dst, src))
    off = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(np.bincount(src, minlength=n), out=off[1:])
    return off, dst[ix].copy()


# ------------------------------------------------------------------------------------------------ RNG (SplitMix64 of insert_ils.cpp)
@njit(inline='always')
def rng_next(rs):
    s = rs[0] + GOLD
    rs[0] = s
    z = s
    z = (z ^ (z >> S30)) * MIX1
    z = (z ^ (z >> S27)) * MIX2
    return z ^ (z >> S31)


@njit(inline='always')
def rng_below(rs, k):
    return np.int64(rng_next(rs) % np.uint64(k))


@njit(inline='always')
def rng_unif(rs):
    return np.float64(rng_next(rs) >> S11) * INV53


def rng_state(seed):
    """C++: Rng(U seed) : s(seed * 0x9E3779B97F4A7C15ULL + 777)."""
    s = (int(seed) * 0x9E3779B97F4A7C15 + 777) % (1 << 64)
    return np.array([s], dtype=np.uint64)


# ------------------------------------------------------------------------------------------------ ILS state
@njit(cache=True)
def _relabel(u, key, nxt, prv):
    """u is linked in the list; spread keys over the smallest doubling window around u with enough room."""
    step = 1
    moved = 0
    while True:
        left = u
        right = u
        cnt = 1
        for _ in range(step):
            if prv[left] == -1:
                break
            left = prv[left]
            cnt += 1
        for _ in range(step):
            if nxt[right] == -1:
                break
            right = nxt[right]
            cnt += 1
        lo = key[prv[left]] if prv[left] != -1 else np.int64(0)
        hi = key[nxt[right]] if nxt[right] != -1 else TOP
        g = (hi - lo) // (cnt + 1)
        full = prv[left] == -1 and nxt[right] == -1
        if g >= MINGAP or (full and g >= 1):
            x = left
            kk = lo
            for _ in range(cnt):
                kk += g
                key[x] = kk
                x = nxt[x]
            moved += cnt
            return moved
        if full:
            raise RuntimeError('order-maintenance key space exhausted')
        step *= 2


@njit(cache=True)
def _place_key(u, key, nxt, prv, meta):
    lo = key[prv[u]] if prv[u] != -1 else np.int64(0)
    hi = key[nxt[u]] if nxt[u] != -1 else TOP
    if hi - lo >= 2:
        key[u] = lo + (hi - lo) // 2
    else:
        meta[RELABELS] += 1
        meta[RELABEL_WORK] += _relabel(u, key, nxt, prv)


@njit(cache=True)
def _move(u, v, after, mode, p, E, off, inc, key, nxt, prv, meta, deg, change, seen, aff, terms, rs,
          logu, logp, logn, logging):
    """Exactly insert_ils.cpp State::move. mode 0 strict, 1 plateau with prob p, 2 forced.
    Returns (accepted, delta)."""
    if u == v:
        return False, np.int64(0)
    meta[STAMP] += 1
    stamp = meta[STAMP]
    na = 0
    aff[na] = u
    na += 1
    ku = key[u]
    kv = key[v]
    for q in range(off[u], off[u + 1]):
        e = inc[q]
        w = E[2 * e] + E[2 * e + 1] - u
        kw = key[w]
        old = ku < kw
        if after:
            now = kw > kv
        else:
            now = kw >= kv
        if old != now:
            z = 1 if now else -1
            change[u] += z
            change[w] -= z
            aff[na] = w
            na += 1
    nt = 0
    for i in range(na):
        x = aff[i]
        for q in range(off[x], off[x + 1]):
            e = inc[q]
            if seen[e] != stamp:
                seen[e] = stamp
                terms[nt] = e
                nt += 1
    delta = np.int64(0)
    for i in range(nt):
        e = terms[i]
        a = E[2 * e]
        b = E[2 * e + 1]
        da = deg[a]
        db = deg[b]
        ca = da + change[a]
        cb = db + change[b]
        delta += (ca if ca < cb else cb) - (da if da < db else db)
    acc = False
    if delta < 0:
        acc = True
    elif mode == 1 and delta == 0:
        acc = rng_unif(rs) < p
    elif mode == 2:
        acc = True
    if acc:
        op = prv[u]
        on = nxt[u]
        if logging:
            L = meta[LOGLEN]
            if L >= logu.shape[0]:
                raise RuntimeError('ILS undo log overflow')
            logu[L] = u
            logp[L] = op
            logn[L] = on
            meta[LOGLEN] = L + 1
        # unlink u
        if op != -1:
            nxt[op] = on
        else:
            meta[HEAD] = on
        if on != -1:
            prv[on] = op
        else:
            meta[TAIL] = op
        if after:
            q2 = nxt[v]
            prv[u] = v
            nxt[u] = q2
            nxt[v] = u
            if q2 != -1:
                prv[q2] = u
            else:
                meta[TAIL] = u
        else:
            p2 = prv[v]
            nxt[u] = v
            prv[u] = p2
            prv[v] = u
            if p2 != -1:
                nxt[p2] = u
            else:
                meta[HEAD] = u
        _place_key(u, key, nxt, prv, meta)
        for i in range(na):
            x = aff[i]
            deg[x] += change[x]
        meta[WORK] += delta
    for i in range(na):
        change[aff[i]] = 0
    if acc:
        return True, delta
    return False, np.int64(0)


@njit(cache=True)
def ils_init(order, E, key, nxt, prv, meta, deg):
    n = order.shape[0]
    g = TOP // (n + 1)
    for i in range(n):
        x = order[i]
        key[x] = (i + 1) * g
        prv[x] = order[i - 1] if i > 0 else -1
        nxt[x] = order[i + 1] if i + 1 < n else -1
    meta[HEAD] = order[0] if n > 0 else -1
    meta[TAIL] = order[n - 1] if n > 0 else -1
    rank = rank_of(order)
    d = outdeg_from_rank(n, E, rank)
    for i in range(n):
        deg[i] = d[i]
    meta[WORK] = work_of_degrees(E, deg)
    meta[LOGLEN] = 0


@njit(cache=True)
def ils_export(n, nxt, meta):
    out = np.empty(n, np.int64)
    x = meta[HEAD]
    i = 0
    while x != -1:
        out[i] = x
        i += 1
        x = nxt[x]
    if i != n:
        raise RuntimeError('linked list corrupted')
    return out


@njit(cache=True)
def ils_run_count(count, p, E, off, inc, key, nxt, prv, meta, deg, change, seen, aff, terms, rs,
                  logu, logp, logn, logging, stats):
    """C++ ils_run: count proposals (uniform edge, endpoint, side); plateau mode iff p > 0."""
    m = E.shape[0] // 2
    mode = 1 if p > 0 else 0
    for _ in range(count):
        e = rng_below(rs, m)
        side = rng_below(rs, 2)
        aft = rng_below(rs, 2)
        u = E[2 * e + side]
        v = E[2 * e + 1 - side]
        acc, dl = _move(u, v, aft != 0, mode, p, E, off, inc, key, nxt, prv, meta, deg, change, seen, aff,
                        terms, rs, logu, logp, logn, logging)
        stats[ST_PROPOSALS] += 1
        if acc:
            stats[ST_ACCEPTED] += 1
            if dl < 0:
                stats[ST_IMPROVING] += 1


@njit(cache=True)
def ils_kick(k, E, off, inc, key, nxt, prv, meta, deg, change, seen, aff, terms, rs, logu, logp, logn,
             logging):
    """C++ ils_kick: k random moves applied regardless of delta."""
    m = E.shape[0] // 2
    for _ in range(k):
        e = rng_below(rs, m)
        side = rng_below(rs, 2)
        aft = rng_below(rs, 2)
        _move(E[2 * e + side], E[2 * e + 1 - side], aft != 0, 2, 0.0, E, off, inc, key, nxt, prv, meta, deg,
              change, seen, aff, terms, rs, logu, logp, logn, logging)


@njit(cache=True)
def ils_undo_all(E, off, inc, key, nxt, prv, meta, deg, change, seen, aff, terms, rs, logu, logp, logn):
    """Undo every logged move (newest first): restores exactly the order (hence degrees and work) of the
    state at the time the log was last cleared."""
    L = meta[LOGLEN]
    for i in range(L - 1, -1, -1):
        u = logu[i]
        if logp[i] != -1:
            _move(u, logp[i], True, 2, 0.0, E, off, inc, key, nxt, prv, meta, deg, change, seen, aff, terms, rs,
                  logu, logp, logn, False)
        else:
            _move(u, logn[i], False, 2, 0.0, E, off, inc, key, nxt, prv, meta, deg, change, seen, aff, terms, rs,
                  logu, logp, logn, False)
    meta[LOGLEN] = 0
    return L


@njit(cache=True)
def ils_chunks(nchunks, chunk, p, k, S, E, off, inc, key, nxt, prv, meta, deg, change, seen, aff, terms, rs,
               logu, logp, logn, stats):
    """nchunks iterations of Algorithm 1 lines 7-9. The current state is always the incumbent plus the logged
    moves; meta[INCUMBENT] = Lambda(pi*), meta[COUNTER] = c."""
    for _ in range(nchunks):
        ils_run_count(chunk, p, E, off, inc, key, nxt, prv, meta, deg, change, seen, aff, terms, rs,
                      logu, logp, logn, True, stats)
        stats[ST_CHUNKS] += 1
        if meta[WORK] < meta[INCUMBENT]:
            meta[INCUMBENT] = meta[WORK]
            meta[LOGLEN] = 0
            meta[COUNTER] = 0
            stats[ST_INCUMBENT_UPDATES] += 1
        else:
            meta[COUNTER] += 1
        if meta[COUNTER] == S:
            stats[ST_UNDONE] += ils_undo_all(E, off, inc, key, nxt, prv, meta, deg, change, seen, aff, terms, rs,
                                             logu, logp, logn)
            if meta[WORK] != meta[INCUMBENT]:
                raise RuntimeError('undo did not restore the incumbent work')
            ils_kick(k, E, off, inc, key, nxt, prv, meta, deg, change, seen, aff, terms, rs, logu, logp, logn,
                     True)
            stats[ST_KICKS] += 1
            meta[COUNTER] = 0


class ILSState:
    """Python holder of the numba ILS state arrays."""

    def __init__(self, n, off, inc, E, order, seed, log_capacity):
        self.n = n
        self.off, self.inc, self.E = off, inc, E
        m = E.shape[0] // 2
        self.key = np.empty(n, np.int64)
        self.nxt = np.empty(n, np.int64)
        self.prv = np.empty(n, np.int64)
        self.meta = np.zeros(META_SIZE, np.int64)
        self.deg = np.zeros(n, np.int64)
        self.change = np.zeros(n, np.int64)
        self.seen = np.zeros(m, np.int64)
        self.aff = np.empty(max(n, 1), np.int64)
        self.terms = np.empty(max(m, 1), np.int64)
        self.rs = rng_state(seed)
        cap = int(log_capacity)
        self.logu = np.empty(cap, np.int64)
        self.logp = np.empty(cap, np.int64)
        self.logn = np.empty(cap, np.int64)
        self.stats = np.zeros(STATS_SIZE, np.int64)
        ils_init(np.ascontiguousarray(order, dtype=np.int64), E, self.key, self.nxt, self.prv, self.meta, self.deg)

    def _a(self):
        return (self.E, self.off, self.inc, self.key, self.nxt, self.prv, self.meta, self.deg, self.change,
                self.seen, self.aff, self.terms, self.rs, self.logu, self.logp, self.logn)

    @property
    def work(self):
        return int(self.meta[WORK])

    def run(self, count, p, logging=False):
        ils_run_count(int(count), float(p), *self._a(), bool(logging), self.stats)

    def kick(self, k, logging=False):
        ils_kick(int(k), *self._a(), bool(logging))

    def undo_all(self):
        return int(ils_undo_all(*self._a()))

    def chunks(self, nchunks, chunk, p, k, S):
        ils_chunks(int(nchunks), int(chunk), float(p), int(k), int(S), *self._a(), self.stats)

    def export(self):
        return ils_export(self.n, self.nxt, self.meta)

    def check_keys(self):
        o = self.export()
        return bool(np.all(np.diff(self.key[o]) > 0)) and (self.n == 0 or (self.key[o[0]] > 0 and self.key[o[-1]] < TOP))


# ------------------------------------------------------------------------------------------------ verification lister
@njit(cache=True)
def adaptive_forward_lister(n, E, order):
    """Independent adaptive-forward triangle lister: builds sorted forward adjacency under `order`; for each
    edge u->v scans the smaller forward set and tests membership in the other by binary search.
    Returns (probes, triangles). probes must equal Lambda(order)."""
    m = E.shape[0] // 2
    rank = np.empty(n, np.int64)
    for i in range(n):
        rank[order[i]] = i
    fo = np.zeros(n + 1, np.int64)
    for e in range(m):
        a = E[2 * e]
        b = E[2 * e + 1]
        t = a if rank[a] < rank[b] else b
        fo[t + 1] += 1
    for i in range(n):
        fo[i + 1] += fo[i]
    fill = fo[:n].copy()
    fw = np.empty(m, np.int64)
    for e in range(m):
        a = E[2 * e]
        b = E[2 * e + 1]
        if rank[a] < rank[b]:
            fw[fill[a]] = b
            fill[a] += 1
        else:
            fw[fill[b]] = a
            fill[b] += 1
    for x in range(n):
        if fo[x + 1] - fo[x] > 1:
            fw[fo[x]:fo[x + 1]].sort()
    probes = np.int64(0)
    tri = np.int64(0)
    for u in range(n):
        for q in range(fo[u], fo[u + 1]):
            v = fw[q]
            du = fo[u + 1] - fo[u]
            dv = fo[v + 1] - fo[v]
            if du <= dv:
                s0, s1, l0, l1 = fo[u], fo[u + 1], fo[v], fo[v + 1]
            else:
                s0, s1, l0, l1 = fo[v], fo[v + 1], fo[u], fo[u + 1]
            for i in range(s0, s1):
                w = fw[i]
                probes += 1
                lo = l0
                hi = l1
                while lo < hi:
                    mid = (lo + hi) // 2
                    if fw[mid] < w:
                        lo = mid + 1
                    else:
                        hi = mid
                if lo < l1 and fw[lo] == w:
                    tri += 1
    return probes, tri


# ------------------------------------------------------------------------------------------------ warm-up
def warmup():
    """Compile every kernel on a toy graph (excluded from all timings)."""
    edges = np.array([[0, 1], [0, 2], [1, 2], [2, 3], [3, 4], [1, 4], [0, 4], [2, 4]], dtype=np.int64)
    n = 5
    off, inc, E = csr(n, edges)
    o = np.arange(n, dtype=np.int64)
    rank_of(o); is_permutation(o)
    d = outdeg_from_rank(n, E, rank_of(o)); work_of_degrees(E, d)
    a, nd, w = tangent_round(n, E, d); order_by_key_then_id(a)
    tangent_refine(n, E, o)
    noff, nbr = neighbor_csr(n, edges); smallest_last(n, noff, nbr)
    st = ILSState(n, off, inc, E, o, 1, 256 * 3 + 20)
    st.meta[INCUMBENT] = st.work
    st.run(10, 1.0); st.run(10, 0.0); st.kick(3); st.chunks(3, 16, 1.0, 2, 2); st.undo_all(); st.export()
    adaptive_forward_lister(n, E, o)
