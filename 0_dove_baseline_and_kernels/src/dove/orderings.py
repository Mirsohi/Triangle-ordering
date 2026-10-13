"""Handcrafted orderings and exact ordered triangle-listing primitives."""

from __future__ import annotations

import heapq

import numpy as np

from .graph import GraphData


def order_from_scores(scores: np.ndarray) -> np.ndarray:
    """Return the vertex permutation induced by scores, breaking ties by id."""

    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1 or not np.all(np.isfinite(scores)):
        raise ValueError("scores must be a finite one-dimensional vector")
    return np.lexsort((np.arange(scores.size, dtype=np.int64), scores))


def rank_from_order(order: np.ndarray) -> np.ndarray:
    order = np.asarray(order, dtype=np.int64)
    if order.ndim != 1:
        raise ValueError("order must be one-dimensional")
    if order.size and (
        int(order.min()) < 0
        or int(order.max()) >= order.size
        or not np.all(np.bincount(order, minlength=order.size) == 1)
    ):
        raise ValueError("order must be a permutation of 0, ..., n-1")
    rank = np.empty(order.size, dtype=np.int64)
    rank[order] = np.arange(order.size, dtype=np.int64)
    return rank


def normalized_scores(order: np.ndarray) -> np.ndarray:
    """Map a permutation to centered scores in [-1, 1]."""

    rank = rank_from_order(order).astype(np.float64)
    if rank.size <= 1:
        return np.zeros_like(rank)
    return 2.0 * rank / (rank.size - 1) - 1.0


def candidate_work(graph: GraphData, order_or_scores: np.ndarray) -> int:
    """Exact AOT candidate-probe count Lambda for an order or score vector.

    An integer permutation is interpreted as an order.  A floating vector is
    interpreted as one score per vertex. Only edge comparisons are required.
    """

    value = np.asarray(order_or_scores)
    if value.shape != (graph.n,):
        raise ValueError("one order position or score is required per vertex")
    u, v = graph.edges[:, 0], graph.edges[:, 1]
    if value.dtype.kind in "iu":
        rank = rank_from_order(value)
        before = rank[u] < rank[v]
    else:
        if not np.all(np.isfinite(value)):
            raise ValueError("scores must be finite")
        before = (value[u] < value[v]) | ((value[u] == value[v]) & (u < v))
    tails = np.where(before, u, v)
    outdegree = np.bincount(tails, minlength=graph.n)
    return int(np.minimum(outdegree[u], outdegree[v]).sum(dtype=np.int64))


def listing_costs(
    graph: GraphData, order_or_scores: np.ndarray
) -> tuple[int, int, int]:
    """Return ``(Lambda, C_pp, C_pm)`` for one orientation.

    ``C_pp = sum_v d+(v)^2`` and ``C_pm = sum_v d+(v)d-(v)`` are the two
    ordering objectives studied by Lécuyer et al.  The adaptive AOT work
    satisfies ``Lambda <= min(C_pp, C_pm)`` edge by edge.
    """

    value = np.asarray(order_or_scores)
    if value.shape != (graph.n,):
        raise ValueError("one order position or score is required per vertex")
    order = (
        value.astype(np.int64, copy=False)
        if value.dtype.kind in "iu"
        else order_from_scores(value)
    )
    rank = rank_from_order(order)
    u, v = graph.edges[:, 0], graph.edges[:, 1]
    tails = np.where(rank[u] < rank[v], u, v)
    heads = np.where(rank[u] < rank[v], v, u)
    outdegree = np.bincount(tails, minlength=graph.n)
    adaptive = int(np.minimum(outdegree[u], outdegree[v]).sum(dtype=np.int64))
    c_pp = int(outdegree[tails].sum(dtype=np.int64))
    c_pm = int(outdegree[heads].sum(dtype=np.int64))
    return adaptive, c_pp, c_pm


def natural_order(graph: GraphData) -> np.ndarray:
    return np.arange(graph.n, dtype=np.int64)


def random_order(graph: GraphData, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).permutation(graph.n)


def degree_order(graph: GraphData, descending: bool = False) -> np.ndarray:
    degree = graph.degree
    key = -degree if descending else degree
    return np.lexsort((np.arange(graph.n), key))


def degeneracy_order(graph: GraphData) -> np.ndarray:
    """Smallest-last peeling order with deterministic ties."""

    current = graph.degree.copy()
    removed = np.zeros(graph.n, dtype=bool)
    queue = [(int(current[v]), v) for v in range(graph.n)]
    heapq.heapify(queue)
    order: list[int] = []
    while queue:
        degree, u = heapq.heappop(queue)
        if removed[u] or degree != current[u]:
            continue
        removed[u] = True
        order.append(u)
        for raw_v in graph.adjacency[u]:
            v = int(raw_v)
            if not removed[v] and current[v] > degree:
                current[v] -= 1
                heapq.heappush(queue, (int(current[v]), v))
    return np.asarray(order, dtype=np.int64)


def split_order(graph: GraphData) -> np.ndarray:
    """The linear-time Split ordering of Lecuyer et al. (2023)."""

    sequence = degree_order(graph, descending=True)
    return np.concatenate((sequence[::2], sequence[1::2][::-1]))


def check_order(graph: GraphData) -> np.ndarray:
    """The linear-time Check ordering of Lecuyer et al. (2023)."""

    sequence = degree_order(graph, descending=True)
    state = np.zeros(graph.n, dtype=np.int8)  # -1 left, 0 unseen, +1 right
    left: list[int] = []
    right: list[int] = []
    for raw_u in sequence:
        u = int(raw_u)
        neighbor_state = state[graph.adjacency[u]]
        n_left = int(np.count_nonzero(neighbor_state < 0))
        n_right = int(np.count_nonzero(neighbor_state > 0))
        n_unseen = len(neighbor_state) - n_left - n_right
        cost_left = n_left * (n_right + n_unseen)
        cost_right = (n_left + n_unseen) * n_right
        if cost_left <= cost_right:
            state[u] = -1
            left.append(u)
        else:
            state[u] = 1
            right.append(u)
    return np.asarray(left + right[::-1], dtype=np.int64)


def baseline_orders(graph: GraphData, seed: int = 0) -> dict[str, np.ndarray]:
    return {
        "Natural": natural_order(graph),
        "Random": random_order(graph, seed),
        "Degree": degree_order(graph),
        "Degeneracy": degeneracy_order(graph),
        "Split": split_order(graph),
        "Check": check_order(graph),
    }


def best_baseline(graph: GraphData, seed: int = 0) -> tuple[str, np.ndarray, int]:
    candidates = baseline_orders(graph, seed)
    measured = {name: candidate_work(graph, order) for name, order in candidates.items()}
    name = min(measured, key=lambda item: (measured[item], item))
    return name, candidates[name], measured[name]


def forward_adjacency(graph: GraphData, order_or_scores: np.ndarray) -> list[set[int]]:
    value = np.asarray(order_or_scores)
    order = value.astype(np.int64, copy=False) if value.dtype.kind in "iu" else order_from_scores(value)
    rank = rank_from_order(order)
    forward: list[set[int]] = [set() for _ in range(graph.n)]
    for raw_u, raw_v in graph.edges:
        u, v = int(raw_u), int(raw_v)
        if rank[u] < rank[v]:
            forward[u].add(v)
        else:
            forward[v].add(u)
    return forward


def exact_triangle_count(graph: GraphData, order_or_scores: np.ndarray) -> int:
    """Count every triangle once by intersecting forward neighborhoods."""

    forward = forward_adjacency(graph, order_or_scores)
    triangles = 0
    for u, out_u in enumerate(forward):
        for v in out_u:
            out_v = forward[v]
            small, large = (out_u, out_v) if len(out_u) <= len(out_v) else (out_v, out_u)
            triangles += sum(1 for w in small if w in large)
    return triangles
