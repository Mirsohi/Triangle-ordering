"""Structural features for amortized rank distillation.

Some pilot features are baseline-relative (Check rank/fractions) and inherit
Check's deterministic tie breaking; they should not be described as strictly
permutation-equivariant.
"""

from __future__ import annotations

import heapq

import numpy as np

from .graph import GraphData
from .orderings import check_order, normalized_scores, rank_from_order
from .relaxation import smooth_objective_and_gradient


FEATURE_NAMES = (
    "baseline_rank",
    "log_degree",
    "core_number",
    "neighbor_log_degree_mean",
    "neighbor_log_degree_std",
    "neighbor_log_degree_max",
    "neighbor_log_degree_min",
    "baseline_out_fraction",
    "baseline_in_fraction",
    "initial_surrogate_gradient",
)


def _robust_scale(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    median = float(np.median(values))
    spread = float(np.quantile(values, 0.75) - np.quantile(values, 0.25))
    return (values - median) / (spread + 1e-6)


def core_numbers(graph: GraphData) -> np.ndarray:
    """Compute k-core numbers by deterministic heap peeling."""

    current = graph.degree.copy()
    dead = np.zeros(graph.n, dtype=bool)
    core = np.zeros(graph.n, dtype=np.float64)
    queue = [(int(current[v]), v) for v in range(graph.n)]
    heapq.heapify(queue)
    while queue:
        degree, u = heapq.heappop(queue)
        if dead[u] or degree != current[u]:
            continue
        dead[u] = True
        core[u] = degree
        for raw_v in graph.adjacency[u]:
            v = int(raw_v)
            if not dead[v] and current[v] > degree:
                current[v] -= 1
                heapq.heappush(queue, (int(current[v]), v))
    return core


def baseline_relative_features(
    graph: GraphData, baseline_order: np.ndarray
) -> np.ndarray:
    """Build ten features relative to a supplied algorithmic baseline.

    The feature computation never counts or lists triangles.  Supplying the
    baseline explicitly avoids silently assuming that Check is best on every
    graph and makes the distillation experiment match its measured-work guard.
    """

    degree = graph.degree.astype(np.float64)
    log_degree = np.log1p(degree)
    baseline_order = np.asarray(baseline_order, dtype=np.int64)
    if baseline_order.shape != (graph.n,) or not np.array_equal(
        np.sort(baseline_order), np.arange(graph.n)
    ):
        raise ValueError("baseline_order must be a permutation of graph vertices")
    baseline_rank = rank_from_order(baseline_order).astype(np.float64)
    if graph.n > 1:
        baseline_rank /= graph.n - 1

    neighbor_mean = np.empty(graph.n, dtype=np.float64)
    neighbor_std = np.empty(graph.n, dtype=np.float64)
    neighbor_max = np.empty(graph.n, dtype=np.float64)
    neighbor_min = np.empty(graph.n, dtype=np.float64)
    for v, row in enumerate(graph.adjacency):
        values = log_degree[row]
        if values.size:
            neighbor_mean[v] = values.mean()
            neighbor_std[v] = values.std()
            neighbor_max[v] = values.max()
            neighbor_min[v] = values.min()
        else:
            neighbor_mean[v] = neighbor_std[v] = 0.0
            neighbor_max[v] = neighbor_min[v] = 0.0

    u, v = graph.edges[:, 0], graph.edges[:, 1]
    tails = np.where(baseline_rank[u] < baseline_rank[v], u, v)
    outdegree = np.bincount(tails, minlength=graph.n).astype(np.float64)
    denominator = degree + 1.0
    initial_scores = normalized_scores(baseline_order)
    _, initial_gradient = smooth_objective_and_gradient(
        graph, initial_scores, orientation_temperature=0.1, min_smoothing=0.01
    )

    return np.column_stack(
        (
            baseline_rank,
            _robust_scale(log_degree),
            _robust_scale(core_numbers(graph)),
            _robust_scale(neighbor_mean),
            _robust_scale(neighbor_std),
            _robust_scale(neighbor_max),
            _robust_scale(neighbor_min),
            outdegree / denominator,
            (degree - outdegree) / denominator,
            _robust_scale(initial_gradient),
        )
    )


def structural_features(graph: GraphData) -> np.ndarray:
    """Backward-compatible Check-relative features used by the original pilot."""

    return baseline_relative_features(graph, check_order(graph))
