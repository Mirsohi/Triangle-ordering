"""Derivative-free controls and optional hard-objective refinement."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from .graph import GraphData
from .orderings import best_baseline, candidate_work, normalized_scores, order_from_scores


@dataclass(frozen=True)
class LocalSearchResult:
    order: np.ndarray
    work: int
    evaluations: int
    seconds: float


def relocate(order: np.ndarray, source: int, destination: int) -> np.ndarray:
    proposal = np.asarray(order, dtype=np.int64).copy()
    if source == destination:
        return proposal
    vertex = proposal[source]
    if source < destination:
        proposal[source:destination] = proposal[source + 1 : destination + 1]
    else:
        proposal[destination + 1 : source + 1] = proposal[destination:source]
    proposal[destination] = vertex
    return proposal


def random_local_search(
    graph: GraphData,
    method: str,
    *,
    seed: int,
    max_evaluations: int | None = None,
    seconds: float | None = None,
    initial_order: np.ndarray | None = None,
) -> LocalSearchResult:
    """Run a monotone random search under an evaluation or time budget."""

    if (max_evaluations is None) == (seconds is None):
        raise ValueError("set exactly one of max_evaluations and seconds")
    rng = np.random.default_rng(seed)
    if initial_order is None:
        _, starting_order, starting_work = best_baseline(graph, seed=7)
    else:
        starting_order = np.asarray(initial_order, dtype=np.int64)
        starting_work = candidate_work(graph, starting_order)
    best_order = starting_order.copy()
    best_scores = normalized_scores(best_order)
    best_work = starting_work
    evaluations = 0
    started = perf_counter()

    while True:
        elapsed = perf_counter() - started
        if max_evaluations is not None and evaluations >= max_evaluations:
            break
        if seconds is not None and elapsed >= seconds:
            break
        if method == "swap":
            i, j = map(int, rng.integers(0, graph.n, size=2))
            proposal = best_order.copy()
            proposal[i], proposal[j] = proposal[j], proposal[i]
            work = candidate_work(graph, proposal)
            if work < best_work:
                best_order, best_work = proposal, work
                best_scores = normalized_scores(best_order)
        elif method == "relocate":
            i, j = map(int, rng.integers(0, graph.n, size=2))
            proposal = relocate(best_order, i, j)
            work = candidate_work(graph, proposal)
            if work < best_work:
                best_order, best_work = proposal, work
                best_scores = normalized_scores(best_order)
        elif method == "score_noise":
            if max_evaluations is not None:
                fraction = evaluations / max(1, max_evaluations - 1)
            else:
                fraction = elapsed / max(float(seconds), 1e-12)
            sigma = 0.20 * (0.01 / 0.20) ** min(1.0, float(fraction))
            proposal_scores = best_scores + rng.normal(0.0, sigma, size=graph.n)
            work = candidate_work(graph, proposal_scores)
            if work < best_work:
                best_scores, best_work = proposal_scores, work
                best_order = order_from_scores(best_scores)
        else:
            raise ValueError(f"unknown local-search method: {method}")
        evaluations += 1

    return LocalSearchResult(
        order=best_order,
        work=best_work,
        evaluations=evaluations,
        seconds=perf_counter() - started,
    )
