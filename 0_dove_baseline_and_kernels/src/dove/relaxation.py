"""The differentiable DOVE relaxation and transductive optimizer."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from .graph import GraphData
from .orderings import best_baseline, candidate_work, normalized_scores, order_from_scores


@dataclass(frozen=True)
class OptimizationResult:
    order: np.ndarray
    scores: np.ndarray
    work: int
    initial_method: str
    initial_work: int
    seconds: float
    trace: tuple[tuple[int, float, int], ...]


def smooth_objective_and_gradient(
    graph: GraphData,
    scores: np.ndarray,
    orientation_temperature: float,
    min_smoothing: float,
    objective_kind: str = "lambda",
) -> tuple[float, np.ndarray]:
    """Return the relaxed AOT work and its gradient in O(m+n) time."""

    scores = np.asarray(scores, dtype=np.float64)
    tau = float(orientation_temperature)
    epsilon = float(min_smoothing)
    if tau <= 0 or epsilon <= 0:
        raise ValueError("temperatures must be positive")
    u, v = graph.edges[:, 0], graph.edges[:, 1]
    logits = (scores[v] - scores[u]) / tau
    # Stable logistic evaluation without clipping.  Clipping changes the
    # mathematical objective and makes the analytic gradient incorrect once a
    # logit reaches the clipping threshold.  ``logaddexp`` remains stable for
    # large score gaps while preserving the stated relaxation exactly (up to
    # ordinary floating-point underflow to 0 or rounding to 1).
    p_uv = np.exp(-np.logaddexp(0.0, -logits))
    soft_outdegree = np.bincount(u, p_uv, minlength=graph.n)
    soft_outdegree += np.bincount(v, 1.0 - p_uv, minlength=graph.n)

    if objective_kind == "cpp":
        objective = float(np.dot(soft_outdegree, soft_outdegree))
        degree_gradient = 2 * soft_outdegree
    elif objective_kind == "cpm":
        degree = np.bincount(graph.edges.ravel(), minlength=graph.n)
        objective = float(np.dot(soft_outdegree, degree - soft_outdegree))
        degree_gradient = degree - 2 * soft_outdegree
    elif objective_kind == "lambda":
        degree_u = soft_outdegree[u]
        degree_v = soft_outdegree[v]
        gap = degree_u - degree_v
        root = np.sqrt(gap * gap + epsilon * epsilon)
        smooth_min = 0.5 * (degree_u + degree_v - root + epsilon)
        objective = float(smooth_min.sum())
        partial_u = 0.5 * (1.0 - gap / root)
        partial_v = 0.5 * (1.0 + gap / root)
        degree_gradient = np.bincount(u, partial_u, minlength=graph.n)
        degree_gradient += np.bincount(v, partial_v, minlength=graph.n)
    else:
        raise ValueError("objective_kind must be lambda, cpp, or cpm")

    p_gradient = degree_gradient[u] - degree_gradient[v]
    edge_gradient = p_gradient * p_uv * (1.0 - p_uv) / tau
    score_gradient = np.bincount(v, edge_gradient, minlength=graph.n)
    score_gradient -= np.bincount(u, edge_gradient, minlength=graph.n)
    return objective, score_gradient


def optimize_order(
    graph: GraphData,
    steps: int = 200,
    learning_rate: float = 0.08,
    initial_temperature: float = 0.5,
    final_temperature: float = 0.04,
    eval_every: int = 5,
    seed: int = 0,
    initial_order: np.ndarray | None = None,
    initial_name: str = "Provided",
    guard_with_builtins: bool = True,
    initial_scores: np.ndarray | None = None,
    smoothing_floor: float = 0.01,
    objective_kind: str = "lambda",
) -> OptimizationResult:
    """Optimize one score per vertex without enumerating a triangle.

    When ``initial_order`` is supplied, the default is to start from whichever
    is better under the exact objective: that order or the best built-in
    baseline. Set ``guard_with_builtins=False`` only when the caller has already
    performed that comparison and supplied the guarded incumbent.
    """

    if initial_order is not None and not guard_with_builtins:
        proposed_order = np.asarray(initial_order, dtype=np.int64)
        proposed_work = candidate_work(graph, proposed_order)
        initial_method = initial_name
        selected_order = proposed_order
        initial_work = proposed_work
    else:
        initial_method, selected_order, initial_work = best_baseline(graph, seed=seed)
        if initial_order is not None:
            proposed_order = np.asarray(initial_order, dtype=np.int64)
            proposed_work = candidate_work(graph, proposed_order)
            if proposed_work < initial_work:
                initial_method = initial_name
                selected_order = proposed_order
                initial_work = proposed_work
    scores = normalized_scores(selected_order)
    best_scores = scores.copy()
    best_work = initial_work
    if initial_scores is not None:
        scores = np.asarray(initial_scores, dtype=np.float64).copy()
        if scores.shape != (graph.n,) or not np.all(np.isfinite(scores)):
            raise ValueError("initial_scores must be a finite length-n vector")
        proposed_work = candidate_work(graph, scores)
        if proposed_work < best_work:
            best_scores, best_work = scores.copy(), proposed_work
    first_moment = np.zeros(graph.n, dtype=np.float64)
    second_moment = np.zeros(graph.n, dtype=np.float64)
    trace: list[tuple[int, float, int]] = [(0, float("nan"), initial_work)]
    started = perf_counter()

    for step in range(1, steps + 1):
        fraction = step / max(1, steps)
        tau = initial_temperature * (final_temperature / initial_temperature) ** fraction
        epsilon = max(smoothing_floor, tau * tau)
        objective, gradient = smooth_objective_and_gradient(graph, scores, tau, epsilon, objective_kind)
        gradient /= max(1, graph.m)

        first_moment = 0.9 * first_moment + 0.1 * gradient
        second_moment = 0.999 * second_moment + 0.001 * gradient * gradient
        corrected_first = first_moment / (1.0 - 0.9**step)
        corrected_second = second_moment / (1.0 - 0.999**step)
        scores -= learning_rate * corrected_first / (np.sqrt(corrected_second) + 1e-8)
        scores -= scores.mean()
        scores /= scores.std() + 1e-9

        if step % eval_every == 0 or step == steps:
            hard_work = candidate_work(graph, scores)
            trace.append((step, objective, hard_work))
            if hard_work < best_work:
                best_work = hard_work
                best_scores = scores.copy()

    return OptimizationResult(
        order=order_from_scores(best_scores),
        scores=best_scores,
        work=best_work,
        initial_method=initial_method,
        initial_work=initial_work,
        seconds=perf_counter() - started,
        trace=tuple(trace),
    )
