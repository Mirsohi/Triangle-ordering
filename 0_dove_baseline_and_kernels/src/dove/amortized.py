"""Cross-graph distillation of optimized DOVE ranks."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Sequence

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

from .features import baseline_relative_features, structural_features
from .graph import GraphData
from .orderings import best_baseline, candidate_work, order_from_scores, rank_from_order


@dataclass(frozen=True)
class DistilledResult:
    order: np.ndarray
    predicted_order: np.ndarray
    work: int
    predicted_work: int
    fallback_work: int
    used_guard: bool
    seconds: float


@dataclass(frozen=True)
class ResidualRanker:
    """A graph-balanced model for corrections to a measured baseline rank."""

    model: ExtraTreesRegressor


def normalized_rank_target(order: np.ndarray) -> np.ndarray:
    rank = rank_from_order(order).astype(np.float64)
    if rank.size > 1:
        rank /= rank.size - 1
    return rank


def fit_ranker(
    graphs: Sequence[GraphData],
    teacher_orders: Sequence[np.ndarray],
    seed: int = 0,
    n_estimators: int = 200,
) -> ExtraTreesRegressor:
    if len(graphs) != len(teacher_orders) or not graphs:
        raise ValueError("provide one teacher ordering for every nonempty graph")
    features = np.vstack([structural_features(graph) for graph in graphs])
    targets = np.concatenate([normalized_rank_target(order) for order in teacher_orders])
    model = ExtraTreesRegressor(
        n_estimators=n_estimators,
        min_samples_leaf=3,
        max_features=0.8,
        n_jobs=-1,
        random_state=seed,
    )
    return model.fit(features, targets)


def apply_ranker(
    graph: GraphData, model: ExtraTreesRegressor, guard: bool = True
) -> DistilledResult:
    started = perf_counter()
    predicted = order_from_scores(model.predict(structural_features(graph)))
    predicted_work = candidate_work(graph, predicted)
    _, fallback, fallback_work = best_baseline(graph)
    use_guard = bool(guard and fallback_work < predicted_work)
    order = fallback if use_guard else predicted
    work = fallback_work if use_guard else predicted_work
    return DistilledResult(
        order=order,
        predicted_order=predicted,
        work=work,
        predicted_work=predicted_work,
        fallback_work=fallback_work,
        used_guard=use_guard,
        seconds=perf_counter() - started,
    )


def fit_residual_ranker(
    graphs: Sequence[GraphData],
    teacher_orders: Sequence[np.ndarray],
    seed: int = 0,
    n_estimators: int = 400,
) -> ResidualRanker:
    """Distill DOVE-T as a correction to each graph's best baseline.

    Every graph receives equal total sample weight, preventing a large graph
    from dominating the node-level regression solely because it has more
    vertices.
    """

    if len(graphs) != len(teacher_orders) or not graphs:
        raise ValueError("provide one teacher ordering for every nonempty graph")
    feature_blocks = []
    target_blocks = []
    weight_blocks = []
    for graph, teacher in zip(graphs, teacher_orders, strict=True):
        _, baseline, _ = best_baseline(graph, seed=seed)
        baseline_rank = normalized_rank_target(baseline)
        feature_blocks.append(baseline_relative_features(graph, baseline))
        target_blocks.append(normalized_rank_target(teacher) - baseline_rank)
        weight_blocks.append(np.full(graph.n, 1.0 / graph.n, dtype=np.float64))
    model = ExtraTreesRegressor(
        n_estimators=n_estimators,
        min_samples_leaf=3,
        max_features=0.8,
        n_jobs=-1,
        random_state=seed,
    )
    model.fit(
        np.vstack(feature_blocks),
        np.concatenate(target_blocks),
        sample_weight=np.concatenate(weight_blocks),
    )
    return ResidualRanker(model=model)


def apply_residual_ranker(
    graph: GraphData,
    ranker: ResidualRanker,
    seed: int = 0,
    guard: bool = True,
) -> DistilledResult:
    started = perf_counter()
    _, baseline, baseline_work = best_baseline(graph, seed=seed)
    baseline_rank = normalized_rank_target(baseline)
    correction = ranker.model.predict(baseline_relative_features(graph, baseline))
    predicted = order_from_scores(baseline_rank + correction)
    predicted_work = candidate_work(graph, predicted)
    use_guard = bool(guard and baseline_work < predicted_work)
    order = baseline if use_guard else predicted
    work = baseline_work if use_guard else predicted_work
    return DistilledResult(
        order=order,
        predicted_order=predicted,
        work=work,
        predicted_work=predicted_work,
        fallback_work=baseline_work,
        used_guard=use_guard,
        seconds=perf_counter() - started,
    )
