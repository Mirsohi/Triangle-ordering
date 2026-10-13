from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from dove.graph import graph_from_edges  # noqa: E402
from dove.orderings import (  # noqa: E402
    baseline_orders,
    best_baseline,
    candidate_work,
    exact_triangle_count,
    listing_costs,
    rank_from_order,
)
from dove.relaxation import optimize_order, smooth_objective_and_gradient  # noqa: E402
from dove.equivariant_features import equivariant_structural_features  # noqa: E402
from dove.local_search import random_local_search  # noqa: E402


class DoveTests(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = graph_from_edges(
            [(0, 1), (1, 2), (0, 2), (1, 3), (2, 3), (3, 4)], "toy"
        )

    def test_exactness_is_order_invariant(self) -> None:
        counts = {
            exact_triangle_count(self.graph, order)
            for order in baseline_orders(self.graph, seed=3).values()
        }
        self.assertEqual(counts, {2})

    def test_work_lower_bounds_output(self) -> None:
        for order in baseline_orders(self.graph).values():
            self.assertGreaterEqual(candidate_work(self.graph, order), 2)

    def test_adaptive_work_is_below_one_sided_costs(self) -> None:
        for order in baseline_orders(self.graph, seed=5).values():
            adaptive, c_pp, c_pm = listing_costs(self.graph, order)
            self.assertEqual(adaptive, candidate_work(self.graph, order))
            self.assertLessEqual(adaptive, min(c_pp, c_pm))

    def test_analytic_gradient(self) -> None:
        scores = np.asarray([-0.8, -0.2, 0.1, 0.5, 0.9])
        value, gradient = smooth_objective_and_gradient(self.graph, scores, 0.4, 0.2)
        numerical = np.empty_like(scores)
        delta = 1e-6
        for i in range(scores.size):
            shifted = scores.copy()
            shifted[i] += delta
            upper, _ = smooth_objective_and_gradient(self.graph, shifted, 0.4, 0.2)
            shifted[i] -= 2 * delta
            lower, _ = smooth_objective_and_gradient(self.graph, shifted, 0.4, 0.2)
            numerical[i] = (upper - lower) / (2 * delta)
        self.assertTrue(np.isfinite(value))
        np.testing.assert_allclose(gradient, numerical, atol=2e-5, rtol=2e-5)

    def test_analytic_gradient_with_saturated_logits(self) -> None:
        """The stable sigmoid must remain the derivative of the value used."""

        scores = np.asarray([-100.0, -1.0, 0.0, 100.0])
        graph = graph_from_edges(
            [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)], "saturated"
        )
        _, gradient = smooth_objective_and_gradient(graph, scores, 0.01, 0.01)
        numerical = np.empty_like(scores)
        delta = 1e-5
        for i in range(scores.size):
            upper_scores = scores.copy()
            lower_scores = scores.copy()
            upper_scores[i] += delta
            lower_scores[i] -= delta
            upper = smooth_objective_and_gradient(
                graph, upper_scores, 0.01, 0.01
            )[0]
            lower = smooth_objective_and_gradient(
                graph, lower_scores, 0.01, 0.01
            )[0]
            numerical[i] = (upper - lower) / (2 * delta)
        np.testing.assert_allclose(gradient, numerical, atol=1e-9, rtol=1e-7)

    def test_optimizer_keeps_best_initial_order(self) -> None:
        result = optimize_order(self.graph, steps=20)
        self.assertLessEqual(result.work, result.initial_work)
        self.assertEqual(exact_triangle_count(self.graph, result.order), 2)

    def test_optimizer_accepts_guarded_external_initializer(self) -> None:
        proposed = np.asarray([4, 3, 2, 1, 0])
        result = optimize_order(
            self.graph,
            steps=0,
            initial_order=proposed,
            initial_name="external",
            guard_with_builtins=False,
        )
        self.assertEqual(result.work, candidate_work(self.graph, proposed))

    def test_invalid_order_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "permutation"):
            rank_from_order(np.asarray([0, 1, 1, 3, 4]))

    def test_local_search_is_monotone(self) -> None:
        _, _, baseline = best_baseline(self.graph, seed=7)
        for method in ("swap", "relocate", "score_noise"):
            result = random_local_search(
                self.graph, method, seed=3, max_evaluations=20
            )
            self.assertLessEqual(result.work, baseline)
            self.assertEqual(exact_triangle_count(self.graph, result.order), 2)

    def test_finite_temperature_bound(self) -> None:
        scores = np.asarray([-1.2, -0.4, 0.2, 0.7, 1.3])
        tau, epsilon = 0.25, 0.04
        soft, _ = smooth_objective_and_gradient(self.graph, scores, tau, epsilon)
        hard = candidate_work(self.graph, scores)
        u, v = self.graph.edges[:, 0], self.graph.edges[:, 1]
        gamma = float(np.min(np.abs(scores[u] - scores[v])))
        eta = 1.0 / (1.0 + np.exp(gamma / tau))
        degree = self.graph.degree.astype(float)
        bound = eta * float(np.maximum(degree[u], degree[v]).sum()) + self.graph.m * epsilon / 2.0
        self.assertLessEqual(abs(soft - hard), bound + 1e-12)

    def test_equivariant_features_under_relabeling(self) -> None:
        perm = np.asarray([3, 1, 4, 0, 2])
        relabeled = graph_from_edges(
            [(int(perm[u]), int(perm[v])) for u, v in self.graph.edges], "toy-permuted"
        )
        base = equivariant_structural_features(self.graph)
        moved = equivariant_structural_features(relabeled)[perm]
        np.testing.assert_allclose(base, moved, atol=1e-10, rtol=1e-10)


if __name__ == "__main__":
    unittest.main()
