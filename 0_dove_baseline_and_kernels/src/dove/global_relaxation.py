"""A documented NeuralSort-to-Lambda adapter (not an original NeuralSort task).

NeuralSort: Grover et al., ICLR 2019, arXiv:1903.08850, Eq. 5.
Rows are positions, columns vertices. We column-normalize the row-stochastic
matrix, then compare two independent position draws, splitting ties evenly.
This yields complementary comparisons and the correct hard limit. It does not
assert a coherent permutation distribution at finite temperature.

The dense contraction below costs O(n^3) time and O(n^2) storage. These costs
describe this explicit adapter, not all implementations of global sorting.
"""
from time import perf_counter
import numpy as np
from .orderings import candidate_work, normalized_scores, order_from_scores
from .relaxation import OptimizationResult, smooth_objective_and_gradient


def position_probabilities(scores, temperature):
    s = np.asarray(scores, dtype=np.float64)
    if temperature <= 0 or not np.all(np.isfinite(s)):
        raise ValueError("finite scores and positive temperature required")
    n = len(s)
    differences = s[:, None] - s[None, :]
    coefficients = n + 1 - 2 * np.arange(1, n + 1)
    logits = (-coefficients[:, None] * s - np.abs(differences).sum(1)) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    p = np.exp(logits)
    p /= p.sum(axis=1, keepdims=True)
    normalizer = p.sum(axis=0, keepdims=True)
    if np.any(normalizer == 0):
        raise FloatingPointError("NeuralSort column underflow")
    return p / normalizer, p, normalizer, coefficients, np.sign(differences)


def upper_half(x):
    """K @ x, K_ij = 1(j>i) + .5 1(j=i)."""
    return np.cumsum(x[::-1], axis=0)[::-1] - .5 * x


def lower_half(x):
    return np.cumsum(x, axis=0) - .5 * x


def neural_objective_and_gradient(graph, scores, temperature, epsilon):
    if epsilon <= 0:
        raise ValueError("positive smoothing required")
    h, p, normalizer, coefficients, signs = position_probabilities(scores, temperature)
    kh = upper_half(h)
    q = h.T @ kh
    u, v = graph.edges.T
    edge_p = q[u, v]
    d = np.bincount(u, edge_p, minlength=graph.n)
    d += np.bincount(v, 1 - edge_p, minlength=graph.n)
    gap = d[u] - d[v]
    root = np.sqrt(gap * gap + epsilon * epsilon)
    value = float((.5 * (d[u] + d[v] - root + epsilon)).sum())
    gd = np.bincount(u, .5 * (1 - gap/root), minlength=graph.n)
    gd += np.bincount(v, .5 * (1 + gap/root), minlength=graph.n)
    w = np.zeros_like(q)
    w[u, v] = gd[u] - gd[v]
    gh = kh @ w.T + lower_half(h) @ w
    gp = (gh - (gh * h).sum(axis=0, keepdims=True)) / normalizer
    gl = p * (gp - (gp * p).sum(axis=1, keepdims=True)) / temperature
    # logits_ij = -coeff_i*s_j - sum_k |s_j-s_k|, before /temperature.
    ga = -gl.sum(axis=0)
    gradient = -(coefficients[:, None] * gl).sum(axis=0)
    gradient += ga * signs.sum(axis=1) - ga @ signs
    return value, gradient


def optimize_same_target(graph, initial_order, method, learning_rate=.08,
                         temperature_multiplier=1., steps=100, seed=0, jitter=0.):
    if method not in {"DOVE", "NeuralSort-Lambda"}:
        raise ValueError(method)
    start = perf_counter()
    scores = normalized_scores(initial_order)
    initial = best = candidate_work(graph, initial_order)
    best_scores = scores.copy()
    if jitter:
        scores += np.random.default_rng(seed).normal(0, jitter, graph.n)
    m1 = np.zeros(graph.n)
    m2 = np.zeros(graph.n)
    trace = []
    objective = (smooth_objective_and_gradient if method == "DOVE"
                 else neural_objective_and_gradient)
    for step in range(1, steps + 1):
        base_tau = .5 * (.04/.5)**(step/steps)
        value, grad = objective(graph, scores, base_tau*temperature_multiplier,
                                max(.01, base_tau**2))
        grad /= max(graph.m, 1)
        m1 = .9*m1 + .1*grad
        m2 = .999*m2 + .001*grad*grad
        scores -= learning_rate*(m1/(1-.9**step))/(np.sqrt(m2/(1-.999**step))+1e-8)
        scores -= scores.mean()
        scores /= scores.std()+1e-9
        if step % 5 == 0 or step == steps:
            hard = candidate_work(graph, scores)
            trace.append((step, value, hard))
            if hard < best:
                best, best_scores = hard, scores.copy()
    order = order_from_scores(best_scores)
    assert candidate_work(graph, order) == best <= initial
    return OptimizationResult(order, best_scores, best, "Shared reference", initial,
                              perf_counter()-start, tuple(trace))
