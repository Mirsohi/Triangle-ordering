"""Relabeling-equivariant node features that do not enumerate triangles.

Unlike the original pilot distillation features, these features deliberately avoid
Check rank and any vertex-id tie breaking.  They are suitable inputs to a shared
score model whose real-valued output is permutation equivariant.
"""
from __future__ import annotations

import numpy as np

from .features import core_numbers
from .graph import GraphData

EQUIVARIANT_FEATURE_NAMES = (
    "log_degree",
    "core_number",
    "neighbor_log_degree_mean",
    "neighbor_log_degree_std",
    "neighbor_log_degree_max",
    "neighbor_log_degree_min",
    "relative_degree",
    "relative_core",
    "neighbor_degree_sum",
    "degree_neighbor_contrast",
)


def _robust_scale(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return values.copy()
    med = float(np.median(values))
    iqr = float(np.quantile(values, 0.75) - np.quantile(values, 0.25))
    return (values - med) / (iqr + 1e-6)


def equivariant_structural_features(graph: GraphData) -> np.ndarray:
    degree = graph.degree.astype(np.float64)
    log_degree = np.log1p(degree)
    core = core_numbers(graph).astype(np.float64)

    nmean = np.zeros(graph.n, dtype=np.float64)
    nstd = np.zeros(graph.n, dtype=np.float64)
    nmax = np.zeros(graph.n, dtype=np.float64)
    nmin = np.zeros(graph.n, dtype=np.float64)
    nsum_deg = np.zeros(graph.n, dtype=np.float64)
    for u, row in enumerate(graph.adjacency):
        if row.size:
            vals = log_degree[row]
            nmean[u] = vals.mean()
            nstd[u] = vals.std()
            nmax[u] = vals.max()
            nmin[u] = vals.min()
            nsum_deg[u] = degree[row].sum()

    rel_degree = degree / max(1.0, float(degree.max(initial=0)))
    rel_core = core / max(1.0, float(core.max(initial=0)))
    contrast = nmean - log_degree

    x = np.column_stack(
        (
            _robust_scale(log_degree),
            _robust_scale(core),
            _robust_scale(nmean),
            _robust_scale(nstd),
            _robust_scale(nmax),
            _robust_scale(nmin),
            rel_degree,
            rel_core,
            _robust_scale(np.log1p(nsum_deg)),
            _robust_scale(contrast),
        )
    )
    return np.nan_to_num(x, copy=False)
