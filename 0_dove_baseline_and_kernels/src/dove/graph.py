"""Dependency-light graph loading used by the DOVE experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np


@dataclass(frozen=True)
class GraphData:
    """A simple undirected graph with dense internal vertex identifiers."""

    name: str
    edges: np.ndarray
    adjacency: tuple[np.ndarray, ...]
    original_ids: np.ndarray
    family: str = "unknown"

    @property
    def n(self) -> int:
        return len(self.adjacency)

    @property
    def m(self) -> int:
        return int(self.edges.shape[0])

    @property
    def degree(self) -> np.ndarray:
        return np.fromiter(
            (len(row) for row in self.adjacency), dtype=np.int64, count=self.n
        )


def graph_from_edges(
    edges: Iterable[tuple[int, int]], name: str, family: str = "unknown"
) -> GraphData:
    """Clean, deduplicate, and densely remap an integer edge stream."""

    clean: set[tuple[int, int]] = set()
    vertices: set[int] = set()
    for raw_u, raw_v in edges:
        u, v = int(raw_u), int(raw_v)
        if u == v:
            continue
        if u > v:
            u, v = v, u
        clean.add((u, v))
        vertices.add(u)
        vertices.add(v)
    if not clean:
        raise ValueError(f"{name}: no usable undirected edges")

    original_ids = np.asarray(sorted(vertices), dtype=np.int64)
    remap = {int(v): i for i, v in enumerate(original_ids)}
    dense_edges = np.asarray(
        [(remap[u], remap[v]) for u, v in sorted(clean)], dtype=np.int32
    )
    rows: list[list[int]] = [[] for _ in range(len(original_ids))]
    for u, v in dense_edges:
        rows[int(u)].append(int(v))
        rows[int(v)].append(int(u))
    adjacency = tuple(np.asarray(sorted(row), dtype=np.int32) for row in rows)
    return GraphData(name, dense_edges, adjacency, original_ids, family)


def _edge_stream(path: Path) -> Iterator[tuple[int, int]]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith(("#", "%")):
                continue
            fields = stripped.replace(",", " ").split()
            if len(fields) < 2:
                continue
            try:
                yield int(fields[0]), int(fields[1])
            except ValueError:
                continue


def load_edgelist(
    path: str | Path, name: str | None = None, family: str = "unknown"
) -> GraphData:
    path = Path(path)
    return graph_from_edges(_edge_stream(path), name or path.stem, family)
