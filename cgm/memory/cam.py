from __future__ import annotations

import functools

import numpy as np

from cgm.memory.hnsw import HNSW
from cgm.vsa.hypervector import SparseBinaryHV
from cgm.vsa.operations import overlap, overlap_batch


def _overlap_distance(K: int, a: SparseBinaryHV, b: SparseBinaryHV) -> float:
    return float(K - overlap(a, b))


def _overlap_distance_batch(
    K: int,
    query: SparseBinaryHV,
    items: list[SparseBinaryHV],
) -> np.ndarray:
    """Batched ``_overlap_distance``. Returns float64 array of distances
    for ``query`` against each item. Wired into HNSW's ``batch_distance``
    parameter; gives ~3.3x speedup on the per-search-layer neighbor loop
    by collapsing N pairwise ``intersect1d`` calls into one
    ``np.bitwise_count`` over packed dense bitmaps (open question #20
    step (C))."""
    return (K - overlap_batch(query, items)).astype(np.float64)


class CAM:
    """Content-addressable memory over sparse binary hypervectors.

    Wraps an ``HNSW`` index with overlap-based distance ``K - overlap`` and
    enforces dimension/sparsity contracts on insert. Tracks per-node access
    counts for downstream policies (replay prioritization, LRU-style eviction).

    The write policy here is the unconditional ``insert`` — a plain CAM. The
    episodic and semantic write policies (novelty gating, EMA update-or-insert)
    are built on top of this class in sibling modules.
    """

    def __init__(
        self,
        D: int,
        K: int,
        *,
        M: int = 16,
        ef_construction: int = 100,
        ef_search: int = 50,
        rng: np.random.Generator | None = None,
    ) -> None:
        if D <= 0:
            raise ValueError(f"D must be positive, got {D}")
        if K < 0 or K > D:
            raise ValueError(f"K must be in [0, {D}], got {K}")

        self._D = D
        self._K = K

        self._hnsw: HNSW = HNSW(
            distance=functools.partial(_overlap_distance, K),
            batch_distance=functools.partial(_overlap_distance_batch, K),
            M=M,
            ef_construction=ef_construction,
            ef_search=ef_search,
            rng=rng,
        )
        self._access_counts: dict[int, int] = {}
        self._last_access: dict[int, int] = {}
        self._step: int = 0

    @property
    def D(self) -> int:
        return self._D

    @property
    def K(self) -> int:
        return self._K

    def __len__(self) -> int:
        return len(self._hnsw)

    def insert(self, hv: SparseBinaryHV) -> int:
        if hv.D != self._D:
            raise ValueError(f"HV dimension {hv.D} != CAM D {self._D}")
        if hv.K != self._K:
            raise ValueError(f"HV sparsity {hv.K} != CAM K {self._K}")
        self._step += 1
        node_id = self._hnsw.insert(hv)
        self._access_counts[node_id] = 0
        self._last_access[node_id] = self._step
        return node_id

    def nearest(
        self,
        query: SparseBinaryHV,
        k: int = 1,
        ef: int | None = None,
        *,
        track: bool = True,
    ) -> list[tuple[int, float]]:
        """Top-k nearest stored vectors. Returns (node_id, similarity) pairs
        sorted by similarity descending. When ``track`` is True (default),
        increments access count and updates last-access for each returned
        node. Internal policy checks (novelty, update-or-insert) should pass
        ``track=False`` to avoid polluting access statistics.

        Similarity is overlap normalized by the CAM's K. For a query with K
        less than the CAM's K (e.g., a partial cue), similarity caps below 1.0.
        """
        if query.D != self._D:
            raise ValueError(f"query dimension {query.D} != CAM D {self._D}")
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")

        self._step += 1
        raw = self._hnsw.search(query, k=k, ef=ef)
        out: list[tuple[int, float]] = []
        for dist, node_id in raw:
            sim = (self._K - dist) / self._K if self._K > 0 else 0.0
            if track:
                self._access_counts[node_id] = self._access_counts.get(node_id, 0) + 1
                self._last_access[node_id] = self._step
            out.append((node_id, sim))
        return out

    def get(self, node_id: int) -> SparseBinaryHV:
        return self._hnsw.get(node_id)

    def access_count(self, node_id: int) -> int:
        return self._access_counts.get(node_id, 0)

    def last_access(self, node_id: int) -> int:
        return self._last_access.get(node_id, 0)

    def delete(self, node_id: int) -> None:
        """Soft-delete a node: it no longer appears in ``nearest`` results
        and is subtracted from ``len``. Access-count bookkeeping for the
        node is retained for post-hoc analysis."""
        self._hnsw.delete(node_id)

    def _update_hv(self, node_id: int, new_hv: SparseBinaryHV) -> None:
        """Replace the stored HV at node_id in place (for EMA prototype
        updates in semantic memory). Graph edges remain from the original
        placement and become stale — drift is measured, not prevented."""
        if new_hv.D != self._D:
            raise ValueError(f"HV dimension {new_hv.D} != CAM D {self._D}")
        if new_hv.K != self._K:
            raise ValueError(f"HV sparsity {new_hv.K} != CAM K {self._K}")
        self._hnsw.update_item(node_id, new_hv)

    def compact(self) -> int:
        """Rebuild the underlying HNSW from live entries to reclaim space
        spent on soft-deleted routing hubs. Preserves node_ids and access
        statistics. Returns the number of nodes that were reclaimed.
        """
        return self._hnsw.compact()

    def clear(self) -> None:
        """Drop all entries and reset the graph. Used by Phase-7 body
        death (episodic dies with the body). Access statistics are also
        cleared, since they were body-scoped."""
        self._hnsw.clear()
        self._access_counts.clear()
        self._last_access.clear()
        self._step = 0
