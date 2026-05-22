from __future__ import annotations

import numpy as np

from cgm.memory.cam import CAM
from cgm.vsa.hypervector import SparseBinaryHV


class SemanticMemory:
    """EMA update-or-insert CAM for consolidated prototype storage.

    Per planning doc Section 4.5:
    - On write, if the nearest existing node has similarity at or above
      ``update_threshold``, that node's prototype is updated via EMA rather
      than inserting a new node. Below-threshold writes insert.
    - Each node maintains a dense ``D``-long float accumulator of bit-level
      counts. On each update the accumulator decays by ``(1 - alpha)`` and
      increments by 1.0 at the observation's active indices. The stored
      hypervector is the top-K-by-count projection, refreshed on every
      update.
    - The HNSW graph is not re-linked when prototypes drift. Graph quality
      is measured separately via ``drift_recall_at_1`` on a held-out probe
      set (per the planning-doc drift-handling strategy).

    Update-or-insert policy queries use CAM ``track=False`` to avoid
    polluting access statistics.
    """

    def __init__(
        self,
        D: int,
        K: int,
        *,
        update_threshold: float = 0.7,
        alpha: float = 0.05,
        M: int = 16,
        ef_construction: int = 100,
        ef_search: int = 50,
        rng: np.random.Generator | None = None,
    ) -> None:
        if not 0.0 <= update_threshold <= 1.0:
            raise ValueError(
                f"update_threshold must be in [0, 1], got {update_threshold}"
            )
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")

        self._cam = CAM(
            D=D,
            K=K,
            M=M,
            ef_construction=ef_construction,
            ef_search=ef_search,
            rng=rng,
        )
        self._update_threshold = update_threshold
        self._alpha = alpha

        self._bit_counts: dict[int, np.ndarray] = {}
        self._write_counts: dict[int, int] = {}

        self._n_inserts = 0
        self._n_updates = 0

    @property
    def D(self) -> int:
        return self._cam.D

    @property
    def K(self) -> int:
        return self._cam.K

    @property
    def update_threshold(self) -> float:
        return self._update_threshold

    @property
    def alpha(self) -> float:
        return self._alpha

    @property
    def n_inserts(self) -> int:
        return self._n_inserts

    @property
    def n_updates(self) -> int:
        return self._n_updates

    def __len__(self) -> int:
        return len(self._cam)

    def write(self, hv: SparseBinaryHV) -> int:
        """Consolidate ``hv`` into semantic memory. If similar enough to an
        existing prototype, update it via EMA; else insert a new node.
        Returns the node id written to."""
        if len(self._cam) > 0:
            results = self._cam.nearest(hv, k=1, track=False)
            if results and results[0][1] >= self._update_threshold:
                node_id = results[0][0]
                self._apply_ema_update(node_id, hv)
                self._n_updates += 1
                return node_id

        node_id = self._cam.insert(hv)
        counts = np.zeros(self._cam.D, dtype=np.float64)
        counts[hv.indices] = 1.0
        self._bit_counts[node_id] = counts
        self._write_counts[node_id] = 1
        self._n_inserts += 1
        return node_id

    def _apply_ema_update(self, node_id: int, new_hv: SparseBinaryHV) -> None:
        counts = self._bit_counts[node_id]
        counts *= 1.0 - self._alpha
        counts[new_hv.indices] += 1.0
        self._write_counts[node_id] += 1

        top_k = np.argpartition(counts, -self._cam.K)[-self._cam.K :]
        top_k = np.sort(top_k).astype(np.int32)
        new_prototype = SparseBinaryHV._unchecked(self._cam.D, top_k)
        self._cam._update_hv(node_id, new_prototype)

    def nearest(
        self,
        query: SparseBinaryHV,
        k: int = 1,
        ef: int | None = None,
        *,
        track: bool = True,
    ) -> list[tuple[int, float]]:
        return self._cam.nearest(query, k=k, ef=ef, track=track)

    def get(self, node_id: int) -> SparseBinaryHV:
        return self._cam.get(node_id)

    def compact(self) -> int:
        """Rebuild the underlying HNSW from live entries to reclaim
        space spent on soft-deleted routing hubs *and* refresh edges
        that became stale under EMA prototype drift. Preserves node_ids
        and access stats. Returns the number of nodes reclaimed.
        """
        return self._cam.compact()

    def write_count(self, node_id: int) -> int:
        """Number of observations consolidated into this prototype (initial
        insert counts as 1; each subsequent EMA update adds 1)."""
        return self._write_counts.get(node_id, 0)

    def drift_recall_at_1(
        self,
        probes: list[SparseBinaryHV],
        ground_truth_ids: list[int],
    ) -> float:
        """Fraction of probes whose top-1 nearest is the expected id.

        Use with a stable, held-out probe set across consolidation phases
        to track whether prototype drift degrades graph routing quality.
        Returns NaN for an empty probe set."""
        if len(probes) != len(ground_truth_ids):
            raise ValueError(
                f"probes/ground_truth_ids length mismatch: "
                f"{len(probes)} vs {len(ground_truth_ids)}"
            )
        if not probes:
            return float("nan")
        hits = 0
        for probe, gt in zip(probes, ground_truth_ids, strict=True):
            results = self._cam.nearest(probe, k=1, track=False)
            if results and results[0][0] == gt:
                hits += 1
        return hits / len(probes)
