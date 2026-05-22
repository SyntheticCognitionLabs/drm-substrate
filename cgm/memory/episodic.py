from __future__ import annotations

import collections
from typing import Literal

import numpy as np

from cgm.memory.cam import CAM
from cgm.vsa.hypervector import SparseBinaryHV


class EpisodicMemory:
    """Novelty-gated, capacity-bounded CAM for one-shot episode storage.

    Per planning doc Section 4.4:
    - A write is accepted only if the similarity to the nearest existing node
      is below ``novelty_threshold``. Otherwise the write is discarded.
    - When ``capacity`` is reached, an accepted write triggers eviction of the
      oldest node (``eviction="fifo"``) or the least-recently-accessed node
      (``eviction="lru"``).
    - ``nearest`` delegates to the underlying CAM and drives access-count and
      last-access statistics used for replay prioritization and LRU eviction.

    Novelty checks use the CAM's ``track=False`` read so that policy queries
    do not pollute access statistics.
    """

    def __init__(
        self,
        D: int,
        K: int,
        *,
        novelty_threshold: float = 0.3,
        capacity: int = 5000,
        eviction: Literal["fifo", "lru"] = "fifo",
        M: int = 16,
        ef_construction: int = 100,
        ef_search: int = 50,
        rng: np.random.Generator | None = None,
    ) -> None:
        if not 0.0 <= novelty_threshold <= 1.0:
            raise ValueError(
                f"novelty_threshold must be in [0, 1], got {novelty_threshold}"
            )
        if capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {capacity}")
        if eviction not in ("fifo", "lru"):
            raise ValueError(f"eviction must be 'fifo' or 'lru', got {eviction!r}")

        self._cam = CAM(
            D=D,
            K=K,
            M=M,
            ef_construction=ef_construction,
            ef_search=ef_search,
            rng=rng,
        )
        self._novelty_threshold = novelty_threshold
        self._capacity = capacity
        self._eviction = eviction
        self._insertion_order: collections.deque[int] = collections.deque()

        self._n_accepted = 0
        self._n_rejected_novelty = 0
        self._n_evicted = 0

    @property
    def D(self) -> int:
        return self._cam.D

    @property
    def K(self) -> int:
        return self._cam.K

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def novelty_threshold(self) -> float:
        return self._novelty_threshold

    @property
    def eviction(self) -> str:
        return self._eviction

    @property
    def n_accepted(self) -> int:
        return self._n_accepted

    @property
    def n_rejected_novelty(self) -> int:
        return self._n_rejected_novelty

    @property
    def n_evicted(self) -> int:
        return self._n_evicted

    def __len__(self) -> int:
        return len(self._cam)

    def write(self, hv: SparseBinaryHV) -> int | None:
        """Attempt to store ``hv``. Returns the new node id on acceptance,
        or ``None`` if the novelty gate rejects it."""
        if len(self._cam) > 0:
            results = self._cam.nearest(hv, k=1, track=False)
            if results and results[0][1] >= self._novelty_threshold:
                self._n_rejected_novelty += 1
                return None

        if len(self._cam) >= self._capacity:
            self._evict_one()

        node_id = self._cam.insert(hv)
        self._insertion_order.append(node_id)
        self._n_accepted += 1
        return node_id

    def _evict_one(self) -> None:
        if self._eviction == "fifo":
            victim = self._insertion_order.popleft()
        else:
            victim = min(self._insertion_order, key=self._cam.last_access)
            self._insertion_order.remove(victim)
        self._cam.delete(victim)
        self._n_evicted += 1

    def evict(self, node_id: int) -> bool:
        """Evict a specific node id. Used by Phase-6 external budget
        policies that select victims by a custom score.

        Returns True if the node was present and removed; False if not
        in the insertion order (already evicted, or never existed).
        """
        if node_id not in self._insertion_order:
            return False
        self._insertion_order.remove(node_id)
        self._cam.delete(node_id)
        self._n_evicted += 1
        return True

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

    def access_count(self, node_id: int) -> int:
        return self._cam.access_count(node_id)

    def last_access(self, node_id: int) -> int:
        return self._cam.last_access(node_id)

    def live_ids(self) -> list[int]:
        """Currently-live node ids in insertion order (oldest first)."""
        return list(self._insertion_order)

    def compact(self) -> int:
        """Rebuild the underlying HNSW to reclaim space spent on
        soft-deleted routing hubs. Preserves node_ids and access stats.
        Used by Phase-7 sleep-cycle index maintenance (open question
        #22). Returns the number of nodes reclaimed.
        """
        return self._cam.compact()

    def clear(self) -> None:
        """Drop all entries and reset access statistics. Used by Phase-7
        body-death cleanup — episodic dies with the body, semantic
        persists. Lifetime counters (n_accepted, n_rejected_novelty,
        n_evicted) are *cumulative* across bodies and intentionally not
        reset, so substrate-lifetime stats remain interpretable.
        """
        self._cam.clear()
        self._insertion_order.clear()
