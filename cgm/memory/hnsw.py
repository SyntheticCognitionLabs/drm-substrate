from __future__ import annotations

import heapq
import math
from collections.abc import Callable
from typing import Any

import numpy as np


class HNSW:
    """Hierarchical Navigable Small World approximate nearest neighbor index.

    Generic over item type and distance function (per Malkov & Yashunin 2018).
    Items are stored in the index; ``insert`` returns an integer node id;
    ``search`` returns ``(distance, node_id)`` pairs sorted ascending.

    The distance function must satisfy ``d(x, x) == 0`` and be non-negative.
    Metric-space triangle inequality is not required — common ANN practice.
    """

    def __init__(
        self,
        distance: Callable[[Any, Any], float],
        *,
        batch_distance: Callable[[Any, list[Any]], np.ndarray] | None = None,
        M: int = 16,
        ef_construction: int = 100,
        ef_search: int = 50,
        mL: float | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        if M < 2:
            raise ValueError(f"M must be >= 2, got {M}")
        if ef_construction < 1:
            raise ValueError(f"ef_construction must be >= 1, got {ef_construction}")
        if ef_search < 1:
            raise ValueError(f"ef_search must be >= 1, got {ef_search}")

        self._distance = distance
        # Optional batched-distance callable for the _search_layer hot
        # path. ``batch_distance(query, [items])`` must return a 1D array
        # of the same length as ``items``, with ``out[i] == distance(query,
        # items[i])`` to within float rounding. Used per open question
        # #20 step (C) — amortizes ~70% of pairwise-call NumPy overhead
        # by computing many overlaps in one ``np.bitwise_count`` call.
        # ``None`` keeps the per-pair fallback for non-batched call
        # sites and for HNSW instances constructed without this kwarg
        # (backward compat for existing tests).
        self._batch_distance = batch_distance
        self._M = M
        self._M_max = M
        self._M_max0 = 2 * M
        self._ef_construction = ef_construction
        self._ef_search = ef_search
        self._mL = mL if mL is not None else 1.0 / math.log(M)
        self._rng = rng if rng is not None else np.random.default_rng()

        self._items: dict[int, Any] = {}
        self._levels: dict[int, int] = {}
        self._neighbors: dict[int, list[set[int]]] = {}
        self._deleted: set[int] = set()
        self._entry_point: int | None = None
        self._top_level: int = -1
        self._next_id: int = 0

    def __len__(self) -> int:
        return len(self._items) - len(self._deleted)

    @property
    def total_nodes(self) -> int:
        """Count including soft-deleted nodes (for graph-size bookkeeping)."""
        return len(self._items)

    @property
    def entry_point(self) -> int | None:
        return self._entry_point

    @property
    def top_level(self) -> int:
        return self._top_level

    def get(self, node_id: int) -> Any:
        return self._items[node_id]

    def level_of(self, node_id: int) -> int:
        return self._levels[node_id]

    def neighbors(self, node_id: int, layer: int) -> set[int]:
        return self._neighbors[node_id][layer]

    def is_deleted(self, node_id: int) -> bool:
        return node_id in self._deleted

    def update_item(self, node_id: int, new_item: Any) -> None:
        """Replace the stored item for ``node_id`` without touching the graph.

        Used for prototype drift (EMA updates in semantic memory). Edges that
        were placed based on the old item become stale; graph quality must be
        measured separately (recall@k on held-out probes)."""
        if node_id not in self._items:
            raise KeyError(f"unknown node_id {node_id}")
        if node_id in self._deleted:
            raise KeyError(f"node_id {node_id} is deleted")
        self._items[node_id] = new_item

    def delete(self, node_id: int) -> None:
        """Soft-delete: node stays in the graph (as a routing hub) but is
        filtered from search results and counted out of ``len``. If the entry
        point is deleted, a replacement is promoted from remaining live nodes
        by max level.
        """
        if node_id not in self._items:
            raise KeyError(f"unknown node_id {node_id}")
        if node_id in self._deleted:
            return
        self._deleted.add(node_id)

        if self._entry_point == node_id:
            live = [nid for nid in self._items if nid not in self._deleted]
            if not live:
                self._entry_point = None
                self._top_level = -1
            else:
                best = max(live, key=lambda nid: self._levels[nid])
                self._entry_point = best
                self._top_level = self._levels[best]

    def _random_level(self) -> int:
        # Geometric distribution: P(level >= l) ≈ exp(-l / mL).
        u = max(float(self._rng.uniform()), 1e-12)
        return int(-math.log(u) * self._mL)

    def _search_layer(
        self,
        query: Any,
        entry_points: list[int],
        ef: int,
        layer: int,
    ) -> list[tuple[float, int]]:
        """Beam-search a single layer starting from ``entry_points``.

        Returns ``(distance, node_id)`` pairs sorted ascending by distance,
        length at most ``ef``.
        """
        visited: set[int] = set()
        candidates: list[tuple[float, int]] = []  # min-heap by distance
        results: list[tuple[float, int]] = []  # max-heap: stored as (-distance, id)

        for ep in entry_points:
            if ep in visited:
                continue
            visited.add(ep)
            d = self._distance(query, self._items[ep])
            heapq.heappush(candidates, (d, ep))
            heapq.heappush(results, (-d, ep))

        while candidates:
            d_c, c = heapq.heappop(candidates)
            # Stop when nearest unexplored is farther than worst in results.
            if results and d_c > -results[0][0]:
                break

            # Collect this node's unvisited neighbors first, then score them
            # in a single batched-distance call (when one is available).
            # Mirrors the original per-neighbor loop semantically; the heap
            # updates still run in order. Per open question #20 step (C),
            # the batched call amortizes ~70% of NumPy dispatch overhead.
            new_neighbors: list[int] = []
            for neighbor in self._neighbors[c][layer]:
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                new_neighbors.append(neighbor)

            if not new_neighbors:
                continue

            if self._batch_distance is not None and len(new_neighbors) > 1:
                items = [self._items[nid] for nid in new_neighbors]
                dists = self._batch_distance(query, items)
                neighbor_dists = zip(new_neighbors, (float(d) for d in dists))
            else:
                # Single neighbor — pairwise call is faster than the batch
                # path's stacking + bitwise_count overhead.
                neighbor_dists = (
                    (nid, self._distance(query, self._items[nid]))
                    for nid in new_neighbors
                )

            for neighbor, d_n in neighbor_dists:
                if len(results) < ef:
                    heapq.heappush(candidates, (d_n, neighbor))
                    heapq.heappush(results, (-d_n, neighbor))
                elif d_n < -results[0][0]:
                    heapq.heappush(candidates, (d_n, neighbor))
                    heapq.heappush(results, (-d_n, neighbor))
                    heapq.heappop(results)

        return sorted(((-nd, nid) for nd, nid in results), key=lambda t: t[0])

    def _select_neighbors(
        self,
        candidates: list[tuple[float, int]],
        M: int,
    ) -> list[int]:
        # Simple heuristic: M closest. candidates already sorted ascending.
        return [nid for _, nid in candidates[:M]]

    def _prune_neighbors(self, node_id: int, layer: int, M_max: int) -> None:
        """If node's neighbor list at layer exceeds M_max, keep only M_max closest."""
        nbrs = self._neighbors[node_id][layer]
        if len(nbrs) <= M_max:
            return
        item = self._items[node_id]
        nbr_list = list(nbrs)
        if self._batch_distance is not None and len(nbr_list) > 1:
            items = [self._items[n] for n in nbr_list]
            dists = self._batch_distance(item, items)
            scored = sorted(
                zip((float(d) for d in dists), nbr_list),
                key=lambda t: t[0],
            )
        else:
            scored = sorted(
                ((self._distance(item, self._items[n]), n) for n in nbr_list),
                key=lambda t: t[0],
            )
        self._neighbors[node_id][layer] = {n for _, n in scored[:M_max]}

    def insert(self, item: Any) -> int:
        node_id = self._next_id
        self._next_id += 1
        self._build_node(node_id, item)
        return node_id

    def _build_node(self, node_id: int, item: Any) -> None:
        """Place ``item`` under ``node_id`` and stitch it into the graph.

        Caller owns ``_next_id`` management. Used by ``insert`` (which
        allocates a fresh id) and by ``compact`` (which reuses the
        existing id).
        """
        level = self._random_level()
        self._items[node_id] = item
        self._levels[node_id] = level
        self._neighbors[node_id] = [set() for _ in range(level + 1)]

        if self._entry_point is None:
            self._entry_point = node_id
            self._top_level = level
            return

        ep = self._entry_point
        L = self._top_level

        # Phase 1: greedy descent through layers above the new node's level.
        for lc in range(L, level, -1):
            results = self._search_layer(item, [ep], ef=1, layer=lc)
            if results:
                ep = results[0][1]

        # Phase 2: insert with ef_construction at layers from min(L, level) down to 0.
        current_ep = [ep]
        for lc in range(min(L, level), -1, -1):
            results = self._search_layer(
                item, current_ep, ef=self._ef_construction, layer=lc
            )
            M_max = self._M_max0 if lc == 0 else self._M_max
            selected = self._select_neighbors(results, self._M)

            self._neighbors[node_id][lc].update(selected)
            for n in selected:
                self._neighbors[n][lc].add(node_id)
                self._prune_neighbors(n, lc, M_max)

            current_ep = [nid for _, nid in results]

        if level > L:
            self._entry_point = node_id
            self._top_level = level

    def compact(self) -> int:
        """Rebuild the graph from live items, dropping soft-deleted nodes
        and reconstructing fresh edges. Preserves node_ids of live entries.

        Phase 7 motivation (open question #22): under heavy eviction the
        HNSW retains soft-deleted routing hubs, slowing search as the
        live/deleted ratio worsens. ``compact`` is intended to be called
        periodically during sleep cycles to bound retrieval cost.

        Returns the number of nodes that were soft-deleted before the
        rebuild (i.e. how many got reclaimed). Live insertion order is
        preserved.
        """
        live: list[tuple[int, Any]] = [
            (nid, self._items[nid])
            for nid in self._items
            if nid not in self._deleted
        ]
        n_reclaimed = len(self._deleted)

        self._items.clear()
        self._levels.clear()
        self._neighbors.clear()
        self._deleted.clear()
        self._entry_point = None
        self._top_level = -1

        for node_id, item in live:
            self._build_node(node_id, item)

        return n_reclaimed

    def clear(self) -> None:
        """Drop all items and graph state. ``_next_id`` resets to 0 too —
        a cleared HNSW is indistinguishable from a freshly-constructed
        one. Used by Phase-7 body-death cleanup (episodic dies with the
        body)."""
        self._items.clear()
        self._levels.clear()
        self._neighbors.clear()
        self._deleted.clear()
        self._entry_point = None
        self._top_level = -1
        self._next_id = 0

    def search(
        self,
        query: Any,
        k: int = 1,
        ef: int | None = None,
    ) -> list[tuple[float, int]]:
        if self._entry_point is None:
            return []
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")

        ef = ef if ef is not None else max(self._ef_search, k)
        ep = self._entry_point

        for lc in range(self._top_level, 0, -1):
            results = self._search_layer(query, [ep], ef=1, layer=lc)
            if results:
                ep = results[0][1]

        # Layer-0 search. Deleted nodes remain as routing hubs but are
        # filtered from returned results. If the filtered set is short,
        # expand ef once to try to recover the shortfall.
        results = self._search_layer(query, [ep], ef=ef, layer=0)
        live = [(d, nid) for d, nid in results if nid not in self._deleted]
        if len(live) < k and self._deleted and ef < len(self._items):
            widened = min(ef * 2, len(self._items))
            results = self._search_layer(query, [ep], ef=widened, layer=0)
            live = [(d, nid) for d, nid in results if nid not in self._deleted]
        return live[:k]
