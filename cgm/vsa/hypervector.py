from __future__ import annotations

import numpy as np


class SparseBinaryHV:
    """Sparse binary hypervector with index-list storage.

    Stores a {0,1}^D vector with K active bits as a sorted, unique ``int32``
    array of active indices. Similarity and VSA operations run in O(K) rather
    than O(D). Instances are immutable — operations return new HVs.

    ``packed_bitmap`` (lazy, ~D/8 bytes) caches a dense uint8 packed
    representation for the batched-overlap fast path in
    ``cgm.vsa.operations.overlap_batch``. Trades a small memory increment for
    ~3x faster batched similarity via ``np.bitwise_count(query & batch)``.
    Only populated on first access; HVs that never participate in batched
    overlap stay at the original ~K*4 byte footprint.
    """

    __slots__ = ("_D", "_indices", "_packed")

    def __init__(self, D: int, indices: np.ndarray) -> None:
        if not isinstance(D, int) or D <= 0:
            raise ValueError(f"D must be a positive int, got {D!r}")

        arr = np.array(indices, dtype=np.int32, copy=True)
        if arr.ndim != 1:
            raise ValueError(f"indices must be 1D, got {arr.ndim}D")
        if arr.size > 0:
            if int(arr[0]) < 0 or int(arr[-1]) >= D:
                raise ValueError(
                    f"indices must be in [0, {D}), got range [{int(arr[0])}, {int(arr[-1])}]"
                )
            if arr.size > 1 and np.any(np.diff(arr) <= 0):
                raise ValueError("indices must be sorted ascending and unique")

        arr.flags.writeable = False
        self._D = D
        self._indices = arr
        self._packed: np.ndarray | None = None

    @classmethod
    def _unchecked(cls, D: int, indices: np.ndarray) -> SparseBinaryHV:
        """Construct without validation. Caller guarantees indices are sorted,
        unique, int32, and within [0, D). Internal fast path for operations
        that already produce well-formed output."""
        obj = object.__new__(cls)
        arr = np.ascontiguousarray(indices, dtype=np.int32)
        arr.flags.writeable = False
        obj._D = D
        obj._indices = arr
        obj._packed = None
        return obj

    @classmethod
    def random(cls, D: int, K: int, rng: np.random.Generator) -> SparseBinaryHV:
        if not isinstance(K, int) or K < 0 or K > D:
            raise ValueError(f"K must be an int in [0, {D}], got {K!r}")
        indices = np.sort(rng.choice(D, size=K, replace=False)).astype(np.int32)
        return cls._unchecked(D, indices)

    @property
    def D(self) -> int:
        return self._D

    @property
    def K(self) -> int:
        return int(self._indices.size)

    @property
    def indices(self) -> np.ndarray:
        return self._indices

    def to_dense(self) -> np.ndarray:
        out = np.zeros(self._D, dtype=np.uint8)
        out[self._indices] = 1
        return out

    @property
    def packed_bitmap(self) -> np.ndarray:
        """Lazy, cached packed-bit dense bitmap (uint8, shape ``((D + 7) // 8,)``).

        Used by ``cgm.vsa.operations.overlap_batch`` to do similarity in
        bulk via ``np.bitwise_count(query_packed & batch_packed)``. The
        cache is one-shot: computed on first access, kept for the HV's
        lifetime. HVs that never participate in a batched overlap call
        do not pay the ~D/8 bytes (1.25 KB at D=10000).
        """
        if self._packed is None:
            dense = np.zeros(self._D, dtype=np.uint8)
            dense[self._indices] = 1
            packed = np.packbits(dense)
            packed.flags.writeable = False
            self._packed = packed
        return self._packed

    def copy(self) -> SparseBinaryHV:
        return SparseBinaryHV._unchecked(self._D, self._indices.copy())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SparseBinaryHV):
            return NotImplemented
        return self._D == other._D and np.array_equal(self._indices, other._indices)

    def __hash__(self) -> int:
        return hash((self._D, self._indices.tobytes()))

    def __len__(self) -> int:
        return self.K

    def __repr__(self) -> str:
        return f"SparseBinaryHV(D={self._D}, K={self.K})"
