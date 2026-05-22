from __future__ import annotations

import numpy as np

from cgm.vsa.hypervector import SparseBinaryHV


def overlap(x: SparseBinaryHV, y: SparseBinaryHV) -> int:
    """Number of active bits shared between x and y."""
    if x.D != y.D:
        raise ValueError(f"dimension mismatch: {x.D} vs {y.D}")
    return int(np.intersect1d(x.indices, y.indices, assume_unique=True).size)


def overlap_batch(query: SparseBinaryHV, items: list[SparseBinaryHV]) -> np.ndarray:
    """Compute overlap(query, items[i]) for every i in one shot.

    Returns an ``int64`` array of length ``len(items)``. Empty input yields
    an empty array.

    Per the 2026-05-10 profile (open question #20 step (C)), the bulk of
    Phase 8 per-tick wallclock is spent in ``np.intersect1d`` called once
    per (query, neighbor) pair from HNSW search. The pairwise call has a
    NumPy floor of ~2 µs that dominates the actual algorithm cost at
    K=200. ``overlap_batch`` collapses N such calls into a single
    ``np.bitwise_count(query_packed & batch_packed)``, amortizing the
    NumPy dispatch overhead across the batch — measured 3.3x speedup at
    N=20 (0.63 µs effective vs 2.10 µs pairwise).

    Uses each HV's ``packed_bitmap`` cache. First batch query on a given
    HV pays ~O(D) packbits cost; subsequent queries reuse the cache.
    """
    n = len(items)
    if n == 0:
        return np.empty(0, dtype=np.int64)
    D = query.D
    qp = query.packed_bitmap  # shape ((D + 7) // 8,)
    # Stack the batch's packed bitmaps. Validation happens via shape
    # mismatch from np.stack if any item has a different D.
    batch = np.stack([item.packed_bitmap for item in items])  # (N, D_packed)
    if batch.shape[1] != qp.shape[0]:
        # Find the offending item for a clear error.
        for item in items:
            if item.D != D:
                raise ValueError(
                    f"dimension mismatch: query D={D}, item D={item.D}"
                )
        raise ValueError("packed_bitmap shape mismatch in overlap_batch")
    return np.bitwise_count(np.bitwise_and(qp, batch)).sum(axis=-1)


def similarity(x: SparseBinaryHV, y: SparseBinaryHV) -> float:
    """Normalized overlap in [0, 1]. Requires equal K."""
    if x.K != y.K:
        raise ValueError(f"sparsity mismatch: K={x.K} vs K={y.K}")
    if x.K == 0:
        return 0.0
    return overlap(x, y) / x.K


def random_permutation(D: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a random permutation of {0, ..., D-1} for use as a binding role."""
    return rng.permutation(D)


def inverse_permutation(perm: np.ndarray) -> np.ndarray:
    """Inverse of a permutation array: perm[inv] == arange(D)."""
    inv = np.empty_like(perm)
    inv[perm] = np.arange(perm.size)
    return inv


def permute(x: SparseBinaryHV, perm: np.ndarray) -> SparseBinaryHV:
    """Apply a permutation to the active indices of x, returning a new HV.

    Permutation binding is sparsity-preserving and exactly invertible via
    ``permute(result, inverse_permutation(perm))``.
    """
    if perm.shape != (x.D,):
        raise ValueError(f"permutation shape {perm.shape} does not match D={x.D}")
    new_indices = np.sort(perm[x.indices]).astype(np.int32)
    return SparseBinaryHV._unchecked(x.D, new_indices)


def bind(filler: SparseBinaryHV, role_perm: np.ndarray) -> SparseBinaryHV:
    """Bind a filler hypervector to a role, where the role is a permutation.

    Thin semantic wrapper over ``permute``. Use ``bind``/``unbind`` at call
    sites that express role-filler relationships; use ``permute`` directly
    for structural transforms (e.g. sequence position shifts)."""
    return permute(filler, role_perm)


def unbind(bound: SparseBinaryHV, role_perm: np.ndarray) -> SparseBinaryHV:
    """Recover the filler from a bound HV by applying the inverse role permutation."""
    return permute(bound, inverse_permutation(role_perm))


def bundle(
    vectors: list[SparseBinaryHV],
    K: int | None = None,
    *,
    rng: np.random.Generator,
) -> SparseBinaryHV:
    """Superposition of sparse binary HVs, sparsified to exactly K active bits.

    Per planning doc Section 4.2:
    - 1 vector: returned as-is (must already have K active bits if K given).
    - 2 vectors: union, then uniform-random sample of K indices from the union,
      padding from the non-union complement if the union is too small.
    - 3+ vectors: top-K indices by frequency across the inputs, with ties broken
      uniformly at random.

    Defaults K to ``vectors[0].K`` if not supplied.
    """
    if len(vectors) == 0:
        raise ValueError("cannot bundle empty vector list")

    D = vectors[0].D
    for v in vectors[1:]:
        if v.D != D:
            raise ValueError(f"dimension mismatch in bundle: {v.D} vs {D}")

    if K is None:
        K = vectors[0].K
    if K < 0 or K > D:
        raise ValueError(f"K must be in [0, {D}], got {K}")

    if len(vectors) == 1:
        if vectors[0].K != K:
            raise ValueError(
                f"single-vector bundle requires matching K: input K={vectors[0].K}, target K={K}"
            )
        return vectors[0].copy()

    all_idx = np.concatenate([v.indices for v in vectors])

    if len(vectors) == 2:
        union = np.unique(all_idx)
        if union.size >= K:
            result = np.sort(rng.choice(union, size=K, replace=False)).astype(np.int32)
        else:
            needed = K - union.size
            pool = np.setdiff1d(
                np.arange(D, dtype=np.int32), union, assume_unique=True
            )
            if pool.size < needed:
                raise ValueError(
                    f"cannot achieve K={K} from union — D={D} too small for required padding"
                )
            extra = rng.choice(pool, size=needed, replace=False)
            result = np.sort(np.concatenate([union.astype(np.int32), extra.astype(np.int32)]))
        return SparseBinaryHV._unchecked(D, result)

    # 3+ vectors: top-K by frequency, random tiebreak via pre-shuffle.
    vals, counts = np.unique(all_idx, return_counts=True)
    order = rng.permutation(vals.size)
    vals_shuf = vals[order]
    counts_shuf = counts[order]
    top_k_positions = np.argpartition(counts_shuf, -K)[-K:]
    result = np.sort(vals_shuf[top_k_positions]).astype(np.int32)
    return SparseBinaryHV._unchecked(D, result)
