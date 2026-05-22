"""Smoke test for the DRM paper substrate pipeline.

Verifies the minimal architectural recipe — sparse-binary encoding with
semantic structure, novelty-gated episodic CAM, EMA-merged semantic CAM —
produces the core DRM-emergence pattern:

  - studied associates with high mutual overlap consolidate to a semantic
    centroid that captures the unstudied "concept base"
  - an unstudied lure (independent perturbation of the same base) matches
    the centroid strongly
  - an unrelated distractor does not match the centroid

This is not yet a quantitative DRM replication. It is a minimum-viable
demonstration that the substrate API is usable for our experiments and
that the core architectural claim works end-to-end before we build
encoders, stimuli, or retrieval modules.
"""

from __future__ import annotations

import numpy as np

from cgm.memory.episodic import EpisodicMemory
from cgm.memory.semantic import SemanticMemory
from cgm.vsa.hypervector import SparseBinaryHV

D = 2000
K = 64


def _random_hv(D: int, K: int, rng: np.random.Generator) -> SparseBinaryHV:
    indices = np.sort(rng.choice(D, K, replace=False)).astype(np.int32)
    return SparseBinaryHV(D=D, indices=indices)


def _perturb(base: SparseBinaryHV, drop_frac: float, rng: np.random.Generator) -> SparseBinaryHV:
    """Drop drop_frac of base's bits, replace with random non-base bits.

    Result has similarity (1 - drop_frac) to base.
    """
    n_drop = int(base.K * drop_frac)
    n_keep = base.K - n_drop
    kept = rng.choice(base.indices, size=n_keep, replace=False)

    non_base = np.setdiff1d(
        np.arange(base.D, dtype=np.int32), base.indices, assume_unique=True
    )
    added = rng.choice(non_base, size=n_drop, replace=False)

    new_indices = np.sort(np.concatenate([kept, added])).astype(np.int32)
    return SparseBinaryHV(D=base.D, indices=new_indices)


def test_drm_emergence_pattern() -> None:
    """Consolidation of related associates produces a semantic centroid that
    an unstudied related lure matches more strongly than an unrelated distractor.
    """
    rng = np.random.default_rng(seed=42)

    # The "concept" — never directly studied. Substrate analogue of "doctor"
    # given a study list of {nurse, hospital, medicine, ...}.
    concept_base = _random_hv(D, K, rng)

    # Six associates — independent perturbations of the base
    associates = [_perturb(concept_base, drop_frac=0.35, rng=rng) for _ in range(6)]

    # Critical lure — another independent perturbation of the same base.
    # Same construction as associates but never studied.
    lure = _perturb(concept_base, drop_frac=0.35, rng=rng)

    # Distractor — independent random vector, no overlap with the concept
    distractor = _random_hv(D, K, rng)

    # Memory components. Thresholds are normalized similarity (overlap/K).
    episodic = EpisodicMemory(
        D=D, K=K, novelty_threshold=0.7, capacity=100, rng=rng
    )
    semantic = SemanticMemory(
        D=D, K=K, update_threshold=0.35, alpha=0.2, rng=rng
    )

    # Study phase
    n_accepted = 0
    for a in associates:
        if episodic.write(a) is not None:
            n_accepted += 1
    assert n_accepted >= 5, f"expected most associates accepted, got {n_accepted}/6"

    # Consolidation: replay episodic items into semantic
    for node_id in episodic.live_ids():
        semantic.write(episodic.get(node_id))

    assert len(semantic) >= 1, "semantic memory should hold a consolidated centroid"

    # Test phase: query lure and distractor against the consolidated semantic store
    lure_sim = semantic.nearest(lure, k=1)[0][1]
    distractor_sim = semantic.nearest(distractor, k=1)[0][1]

    # Core DRM-emergence claim
    assert lure_sim > distractor_sim, (
        f"DRM emergence failed: lure similarity {lure_sim:.3f} "
        f"should exceed distractor similarity {distractor_sim:.3f}"
    )

    # The lure should match the consolidated centroid substantially.
    # With drop_frac=0.35, the lure shares ~65% of its bits with the base,
    # and the consolidated centroid approximates the base. So similarity
    # should be well above the ~K/D ≈ 0.03 chance level.
    assert lure_sim >= 0.4, (
        f"lure similarity to consolidated centroid ({lure_sim:.3f}) "
        f"should be substantial (>= 0.4)"
    )


def test_episodic_vs_semantic_match_dissociation() -> None:
    """R/K-precursor: a studied associate matches an episodic trace strongly
    (Remember-like); a lure matches the semantic centroid but not any episodic
    trace (Know-like). This is the architectural basis for the R/K dissociation
    we will formalize in retrieval.py.
    """
    rng = np.random.default_rng(seed=7)

    concept_base = _random_hv(D, K, rng)
    associates = [_perturb(concept_base, drop_frac=0.35, rng=rng) for _ in range(6)]
    lure = _perturb(concept_base, drop_frac=0.35, rng=rng)

    episodic = EpisodicMemory(
        D=D, K=K, novelty_threshold=0.7, capacity=100, rng=rng
    )
    semantic = SemanticMemory(
        D=D, K=K, update_threshold=0.35, alpha=0.2, rng=rng
    )

    for a in associates:
        episodic.write(a)
    for node_id in episodic.live_ids():
        semantic.write(episodic.get(node_id))

    studied_probe = associates[0]

    studied_episodic_sim = episodic.nearest(studied_probe, k=1)[0][1]
    lure_episodic_sim = episodic.nearest(lure, k=1)[0][1]
    lure_semantic_sim = semantic.nearest(lure, k=1)[0][1]

    # Studied probes have a literal episodic trace; the lure does not.
    assert studied_episodic_sim > lure_episodic_sim, (
        f"studied probe should match episodic more than lure does "
        f"(studied={studied_episodic_sim:.3f}, lure={lure_episodic_sim:.3f})"
    )

    # Lure matches semantic centroid at least as well as it matches any
    # individual episodic trace — the structural basis of Know responses.
    assert lure_semantic_sim >= lure_episodic_sim, (
        f"lure should match semantic centroid at least as well as any episodic "
        f"trace (sm={lure_semantic_sim:.3f}, ep={lure_episodic_sim:.3f})"
    )
