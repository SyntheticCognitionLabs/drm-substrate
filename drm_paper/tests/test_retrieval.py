"""Tests for drm_paper.retrieval."""

from __future__ import annotations

import numpy as np
import pytest

from cgm.memory.episodic import EpisodicMemory
from cgm.memory.semantic import SemanticMemory
from cgm.vsa.hypervector import SparseBinaryHV

from drm_paper.retrieval import (
    RetrievalResult,
    iterated_retrieve,
    remember_know_judgment,
)

D = 2000
K = 64


def _random_hv(D: int, K: int, rng: np.random.Generator) -> SparseBinaryHV:
    indices = np.sort(rng.choice(D, K, replace=False)).astype(np.int32)
    return SparseBinaryHV(D=D, indices=indices)


def _perturb(base: SparseBinaryHV, drop_frac: float, rng: np.random.Generator) -> SparseBinaryHV:
    n_drop = int(base.K * drop_frac)
    n_keep = base.K - n_drop
    kept = rng.choice(base.indices, size=n_keep, replace=False)
    non_base = np.setdiff1d(
        np.arange(base.D, dtype=np.int32), base.indices, assume_unique=True
    )
    added = rng.choice(non_base, size=n_drop, replace=False)
    new = np.sort(np.concatenate([kept, added])).astype(np.int32)
    return SparseBinaryHV(D=base.D, indices=new)


def _setup_drm_memory(seed: int) -> tuple[
    EpisodicMemory, SemanticMemory, SparseBinaryHV, SparseBinaryHV, SparseBinaryHV
]:
    """Build a minimal DRM-shaped memory: 6 associates around a concept base,
    plus a lure (unstudied perturbation) and a distractor (unrelated).
    """
    rng = np.random.default_rng(seed)
    base = _random_hv(D, K, rng)
    associates = [_perturb(base, drop_frac=0.35, rng=rng) for _ in range(6)]
    lure = _perturb(base, drop_frac=0.35, rng=rng)
    distractor = _random_hv(D, K, rng)

    episodic = EpisodicMemory(D=D, K=K, novelty_threshold=0.7, capacity=100, rng=rng)
    semantic = SemanticMemory(D=D, K=K, update_threshold=0.35, alpha=0.2, rng=rng)

    for a in associates:
        episodic.write(a)
    for node_id in episodic.live_ids():
        semantic.write(episodic.get(node_id))

    studied_probe = associates[0]
    return episodic, semantic, studied_probe, lure, distractor


class TestSingleStep:
    def test_basic_retrieval_runs(self) -> None:
        episodic, semantic, studied, lure, distractor = _setup_drm_memory(seed=42)
        rng = np.random.default_rng(7)
        result = iterated_retrieve(
            lure, episodic, semantic, n_max=1, top_k_per_step=3, rng=rng
        )
        assert isinstance(result, RetrievalResult)
        assert result.n_iterations == 1
        assert 0.0 <= result.best_overall_similarity <= 1.0
        assert 0.0 <= result.best_episodic_similarity <= 1.0
        assert 0.0 <= result.best_semantic_similarity <= 1.0
        assert result.accumulated_evidence > 0.0

    def test_lure_more_similar_than_distractor(self) -> None:
        episodic, semantic, _studied, lure, distractor = _setup_drm_memory(seed=42)
        rng = np.random.default_rng(7)
        lure_result = iterated_retrieve(
            lure, episodic, semantic, n_max=1, top_k_per_step=3, rng=rng
        )
        distractor_result = iterated_retrieve(
            distractor, episodic, semantic, n_max=1, top_k_per_step=3, rng=rng
        )
        assert lure_result.best_overall_similarity > distractor_result.best_overall_similarity
        assert lure_result.accumulated_evidence > distractor_result.accumulated_evidence

    def test_studied_has_strong_episodic_match(self) -> None:
        episodic, semantic, studied, lure, _distractor = _setup_drm_memory(seed=42)
        rng = np.random.default_rng(7)
        studied_result = iterated_retrieve(
            studied, episodic, semantic, n_max=1, top_k_per_step=3, rng=rng
        )
        lure_result = iterated_retrieve(
            lure, episodic, semantic, n_max=1, top_k_per_step=3, rng=rng
        )
        # Studied probes match an actual episodic trace; lures don't.
        assert studied_result.best_episodic_similarity > lure_result.best_episodic_similarity


class TestIterationDepth:
    def test_more_iterations_accumulate_more_evidence(self) -> None:
        """Multi-step retrieval should produce strictly more accumulated
        evidence than single-step, because each step adds another batch of
        retrieved similarities to the running sum.
        """
        episodic, semantic, _studied, lure, _distractor = _setup_drm_memory(seed=42)

        rng1 = np.random.default_rng(7)
        result_1 = iterated_retrieve(
            lure, episodic, semantic, n_max=1, top_k_per_step=3, rng=rng1
        )

        rng3 = np.random.default_rng(7)
        result_3 = iterated_retrieve(
            lure, episodic, semantic, n_max=3, top_k_per_step=3, rng=rng3
        )

        assert result_3.accumulated_evidence > result_1.accumulated_evidence
        assert result_3.n_iterations == 3

    def test_best_similarity_monotone_non_decreasing(self) -> None:
        """Best-overall-similarity should not decrease with more iterations
        (later iterations can only find better or equal matches).
        """
        episodic, semantic, _studied, lure, _distractor = _setup_drm_memory(seed=42)

        rng1 = np.random.default_rng(7)
        r1 = iterated_retrieve(
            lure, episodic, semantic, n_max=1, top_k_per_step=3, rng=rng1
        )
        rng2 = np.random.default_rng(7)
        r2 = iterated_retrieve(
            lure, episodic, semantic, n_max=2, top_k_per_step=3, rng=rng2
        )
        assert r2.best_overall_similarity >= r1.best_overall_similarity


class TestEmptyMemories:
    def test_empty_stores_returns_zeros(self) -> None:
        rng = np.random.default_rng(0)
        episodic = EpisodicMemory(D=D, K=K, novelty_threshold=0.7, capacity=10, rng=rng)
        semantic = SemanticMemory(D=D, K=K, update_threshold=0.35, alpha=0.2, rng=rng)

        query = _random_hv(D, K, rng)
        result = iterated_retrieve(
            query, episodic, semantic, n_max=3, top_k_per_step=3, rng=rng
        )
        assert result.best_overall_similarity == 0.0
        assert result.accumulated_evidence == 0.0
        # n_iterations should be 1 because the loop runs once, finds nothing, breaks.
        assert result.n_iterations == 1


class TestRKJudgment:
    def test_remember_when_episodic_strong(self) -> None:
        # Hand-construct a result with strong episodic match
        result = RetrievalResult(
            best_overall_similarity=0.85,
            best_episodic_similarity=0.85,
            best_semantic_similarity=0.4,
            accumulated_evidence=2.0,
            n_iterations=1,
        )
        judgment = remember_know_judgment(
            result, threshold_recognition=0.3, threshold_episodic=0.5
        )
        assert judgment.judgment == "Remember"

    def test_know_when_semantic_only(self) -> None:
        # Weak episodic, strong semantic — the canonical DRM lure signature
        result = RetrievalResult(
            best_overall_similarity=0.7,
            best_episodic_similarity=0.2,
            best_semantic_similarity=0.7,
            accumulated_evidence=1.5,
            n_iterations=1,
        )
        judgment = remember_know_judgment(
            result, threshold_recognition=0.3, threshold_episodic=0.5
        )
        assert judgment.judgment == "Know"

    def test_new_when_below_recognition(self) -> None:
        result = RetrievalResult(
            best_overall_similarity=0.1,
            best_episodic_similarity=0.05,
            best_semantic_similarity=0.1,
            accumulated_evidence=0.2,
            n_iterations=1,
        )
        judgment = remember_know_judgment(
            result, threshold_recognition=0.3, threshold_episodic=0.5
        )
        assert judgment.judgment == "New"

    def test_invalid_threshold_raises(self) -> None:
        result = RetrievalResult(
            best_overall_similarity=0.5,
            best_episodic_similarity=0.5,
            best_semantic_similarity=0.5,
            accumulated_evidence=1.0,
            n_iterations=1,
        )
        with pytest.raises(ValueError, match="threshold_recognition"):
            remember_know_judgment(
                result, threshold_recognition=1.5, threshold_episodic=0.5
            )
        with pytest.raises(ValueError, match="threshold_episodic"):
            remember_know_judgment(
                result, threshold_recognition=0.3, threshold_episodic=-0.1
            )


class TestParameterValidation:
    def test_invalid_n_max(self) -> None:
        rng = np.random.default_rng(0)
        episodic = EpisodicMemory(D=D, K=K, novelty_threshold=0.7, capacity=10, rng=rng)
        semantic = SemanticMemory(D=D, K=K, update_threshold=0.35, alpha=0.2, rng=rng)
        query = _random_hv(D, K, rng)
        with pytest.raises(ValueError, match="n_max"):
            iterated_retrieve(
                query, episodic, semantic, n_max=0, top_k_per_step=3, rng=rng
            )

    def test_invalid_top_k(self) -> None:
        rng = np.random.default_rng(0)
        episodic = EpisodicMemory(D=D, K=K, novelty_threshold=0.7, capacity=10, rng=rng)
        semantic = SemanticMemory(D=D, K=K, update_threshold=0.35, alpha=0.2, rng=rng)
        query = _random_hv(D, K, rng)
        with pytest.raises(ValueError, match="top_k_per_step"):
            iterated_retrieve(
                query, episodic, semantic, n_max=1, top_k_per_step=0, rng=rng
            )
