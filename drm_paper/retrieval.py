"""Iterated retrieval mechanism + Remember/Know judgment.

Implements the §6.5 retrieval algorithm from DRM_PAPER_OUTLINE.md. The
iteration depth N_max is the primary FTT–AM manipulation:

- N_max = 1: single-step retrieval. Classical Fuzzy-Trace-Theory pattern —
  direct lures match semantic gist centroids; mediated lures (which are not
  themselves stored or close enough to be retrieved in one step) do not.
  Sharp cliff between direct and mediated lures.

- N_max ≥ 2: iterated spreading retrieval. Retrieved items bundle into the
  next-step query; mediated lures pick up evidence via the intermediate
  associate items retrieved in earlier steps. Activation-Monitoring pattern.

Both modes are the same algorithm with a different depth setting — same
architecture, two retrieval regimes. The empirical FTT/AM dissociation
emerges from the same machinery, not from two competing accounts.

Evidence accumulation tracks the maximum and the sum of overlap-with-
original-query across all retrieved items at all depths. Recognition
judgments threshold the maximum; R/K judgments use the dominant store.

Per the paradigm invariants: this module imports only from cgm/ (no
learned parameters), uses sparse-binary throughout, and exposes its
parameters (n_max, top_k_per_step, thresholds) as explicit knobs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from cgm.memory.episodic import EpisodicMemory
from cgm.memory.semantic import SemanticMemory
from cgm.vsa.hypervector import SparseBinaryHV
from cgm.vsa.operations import bundle, overlap

Judgment = Literal["Remember", "Know", "New"]


@dataclass(frozen=True)
class RetrievalResult:
    """Output of one iterated_retrieve call.

    Similarities are normalized to [0, 1] (overlap / K). The recognition
    decision thresholds best_overall_similarity; the R/K decision uses
    best_episodic_similarity (Remember requires an episodic match).

    Attributes:
        best_overall_similarity: max overlap-with-query across all retrieved
            items at all iteration depths, from either store.
        best_episodic_similarity: max overlap-with-query from episodic CAM only.
        best_semantic_similarity: max overlap-with-query from semantic CAM only.
        accumulated_evidence: sum of overlap-with-query across all retrieved
            items at all iteration depths. Grows with iteration depth; the
            AM-mode advantage on mediated lures lives here.
        n_iterations: actual iterations performed (may be < n_max if both
            stores empty out their useful retrievals).
    """

    best_overall_similarity: float
    best_episodic_similarity: float
    best_semantic_similarity: float
    accumulated_evidence: float
    n_iterations: int


@dataclass(frozen=True)
class RKJudgment:
    judgment: Judgment
    confidence: float
    best_episodic_similarity: float
    best_semantic_similarity: float


def iterated_retrieve(
    query: SparseBinaryHV,
    episodic: EpisodicMemory,
    semantic: SemanticMemory,
    *,
    n_max: int = 1,
    top_k_per_step: int = 5,
    rng: np.random.Generator,
) -> RetrievalResult:
    """Run iterated retrieval on the query against both memory stores.

    Args:
        query: encoded test item.
        episodic: episodic memory store.
        semantic: semantic memory store.
        n_max: iteration depth. 1 = FTT-mode single-step; ≥2 = AM-mode spreading.
        top_k_per_step: how many items to retrieve from each store per step.
        rng: random state for bundle's tie-breaking sparsification.

    Returns:
        RetrievalResult with best similarities by store, accumulated
        evidence, and actual iterations performed.
    """
    if n_max < 1:
        raise ValueError(f"n_max must be >= 1, got {n_max}")
    if top_k_per_step < 1:
        raise ValueError(f"top_k_per_step must be >= 1, got {top_k_per_step}")

    K = query.K
    activation_buffer = query

    best_overall = 0.0
    best_ep = 0.0
    best_sm = 0.0
    evidence = 0.0
    iterations_done = 0

    for i in range(n_max):
        iterations_done = i + 1
        retrieved_hvs: list[SparseBinaryHV] = []

        # Retrieve top-K from each store using the current activation buffer.
        # `nearest` returns (node_id, similarity) tuples sorted by similarity
        # descending. We then compute overlap-with-the-ORIGINAL-query so the
        # accumulated evidence is anchored on the test item, not on the
        # iterated activation buffer.
        if len(episodic) > 0:
            ep_hits = episodic.nearest(activation_buffer, k=min(top_k_per_step, len(episodic)))
            for node_id, _sim_to_buffer in ep_hits:
                hv = episodic.get(node_id)
                sim_to_query = overlap(hv, query) / K
                evidence += sim_to_query
                best_ep = max(best_ep, sim_to_query)
                best_overall = max(best_overall, sim_to_query)
                retrieved_hvs.append(hv)

        if len(semantic) > 0:
            sm_hits = semantic.nearest(activation_buffer, k=min(top_k_per_step, len(semantic)))
            for node_id, _sim_to_buffer in sm_hits:
                hv = semantic.get(node_id)
                sim_to_query = overlap(hv, query) / K
                evidence += sim_to_query
                best_sm = max(best_sm, sim_to_query)
                best_overall = max(best_overall, sim_to_query)
                retrieved_hvs.append(hv)

        # Nothing retrieved (both stores empty) — can't continue.
        if not retrieved_hvs:
            break

        # Form the next-step activation buffer by bundling retrieved items.
        # This is the attention-bounded spreading-activation step: instead of
        # letting activation spread to the entire stored graph, we constrain
        # it to the high-similarity region described by the retrieved items.
        if len(retrieved_hvs) == 1:
            activation_buffer = retrieved_hvs[0]
        else:
            activation_buffer = bundle(retrieved_hvs, K=K, rng=rng)

    return RetrievalResult(
        best_overall_similarity=best_overall,
        best_episodic_similarity=best_ep,
        best_semantic_similarity=best_sm,
        accumulated_evidence=evidence,
        n_iterations=iterations_done,
    )


def remember_know_judgment(
    result: RetrievalResult,
    *,
    threshold_recognition: float,
    threshold_episodic: float,
    recognition_signal: "Literal['best_similarity', 'accumulated_evidence']" = "best_similarity",
) -> RKJudgment:
    """Convert a retrieval result into a Remember/Know/New judgment.

    Args:
        result: the iterated_retrieve output.
        threshold_recognition: the cutoff used to decide "old" vs "New".
            Interpretation depends on ``recognition_signal``:
                - "best_similarity": compared against best_overall_similarity
                  (a depth-independent value in [0, 1]).
                - "accumulated_evidence": compared against accumulated_evidence
                  (a depth-DEPENDENT value that grows with N_max).
        threshold_episodic: cutoff for Remember vs Know, applied to
            ``best_episodic_similarity`` (a depth-independent value in [0, 1]).
        recognition_signal: which retrieval output drives the recognition
            decision. "best_similarity" (default) reproduces the
            depth-independent classical recognition rule; "accumulated_evidence"
            uses the spreading-activation-style continuous evidence signal,
            which grows with iteration depth and allows the substrate's
            recognition decisions to dissociate FTT-mode (N_max=1, evidence
            small) from AM-mode (N_max>=2, evidence accumulates above
            threshold for mediated lures and other indirectly-related items).

    Decision rule:
        if recognition_signal_value <= threshold_recognition: "New"
        elif best_episodic_similarity > threshold_episodic: "Remember"
        else: "Know"

    Predictions:
        - Studied associates: strong episodic match -> Remember.
        - Critical lures: weak episodic match, strong semantic match -> Know.
          (Canonical DRM finding: false memories accompanied by familiarity
          rather than recollection, Roediger & McDermott 1995.)
        - Distractors: weak match in both -> New.
        - Mediated lures under best_similarity recognition: stay below
          threshold regardless of N_max (single-step gist match fails).
        - Mediated lures under accumulated_evidence recognition: cross
          threshold at higher N_max as spreading retrieval accumulates
          activation through the cluster centroid. This produces the
          judgment-level FTT-AM dissociation.
    """
    if not (0.0 <= threshold_recognition):
        raise ValueError(f"threshold_recognition must be >= 0, got {threshold_recognition}")
    if not (0.0 <= threshold_episodic <= 1.0):
        raise ValueError(f"threshold_episodic must be in [0, 1], got {threshold_episodic}")

    if recognition_signal == "best_similarity":
        signal_value = result.best_overall_similarity
        if threshold_recognition > 1.0:
            raise ValueError(
                f"threshold_recognition must be in [0, 1] for best_similarity, "
                f"got {threshold_recognition}"
            )
    elif recognition_signal == "accumulated_evidence":
        signal_value = result.accumulated_evidence
    else:
        raise ValueError(
            f"recognition_signal must be 'best_similarity' or "
            f"'accumulated_evidence', got '{recognition_signal}'"
        )

    if signal_value <= threshold_recognition:
        judgment: Judgment = "New"
    elif result.best_episodic_similarity > threshold_episodic:
        judgment = "Remember"
    else:
        judgment = "Know"

    return RKJudgment(
        judgment=judgment,
        confidence=signal_value,
        best_episodic_similarity=result.best_episodic_similarity,
        best_semantic_similarity=result.best_semantic_similarity,
    )
