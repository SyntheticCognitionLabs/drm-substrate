"""Primary DRM experiment runner.

Wires encoder + stimuli + substrate + retrieval into one configurable
pipeline implementing §6 of DRM_PAPER_OUTLINE.md. This module runs ONE
configuration (one encoder, one parameter set, one seed); higher-level
battery runners (multi-seed, multi-encoder, multi-n_max) compose this.

Per-list flow:
  1. Encode all associates (encoder caches words automatically).
  2. Withhold ``held_out_per_list`` associates from study (these go into
     the test set as the gist-generalization probe; §6.7).
  3. Write studied associates to episodic memory in randomized order.
  4. Run a consolidation cycle: read each live episodic item, write to
     semantic memory (EMA-merge into a gist centroid for the list).
  5. Test phase items per list:
        - ``studied_test_per_list`` previously studied associates
        - 1 critical lure (never studied)
        - ``held_out_per_list`` held-out associates
        - ``distractors_per_list`` distractors drawn from OTHER lists'
          associates (unrelated to this list's gist)
  6. Each test item runs through ``iterated_retrieve`` and gets an
     R/K judgment.

Deferred to later runners: mediated lures, frontier 2D sweeps, P1-vs-P2
prior-knowledge manipulation, multi-seed batteries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from cgm.memory.episodic import EpisodicMemory
from cgm.memory.semantic import SemanticMemory

from drm_paper.encoding import WordEncoder
from drm_paper.retrieval import (
    RKJudgment,
    RetrievalResult,
    iterated_retrieve,
    remember_know_judgment,
)
from drm_paper.stimuli import DEFAULT_FOILS, DRMList

ItemType = Literal["studied", "critical_lure", "mediated_lure", "held_out", "distractor"]


@dataclass(frozen=True)
class ExperimentConfig:
    """One full set of substrate + retrieval parameters."""

    # Seeds. Each seed varies an independent source of randomness.
    list_order_seed: int = 0
    item_order_seed: int = 1
    distractor_sample_seed: int = 2
    substrate_seed: int = 3
    retrieval_seed: int = 4

    # Episodic memory
    novelty_threshold: float = 0.7
    episodic_capacity: int = 5000

    # Semantic memory
    update_threshold: float = 0.35
    ema_alpha: float = 0.2

    # Retrieval
    n_max: int = 1
    top_k_per_step: int = 5

    # R/K thresholds. Interpretation depends on recognition_signal:
    # - "best_similarity": threshold_recognition is in [0, 1] (similarity).
    # - "accumulated_evidence": threshold_recognition is unbounded above
    #   (depends on n_max and top_k_per_step); typical value 2-4 for
    #   ConceptNet encoder at default settings.
    threshold_recognition: float = 0.25
    threshold_episodic: float = 0.5

    # Which signal the recognition decision uses. "best_similarity"
    # (default) is depth-independent: best match across all stored items
    # against the test query. "accumulated_evidence" grows with N_max,
    # producing the judgment-level FTT-AM dissociation we report in §6.5.
    recognition_signal: "Literal['best_similarity', 'accumulated_evidence']" = "best_similarity"

    # Test composition per list
    studied_test_per_list: int = 6
    held_out_per_list: int = 1
    distractors_per_list: int = 6

    # Distractor pool. Must be words that the encoder can encode but that
    # are not studied during this experiment. Defaults to DEFAULT_FOILS
    # from stimuli.py — technical/scientific words unrelated to any canonical
    # DRM-list theme. Override when running with the full Stadler battery
    # or with non-canonical lists.
    foil_words: tuple[str, ...] = DEFAULT_FOILS

    # Ablation flags
    # When True, do not write episodic items into semantic memory during
    # study. Semantic memory stays empty throughout the experiment.
    # Effectively collapses ablations A (no consolidation) and B (no
    # semantic CAM) since both produce the same observable: only episodic
    # retrieval contributes to test-time recognition decisions. Per §6.6.
    disable_consolidation: bool = False


@dataclass(frozen=True)
class TrialResult:
    """One recognition trial: query the substrate on one test item."""

    list_index: int
    critical_lure: str
    item_type: ItemType
    item_word: str
    retrieval: RetrievalResult
    judgment: RKJudgment


@dataclass
class ExperimentResult:
    """All trial-level data from one experiment run."""

    config: ExperimentConfig
    encoder_name: str
    n_lists: int
    trials: list[TrialResult] = field(default_factory=list)

    def trials_of_type(self, item_type: ItemType) -> list[TrialResult]:
        return [t for t in self.trials if t.item_type == item_type]


def _build_distractor_pool(
    drm_lists: tuple[DRMList, ...],
    foil_words: tuple[str, ...],
) -> list[str]:
    """Return the foil-word pool, sanity-checked to not collide with any
    studied vocabulary.

    Distractors must be (a) never studied during this experiment and (b)
    unrelated to any studied gist. The foil pool is shared across lists
    (a foil word is just as unrelated to list 0 as to list 5). Per-list
    sampling without replacement happens at trial-construction time.
    """
    studied_vocab: set[str] = set()
    for dl in drm_lists:
        studied_vocab.update(dl.associates)
        studied_vocab.add(dl.critical_lure)

    collisions = sorted(set(foil_words) & studied_vocab)
    if collisions:
        raise ValueError(
            f"foil_words collide with studied vocabulary: {collisions}. "
            "Foils must not appear as a critical lure or associate in any "
            "DRMList; choose foils from genuinely unrelated semantic domains."
        )
    if len(foil_words) == 0:
        raise ValueError("foil_words pool is empty")

    return list(foil_words)


def run_drm_experiment(
    drm_lists: tuple[DRMList, ...],
    encoder: WordEncoder,
    config: ExperimentConfig | None = None,
    *,
    encoder_name: str = "unknown",
) -> ExperimentResult:
    """Run the full primary DRM experiment for one configuration.

    Args:
        drm_lists: tuple of DRMList stimuli (from stimuli.canonical_lists()
            or stimuli.load_stadler_full()).
        encoder: a WordEncoder with .D, .K, .encode(word) -> SparseBinaryHV.
        config: ExperimentConfig; defaults to all-defaults if None.
        encoder_name: human-readable label for the encoder, stored in the
            result for downstream analysis. Pass e.g. "random_orthogonal"
            or "conceptnet_numberbatch".

    Returns:
        ExperimentResult with one TrialResult per (list, test item).
    """
    if config is None:
        config = ExperimentConfig()

    # Per-seed RNGs. Substrate consumes one; retrieval consumes another;
    # ordering / sampling consume their own. Keeping them independent
    # makes seed effects interpretable.
    list_order_rng = np.random.default_rng(config.list_order_seed)
    item_order_rng = np.random.default_rng(config.item_order_seed)
    distractor_rng = np.random.default_rng(config.distractor_sample_seed)
    substrate_rng = np.random.default_rng(config.substrate_seed)
    retrieval_rng = np.random.default_rng(config.retrieval_seed)

    D, K = encoder.D, encoder.K

    episodic = EpisodicMemory(
        D=D, K=K,
        novelty_threshold=config.novelty_threshold,
        capacity=config.episodic_capacity,
        rng=substrate_rng,
    )
    semantic = SemanticMemory(
        D=D, K=K,
        update_threshold=config.update_threshold,
        alpha=config.ema_alpha,
        rng=substrate_rng,
    )

    distractor_pool = _build_distractor_pool(drm_lists, config.foil_words)

    # Plan the study phase: randomize list presentation order, randomize
    # within-list item order, and pick which associates to withhold for
    # gist-generalization testing.
    list_indices = np.arange(len(drm_lists))
    list_order_rng.shuffle(list_indices)

    # Per-list: which associates are studied vs held out
    studied_associates: dict[int, list[str]] = {}
    held_out_associates: dict[int, list[str]] = {}
    for i in list_indices:
        dl = drm_lists[i]
        associates = list(dl.associates)
        item_order_rng.shuffle(associates)
        held_out = associates[: config.held_out_per_list]
        studied = associates[config.held_out_per_list :]
        # Reshuffle the studied set so presentation order doesn't track
        # held-out-vs-studied selection.
        item_order_rng.shuffle(studied)
        studied_associates[int(i)] = studied
        held_out_associates[int(i)] = held_out

    # Study phase + (optional) per-list consolidation.
    for i in list_indices:
        for word in studied_associates[int(i)]:
            episodic.write(encoder.encode(word))

        if not config.disable_consolidation:
            # Replay-driven consolidation: read every currently-live episodic
            # trace and write it to semantic. The substrate's EMA-merged
            # semantic write builds list-specific gist centroids.
            for node_id in episodic.live_ids():
                semantic.write(episodic.get(node_id))

    # Test phase.
    trials: list[TrialResult] = []
    for i in list_indices:
        dl = drm_lists[int(i)]

        # Studied test items: a random subset of what was actually studied.
        studied_pool = studied_associates[int(i)]
        n_studied_test = min(config.studied_test_per_list, len(studied_pool))
        studied_test = distractor_rng.choice(
            studied_pool, size=n_studied_test, replace=False
        )

        # Distractors: random sample (without replacement) from the foil pool.
        # Sampling is per-list so different lists' tests don't share foils.
        n_dist = min(config.distractors_per_list, len(distractor_pool))
        distractors = distractor_rng.choice(distractor_pool, size=n_dist, replace=False)

        test_items: list[tuple[ItemType, str]] = []
        for w in studied_test:
            test_items.append(("studied", str(w)))
        test_items.append(("critical_lure", dl.critical_lure))
        if dl.mediated_lure is not None:
            test_items.append(("mediated_lure", dl.mediated_lure))
        for w in held_out_associates[int(i)]:
            test_items.append(("held_out", w))
        for w in distractors:
            test_items.append(("distractor", str(w)))

        for item_type, word in test_items:
            query = encoder.encode(word)
            result = iterated_retrieve(
                query, episodic, semantic,
                n_max=config.n_max,
                top_k_per_step=config.top_k_per_step,
                rng=retrieval_rng,
            )
            judgment = remember_know_judgment(
                result,
                threshold_recognition=config.threshold_recognition,
                threshold_episodic=config.threshold_episodic,
                recognition_signal=config.recognition_signal,
            )
            trials.append(
                TrialResult(
                    list_index=int(i),
                    critical_lure=dl.critical_lure,
                    item_type=item_type,
                    item_word=word,
                    retrieval=result,
                    judgment=judgment,
                )
            )

    return ExperimentResult(
        config=config,
        encoder_name=encoder_name,
        n_lists=len(drm_lists),
        trials=trials,
    )
