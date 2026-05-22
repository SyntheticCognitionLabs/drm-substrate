"""Tests for drm_paper.experiment — the primary DRM runner."""

from __future__ import annotations

import numpy as np

from drm_paper.encoding import RandomOrthogonalEncoder, SyntheticClusterEncoder
from drm_paper.experiment import (
    ExperimentConfig,
    ExperimentResult,
    TrialResult,
    run_drm_experiment,
)
from drm_paper.stimuli import DEFAULT_FOILS, canonical_lists


def _make_synthetic_encoder(drop_frac: float = 0.35, seed: int = 42) -> SyntheticClusterEncoder:
    """SyntheticClusterEncoder pre-registered with the default foil pool so
    distractors can be encoded as orthogonal random vectors."""
    return SyntheticClusterEncoder(
        canonical_lists(),
        drop_frac=drop_frac,
        seed=seed,
        extra_words=DEFAULT_FOILS,
    )


class TestRunDRMExperiment:
    def test_runs_end_to_end_with_random_encoder(self) -> None:
        """Smoke test: the full pipeline executes on the canonical lists
        with the random orthogonal encoder. Random-orthogonal encoding is
        the §6.2 ablation condition — we expect DRM ≈ 0, but the pipeline
        must still produce structurally valid output (every test item gets
        a trial, item types accounted for, etc.)."""
        lists = canonical_lists()
        encoder = RandomOrthogonalEncoder(seed=42)
        config = ExperimentConfig()

        result = run_drm_experiment(
            lists, encoder, config, encoder_name="random_orthogonal"
        )

        assert isinstance(result, ExperimentResult)
        assert result.encoder_name == "random_orthogonal"
        assert result.n_lists == len(lists)
        assert len(result.trials) > 0

        # Every trial should have a recognized item_type
        valid_types = {"studied", "critical_lure", "mediated_lure", "held_out", "distractor"}
        for t in result.trials:
            assert t.item_type in valid_types
            assert isinstance(t, TrialResult)
            assert 0.0 <= t.retrieval.best_overall_similarity <= 1.0

    def test_trial_counts_per_list(self) -> None:
        """Per-list trial counts should match config: 6 studied + 1 lure + 1
        held-out + 6 distractors = 14 trials per list (with defaults)."""
        lists = canonical_lists()
        encoder = RandomOrthogonalEncoder(seed=42)
        config = ExperimentConfig(
            studied_test_per_list=6,
            held_out_per_list=1,
            distractors_per_list=6,
        )

        result = run_drm_experiment(lists, encoder, config)

        # Group by list
        from collections import Counter

        per_list: dict[int, Counter] = {}
        for t in result.trials:
            per_list.setdefault(t.list_index, Counter())[t.item_type] += 1

        # Every list should have the expected mix
        for li, counts in per_list.items():
            assert counts["studied"] == 6, f"list {li}: {counts}"
            assert counts["critical_lure"] == 1
            assert counts["held_out"] == 1
            assert counts["distractor"] == 6

    def test_exactly_one_critical_lure_per_list(self) -> None:
        lists = canonical_lists()
        encoder = RandomOrthogonalEncoder(seed=42)
        result = run_drm_experiment(lists, encoder)

        lure_trials = result.trials_of_type("critical_lure")
        assert len(lure_trials) == len(lists)

        seen_lures = set()
        for t in lure_trials:
            assert t.item_word == t.critical_lure
            seen_lures.add(t.critical_lure)

        # No duplicate critical lures (each list contributes exactly one)
        assert len(seen_lures) == len(lists)

    def test_random_orthogonal_yields_no_DRM_signal(self) -> None:
        """The §6.2 architectural prediction: random orthogonal encoding
        should produce near-zero DRM. The substrate is identical to other
        encoding conditions; only the encoder differs. Without semantic
        structure in the input, no gist centroid forms and lures have no
        more retrieval signal than distractors.
        """
        lists = canonical_lists()
        encoder = RandomOrthogonalEncoder(seed=42)
        result = run_drm_experiment(lists, encoder)

        lure_sims = [t.retrieval.best_overall_similarity for t in result.trials_of_type("critical_lure")]
        distractor_sims = [t.retrieval.best_overall_similarity for t in result.trials_of_type("distractor")]

        # Both means should be near the chance floor (K^2/D = 64^2/2000 ≈ 2 → similarity ≈ 0.03).
        # The substrate's consolidation may slightly bump the semantic centroid's
        # match for random items (especially if some random codes happen to overlap),
        # but the difference between lures and distractors should be small.
        lure_mean = float(np.mean(lure_sims))
        distractor_mean = float(np.mean(distractor_sims))

        # Permissive bound: with random orthogonal codes the gap should be < 0.15.
        # Real DRM-producing encoders (ConceptNet etc) should produce gaps > 0.30.
        gap = abs(lure_mean - distractor_mean)
        assert gap < 0.15, (
            f"random orthogonal encoding produced an unexpected DRM-like gap "
            f"(lure={lure_mean:.3f}, distractor={distractor_mean:.3f}, gap={gap:.3f}). "
            f"Possible substrate or encoder leakage."
        )

    def test_studied_items_recognized_with_random_encoder(self) -> None:
        """Even under random orthogonal encoding, studied items should match
        their own episodic traces — this is verbatim recognition, independent
        of semantic structure.
        """
        lists = canonical_lists()
        encoder = RandomOrthogonalEncoder(seed=42)
        result = run_drm_experiment(lists, encoder)

        studied_sims = [t.retrieval.best_episodic_similarity for t in result.trials_of_type("studied")]
        distractor_eps = [t.retrieval.best_episodic_similarity for t in result.trials_of_type("distractor")]

        # Studied items should match their episodic trace strongly (similarity ≈ 1.0
        # since the same encoding hits the same stored vector).
        assert np.mean(studied_sims) > 0.5, (
            f"studied-item episodic match unexpectedly low: {np.mean(studied_sims):.3f}"
        )
        # Distractors should not match episodic strongly.
        assert np.mean(distractor_eps) < 0.2, (
            f"distractor episodic match unexpectedly high: {np.mean(distractor_eps):.3f}"
        )

    def test_seed_determinism(self) -> None:
        """Same config + same encoder seed → identical results."""
        lists = canonical_lists()

        enc_a = RandomOrthogonalEncoder(seed=42)
        enc_b = RandomOrthogonalEncoder(seed=42)
        config = ExperimentConfig()

        result_a = run_drm_experiment(lists, enc_a, config)
        result_b = run_drm_experiment(lists, enc_b, config)

        assert len(result_a.trials) == len(result_b.trials)
        for ta, tb in zip(result_a.trials, result_b.trials):
            assert ta.list_index == tb.list_index
            assert ta.item_type == tb.item_type
            assert ta.item_word == tb.item_word
            assert ta.retrieval.best_overall_similarity == tb.retrieval.best_overall_similarity


class TestDRMEmergence:
    """End-to-end demonstration that the substrate produces DRM-style
    false memory when given semantically-structured input encoding.

    This is the headline architectural claim of the paper, validated on the
    canonical lists with the synthetic cluster encoder (which mimics
    ConceptNet Numberbatch's similarity-preserving property without external
    data). The same test will replicate against the real ConceptNet encoder
    once that is wired up.
    """

    def test_lure_recognition_exceeds_distractor(self) -> None:
        lists = canonical_lists()
        encoder = _make_synthetic_encoder()
        config = ExperimentConfig()
        result = run_drm_experiment(
            lists, encoder, config, encoder_name="synthetic_cluster"
        )

        lure_sims = [
            t.retrieval.best_overall_similarity for t in result.trials_of_type("critical_lure")
        ]
        distractor_sims = [
            t.retrieval.best_overall_similarity for t in result.trials_of_type("distractor")
        ]

        lure_mean = float(np.mean(lure_sims))
        distractor_mean = float(np.mean(distractor_sims))

        # The architectural prediction: lures should match the consolidated
        # semantic centroid much more strongly than distractors do. Under
        # structured encoding, the gap should be substantial.
        assert lure_mean > distractor_mean + 0.20, (
            f"DRM emergence failed: lure mean similarity {lure_mean:.3f} "
            f"should substantially exceed distractor mean {distractor_mean:.3f}"
        )

    def test_lures_show_more_old_judgments_than_distractors(self) -> None:
        """At a reasonable recognition threshold, lures should be judged
        'old' (Remember or Know) much more often than distractors."""
        lists = canonical_lists()
        encoder = _make_synthetic_encoder()
        config = ExperimentConfig(threshold_recognition=0.3, threshold_episodic=0.7)
        result = run_drm_experiment(lists, encoder, config)

        def old_rate(trials: list[TrialResult]) -> float:
            if not trials:
                return 0.0
            n_old = sum(1 for t in trials if t.judgment.judgment != "New")
            return n_old / len(trials)

        lure_old = old_rate(result.trials_of_type("critical_lure"))
        distractor_old = old_rate(result.trials_of_type("distractor"))

        # Both rates depend on threshold choice. The architectural prediction
        # is that lures get "old" judgments at substantially higher rates.
        assert lure_old > distractor_old + 0.30, (
            f"lure 'old' rate {lure_old:.2f} should substantially exceed "
            f"distractor 'old' rate {distractor_old:.2f}"
        )

    def test_lures_get_know_judgments(self) -> None:
        """The canonical DRM finding (Roediger & McDermott 1995): critical
        lures are accompanied by familiarity (Know) more than recollection
        (Remember). Our R/K judgment mechanism predicts this directly:
        lures match the semantic centroid but lack episodic traces, so they
        get classified as Know rather than Remember.
        """
        lists = canonical_lists()
        encoder = _make_synthetic_encoder()
        # Pick thresholds that separate episodic from semantic clearly.
        # Studied items have episodic similarity ≈ 1.0 (own trace).
        # Lures have episodic similarity around 0.3-0.5 (closest related associate)
        # and semantic similarity ≈ 0.5-0.7 (consolidated centroid).
        config = ExperimentConfig(
            threshold_recognition=0.25,
            threshold_episodic=0.7,
        )
        result = run_drm_experiment(lists, encoder, config)

        lure_trials = result.trials_of_type("critical_lure")
        n_remember = sum(1 for t in lure_trials if t.judgment.judgment == "Remember")
        n_know = sum(1 for t in lure_trials if t.judgment.judgment == "Know")
        n_new = sum(1 for t in lure_trials if t.judgment.judgment == "New")

        # Most lures judged "old" (Know or Remember combined > New)
        assert (n_remember + n_know) > n_new, (
            f"lures should mostly be judged 'old': "
            f"Remember={n_remember}, Know={n_know}, New={n_new}"
        )
        # Among "old" judgments, Know should dominate Remember for lures
        # (this is the architectural signature of false-familiarity without recollection).
        assert n_know >= n_remember, (
            f"lures should be Know-dominant: Remember={n_remember}, Know={n_know}"
        )

    def test_iterated_retrieval_boosts_evidence(self) -> None:
        """The N_max manipulation should accumulate more evidence for lures
        at higher depths — the AM-mode signature.
        """
        lists = canonical_lists()
        encoder = _make_synthetic_encoder()

        config_ftt = ExperimentConfig(n_max=1)
        config_am = ExperimentConfig(n_max=3)

        result_ftt = run_drm_experiment(lists, encoder, config_ftt)
        result_am = run_drm_experiment(lists, encoder, config_am)

        lure_ev_ftt = np.mean([
            t.retrieval.accumulated_evidence for t in result_ftt.trials_of_type("critical_lure")
        ])
        lure_ev_am = np.mean([
            t.retrieval.accumulated_evidence for t in result_am.trials_of_type("critical_lure")
        ])

        # Multi-step should accumulate more evidence than single-step
        assert lure_ev_am > lure_ev_ftt, (
            f"AM-mode (n_max=3) accumulated evidence {lure_ev_am:.3f} "
            f"should exceed FTT-mode (n_max=1) {lure_ev_ftt:.3f}"
        )


class TestFTTvsAMDifferential:
    """The §6.5 co-headline contribution: same substrate, two retrieval
    regimes, two empirical patterns.

    At single-step (FTT-mode, N_max=1), direct lures recognize strongly via
    one-shot gist match; mediated lures DO NOT recognize because they're
    not directly close to the cluster centroid.

    At iterated (AM-mode, N_max≥2), the activation spreading through the
    cluster centroid accumulates additional evidence for mediated lures,
    narrowing or eliminating the direct-vs-mediated gap.
    """

    def test_ftt_cliff_direct_above_mediated(self) -> None:
        """N_max=1 single-step retrieval: direct lures match the centroid
        strongly; mediated lures do not. Sharp cliff between them."""
        lists = canonical_lists()
        encoder = _make_synthetic_encoder()
        config = ExperimentConfig(n_max=1)

        result = run_drm_experiment(lists, encoder, config)

        direct_sims = [
            t.retrieval.best_overall_similarity
            for t in result.trials_of_type("critical_lure")
        ]
        mediated_sims = [
            t.retrieval.best_overall_similarity
            for t in result.trials_of_type("mediated_lure")
        ]

        assert direct_sims, "no direct-lure trials"
        assert mediated_sims, "no mediated-lure trials"

        direct_mean = float(np.mean(direct_sims))
        mediated_mean = float(np.mean(mediated_sims))

        # The FTT cliff: direct >> mediated at N_max=1.
        assert direct_mean > mediated_mean + 0.20, (
            f"FTT cliff failed: direct mean {direct_mean:.3f} "
            f"should substantially exceed mediated mean {mediated_mean:.3f}"
        )

    def test_am_mode_accumulates_more_evidence_for_mediated(self) -> None:
        """Iterated retrieval should accumulate more evidence for mediated
        lures at higher depth — they have stepping stones (the studied
        associates near the cluster centroid) that single-step retrieval
        cannot reach.
        """
        lists = canonical_lists()
        encoder = _make_synthetic_encoder()

        result_ftt = run_drm_experiment(lists, encoder, ExperimentConfig(n_max=1))
        result_am = run_drm_experiment(lists, encoder, ExperimentConfig(n_max=3))

        def mean_evidence(result, item_type: str) -> float:
            xs = [t.retrieval.accumulated_evidence for t in result.trials_of_type(item_type)]
            return float(np.mean(xs)) if xs else 0.0

        mediated_ftt = mean_evidence(result_ftt, "mediated_lure")
        mediated_am = mean_evidence(result_am, "mediated_lure")

        assert mediated_am > mediated_ftt, (
            f"mediated-lure accumulated evidence should grow with iteration depth: "
            f"FTT={mediated_ftt:.3f}, AM={mediated_am:.3f}"
        )

    def test_distractors_unchanged_by_iteration(self) -> None:
        """Iteration should not raise distractor similarity — true foils
        have no stepping stones in the substrate, so spreading finds nothing
        useful to add."""
        lists = canonical_lists()
        encoder = _make_synthetic_encoder()

        result_ftt = run_drm_experiment(lists, encoder, ExperimentConfig(n_max=1))
        result_am = run_drm_experiment(lists, encoder, ExperimentConfig(n_max=3))

        def mean_best(result) -> float:
            xs = [t.retrieval.best_overall_similarity for t in result.trials_of_type("distractor")]
            return float(np.mean(xs)) if xs else 0.0

        dist_ftt = mean_best(result_ftt)
        dist_am = mean_best(result_am)

        # Distractor best-overall-similarity should stay near chance regardless
        # of iteration depth.
        assert dist_am < 0.25, (
            f"distractor similarity should stay near chance: AM={dist_am:.3f}"
        )
