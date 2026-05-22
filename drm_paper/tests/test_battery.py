"""Tests for drm_paper.battery."""

from __future__ import annotations

import numpy as np

from drm_paper.battery import (
    AggregateStats,
    BatteryResult,
    aggregate_stats,
    bootstrap_ci,
    per_list_lure_fa_rate_with_ci,
    run_battery,
)
from drm_paper.encoding import RandomOrthogonalEncoder, SyntheticClusterEncoder
from drm_paper.experiment import ExperimentConfig
from drm_paper.stimuli import DEFAULT_FOILS, canonical_lists


def _synthetic_factory(lists):
    """Returns a factory: seed -> SyntheticClusterEncoder built with that seed."""

    def factory(seed: int) -> SyntheticClusterEncoder:
        return SyntheticClusterEncoder(
            lists, drop_frac=0.35, mediated_drop_frac=0.55,
            seed=seed, extra_words=DEFAULT_FOILS,
        )

    return factory


class TestRunBattery:
    def test_runs_n_seeds(self) -> None:
        lists = canonical_lists()
        battery = run_battery(
            lists,
            _synthetic_factory(lists),
            n_seeds=5,
            master_seed=42,
            encoder_name="synthetic_cluster",
        )
        assert isinstance(battery, BatteryResult)
        assert battery.n_seeds == 5
        assert len(battery.per_seed_results) == 5

    def test_deterministic_from_master_seed(self) -> None:
        lists = canonical_lists()
        b1 = run_battery(lists, _synthetic_factory(lists), n_seeds=3, master_seed=42)
        b2 = run_battery(lists, _synthetic_factory(lists), n_seeds=3, master_seed=42)
        for r1, r2 in zip(b1.per_seed_results, b2.per_seed_results):
            for t1, t2 in zip(r1.trials, r2.trials):
                assert t1.list_index == t2.list_index
                assert t1.item_type == t2.item_type
                assert t1.item_word == t2.item_word
                assert t1.retrieval.best_overall_similarity == t2.retrieval.best_overall_similarity

    def test_different_master_seeds_differ(self) -> None:
        lists = canonical_lists()
        b1 = run_battery(lists, _synthetic_factory(lists), n_seeds=3, master_seed=42)
        b2 = run_battery(lists, _synthetic_factory(lists), n_seeds=3, master_seed=7)
        # At least one trial across the batteries should differ
        diff = False
        for r1, r2 in zip(b1.per_seed_results, b2.per_seed_results):
            for t1, t2 in zip(r1.trials, r2.trials):
                if t1.retrieval.best_overall_similarity != t2.retrieval.best_overall_similarity:
                    diff = True
                    break
            if diff:
                break
        assert diff, "different master seeds should produce different results"

    def test_each_seed_uses_independent_encoder(self) -> None:
        """The factory should be called once per seed; the encoder seeds vary."""
        lists = canonical_lists()
        seeds_seen: list[int] = []

        def tracking_factory(seed: int):
            seeds_seen.append(seed)
            return RandomOrthogonalEncoder(seed=seed)

        run_battery(lists, tracking_factory, n_seeds=4, master_seed=42)
        assert len(seeds_seen) == 4
        # Seeds should differ
        assert len(set(seeds_seen)) == 4


class TestBootstrapCI:
    def test_constant_data_yields_tight_ci(self) -> None:
        values = [0.5] * 20
        low, high = bootstrap_ci(values, n_resamples=1000)
        # Constant data → CI degenerate around the mean
        assert abs(low - 0.5) < 1e-9
        assert abs(high - 0.5) < 1e-9

    def test_ci_brackets_mean_for_simple_data(self) -> None:
        rng = np.random.default_rng(0)
        values = rng.normal(0.5, 0.1, size=50).tolist()
        low, high = bootstrap_ci(values, confidence=0.95, n_resamples=2000, rng=rng)
        true_mean = float(np.mean(values))
        assert low < true_mean < high

    def test_single_value(self) -> None:
        low, high = bootstrap_ci([0.7])
        assert low == high == 0.7

    def test_empty(self) -> None:
        assert bootstrap_ci([]) == (0.0, 0.0)


class TestAggregateStats:
    def test_returns_one_per_item_type(self) -> None:
        lists = canonical_lists()
        battery = run_battery(
            lists, _synthetic_factory(lists), n_seeds=3, master_seed=42
        )
        agg = aggregate_stats(battery, n_resamples=200)
        assert set(agg.keys()) == {
            "studied", "critical_lure", "mediated_lure", "held_out", "distractor"
        }
        for it, s in agg.items():
            assert isinstance(s, AggregateStats)
            assert s.n_seeds == 3
            assert s.ci_low <= s.mean_old_rate <= s.ci_high + 1e-9

    def test_drm_pattern_with_synthetic_encoder(self) -> None:
        """End-to-end battery check: structured encoding produces high lure
        old-rate; distractors stay near zero."""
        lists = canonical_lists()
        config = ExperimentConfig(threshold_recognition=0.3, threshold_episodic=0.7)
        battery = run_battery(
            lists, _synthetic_factory(lists), base_config=config,
            n_seeds=5, master_seed=42, encoder_name="synthetic_cluster",
        )
        agg = aggregate_stats(battery, n_resamples=500)
        assert agg["critical_lure"].mean_old_rate > 0.5
        assert agg["distractor"].mean_old_rate < 0.2


class TestPerListFAWithCI:
    def test_returns_one_per_list(self) -> None:
        lists = canonical_lists()
        battery = run_battery(
            lists, _synthetic_factory(lists), n_seeds=5, master_seed=42
        )
        out = per_list_lure_fa_rate_with_ci(battery, n_resamples=500)
        assert len(out) == len(lists)
        for li, (mean, low, high) in out.items():
            assert 0 <= li < len(lists)
            assert 0.0 <= mean <= 1.0
            assert low <= mean <= high + 1e-9
