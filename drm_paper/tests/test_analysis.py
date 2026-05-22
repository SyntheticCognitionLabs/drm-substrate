"""Tests for drm_paper.analysis."""

from __future__ import annotations

from drm_paper.analysis import (
    ItemTypeStats,
    per_list_lure_fa_rate,
    rk_dissociation_table,
    summary_stats,
)
from drm_paper.encoding import RandomOrthogonalEncoder, SyntheticClusterEncoder
from drm_paper.experiment import ExperimentConfig, run_drm_experiment
from drm_paper.stimuli import DEFAULT_FOILS, canonical_lists


def _make_synthetic_encoder(drop_frac: float = 0.35, seed: int = 42) -> SyntheticClusterEncoder:
    return SyntheticClusterEncoder(
        canonical_lists(),
        drop_frac=drop_frac,
        seed=seed,
        extra_words=DEFAULT_FOILS,
    )


class TestSummaryStats:
    def test_returns_one_entry_per_item_type(self) -> None:
        lists = canonical_lists()
        encoder = RandomOrthogonalEncoder(seed=42)
        result = run_drm_experiment(lists, encoder)

        stats = summary_stats(result)
        assert set(stats.keys()) == {"studied", "critical_lure", "mediated_lure", "held_out", "distractor"}
        for s in stats.values():
            assert isinstance(s, ItemTypeStats)
            assert 0.0 <= s.old_rate <= 1.0
            assert s.remember_rate + s.know_rate <= s.old_rate + 1e-9

    def test_studied_old_rate_high_with_structured_encoder(self) -> None:
        """Studied items have their own episodic trace; old rate should be high."""
        lists = canonical_lists()
        encoder = _make_synthetic_encoder()
        result = run_drm_experiment(lists, encoder)
        stats = summary_stats(result)
        assert stats["studied"].old_rate > 0.8, (
            f"studied old rate unexpectedly low: {stats['studied'].old_rate}"
        )

    def test_distractor_old_rate_low(self) -> None:
        """Distractors are unrelated unstudied items; old rate should be near zero."""
        lists = canonical_lists()
        encoder = _make_synthetic_encoder()
        result = run_drm_experiment(lists, encoder)
        stats = summary_stats(result)
        assert stats["distractor"].old_rate < 0.2, (
            f"distractor old rate unexpectedly high: {stats['distractor'].old_rate}"
        )

    def test_lure_old_rate_substantial(self) -> None:
        """Critical lures should show substantial DRM false-alarm rates
        under structured encoding."""
        lists = canonical_lists()
        encoder = _make_synthetic_encoder()
        # Pick a recognition threshold below typical lure similarity
        config = ExperimentConfig(threshold_recognition=0.3)
        result = run_drm_experiment(lists, encoder, config)
        stats = summary_stats(result)
        assert stats["critical_lure"].old_rate > 0.5, (
            f"lure old rate unexpectedly low under structured encoder: "
            f"{stats['critical_lure'].old_rate}"
        )


class TestPerListFA:
    def test_returns_one_per_list(self) -> None:
        lists = canonical_lists()
        encoder = _make_synthetic_encoder()
        result = run_drm_experiment(lists, encoder)
        fa_by_list = per_list_lure_fa_rate(result)
        # Some lists may produce 0 trials of critical_lure if N test items
        # is misconfigured; with defaults all should be present.
        assert len(fa_by_list) == len(lists)
        for li, rate in fa_by_list.items():
            assert 0 <= li < len(lists)
            assert 0.0 <= rate <= 1.0


class TestRKTable:
    def test_table_structure(self) -> None:
        lists = canonical_lists()
        encoder = _make_synthetic_encoder()
        result = run_drm_experiment(lists, encoder)
        table = rk_dissociation_table(result)

        assert set(table.keys()) == {"studied", "critical_lure", "mediated_lure", "held_out", "distractor"}
        for it, row in table.items():
            assert set(row.keys()) == {"Remember", "Know", "New"}
            for j, count in row.items():
                assert count >= 0

    def test_counts_sum_to_total_trials_per_type(self) -> None:
        lists = canonical_lists()
        encoder = _make_synthetic_encoder()
        result = run_drm_experiment(lists, encoder)

        table = rk_dissociation_table(result)
        for item_type, row in table.items():
            total = sum(row.values())
            assert total == len(result.trials_of_type(item_type))
