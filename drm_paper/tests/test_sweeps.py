"""Tests for drm_paper.sweeps."""

from __future__ import annotations

from drm_paper.battery import BatteryResult, aggregate_stats
from drm_paper.encoding import RandomOrthogonalEncoder, SyntheticClusterEncoder
from drm_paper.experiment import ExperimentConfig
from drm_paper.stimuli import DEFAULT_FOILS, canonical_lists
from drm_paper.sweeps import (
    FrontierSweep,
    FrontierTabularRow,
    frontier_to_table,
    run_ablation,
    sweep_frontier,
)


def _synthetic_factory(lists):
    def factory(seed: int) -> SyntheticClusterEncoder:
        return SyntheticClusterEncoder(
            lists, drop_frac=0.35, mediated_drop_frac=0.55,
            seed=seed, extra_words=DEFAULT_FOILS,
        )

    return factory


class TestFrontierSweep:
    def test_grid_is_complete(self) -> None:
        lists = canonical_lists()
        sweep = sweep_frontier(
            lists,
            _synthetic_factory(lists),
            ema_rates=(0.05, 0.20),
            novelty_thresholds=(0.5, 0.9),
            n_seeds_per_point=2,
        )
        assert isinstance(sweep, FrontierSweep)
        assert sweep.ema_rates == (0.05, 0.20)
        assert sweep.novelty_thresholds == (0.5, 0.9)
        assert len(sweep.points) == 4  # 2 × 2

    def test_lookup(self) -> None:
        lists = canonical_lists()
        sweep = sweep_frontier(
            lists,
            _synthetic_factory(lists),
            ema_rates=(0.05, 0.20),
            novelty_thresholds=(0.5,),
            n_seeds_per_point=2,
        )
        assert sweep.lookup(0.05, 0.5) is not None
        assert sweep.lookup(0.20, 0.5) is not None
        assert sweep.lookup(0.99, 0.99) is None

    def test_table_export(self) -> None:
        lists = canonical_lists()
        sweep = sweep_frontier(
            lists,
            _synthetic_factory(lists),
            ema_rates=(0.05, 0.20),
            novelty_thresholds=(0.5, 0.9),
            n_seeds_per_point=2,
        )
        rows = frontier_to_table(sweep, n_resamples=200)
        assert len(rows) == 4
        for row in rows:
            assert isinstance(row, FrontierTabularRow)
            assert 0.0 <= row.lure_old_rate <= 1.0
            assert row.lure_ci_low <= row.lure_old_rate <= row.lure_ci_high + 1e-9


class TestAblations:
    def test_full_vs_no_consolidation(self) -> None:
        """The no_consolidation ablation should yield a lower critical-lure
        FA rate than the full substrate — without semantic consolidation,
        the gist centroid never forms, and the lure has to rely on direct
        overlap with individual stored associates.

        With the synthetic encoder, individual associates already overlap
        ~0.65 with the lure (drop_frac=0.35), so we need a recognition
        threshold above that level for the consolidation effect to be
        visible in judgments. Without consolidation the lure similarity
        drops from ~0.98 (centroid match) to ~0.65 (best-associate match);
        a threshold of 0.8 separates these regimes."""
        lists = canonical_lists()
        factory = _synthetic_factory(lists)
        config = ExperimentConfig(threshold_recognition=0.8, threshold_episodic=0.95)

        full = run_ablation(
            "full", lists, factory, base_config=config, n_seeds=3
        )
        no_consol = run_ablation(
            "no_consolidation", lists, factory, base_config=config, n_seeds=3
        )

        agg_full = aggregate_stats(full, n_resamples=200)
        agg_no = aggregate_stats(no_consol, n_resamples=200)

        # Full substrate produces strong DRM at the elevated threshold
        assert agg_full["critical_lure"].mean_old_rate > 0.5, (
            f"full substrate lure rate {agg_full['critical_lure'].mean_old_rate:.2f} "
            "should still be high at threshold 0.8 (centroid match)"
        )
        # No-consolidation substrate substantially reduces it
        assert agg_no["critical_lure"].mean_old_rate < agg_full["critical_lure"].mean_old_rate - 0.30, (
            f"no_consolidation should suppress DRM: "
            f"full={agg_full['critical_lure'].mean_old_rate:.2f}, "
            f"no_consol={agg_no['critical_lure'].mean_old_rate:.2f}"
        )

    def test_random_orthogonal_ablation_no_drm(self) -> None:
        """Random-orthogonal encoding (§6.6 condition D) should produce
        DRM ≈ 0 regardless of consolidation, since there's no semantic
        structure in the input."""
        lists = canonical_lists()

        def factory(seed: int) -> RandomOrthogonalEncoder:
            return RandomOrthogonalEncoder(seed=seed)

        config = ExperimentConfig(threshold_recognition=0.3, threshold_episodic=0.7)
        battery = run_ablation(
            "full", lists, factory, base_config=config, n_seeds=3,
            encoder_name="random_orthogonal",
        )
        agg = aggregate_stats(battery, n_resamples=200)
        assert agg["critical_lure"].mean_old_rate < 0.1, (
            f"random orthogonal should produce near-zero DRM, got "
            f"{agg['critical_lure'].mean_old_rate:.2f}"
        )

    def test_unknown_ablation_raises(self) -> None:
        lists = canonical_lists()
        factory = _synthetic_factory(lists)
        import pytest

        with pytest.raises(ValueError, match="unknown ablation"):
            run_ablation("xyzzy", lists, factory)
