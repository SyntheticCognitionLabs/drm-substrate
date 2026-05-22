"""Tests for drm_paper.figures.

These tests verify each figure function constructs a valid matplotlib Figure
without errors. They don't assert on rendered pixels — those would be flaky
across matplotlib versions and font availability. Visual inspection of the
saved PDFs is the meaningful check.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless backend for tests

from drm_paper.battery import run_battery
from drm_paper.encoding import SyntheticClusterEncoder
from drm_paper.experiment import ExperimentConfig
from drm_paper.figures import (
    figure_bas_regression,
    figure_evidence_by_iteration,
    figure_frontier_heatmap,
    figure_per_list_fa,
    figure_recognition_rates,
)
from drm_paper.stimuli import DEFAULT_FOILS, canonical_lists
from drm_paper.sweeps import sweep_frontier


def _factory(lists):
    def f(seed: int):
        return SyntheticClusterEncoder(
            lists, drop_frac=0.35, mediated_drop_frac=0.55,
            seed=seed, extra_words=DEFAULT_FOILS,
        )

    return f


def _small_battery(n_max: int = 1):
    lists = canonical_lists()
    return run_battery(
        lists, _factory(lists),
        base_config=ExperimentConfig(n_max=n_max, threshold_recognition=0.3, threshold_episodic=0.7),
        n_seeds=3,
        master_seed=42,
        encoder_name="synthetic_cluster",
    )


class TestFigureRecognitionRates:
    def test_constructs_figure(self) -> None:
        battery = _small_battery()
        fig = figure_recognition_rates(battery, n_resamples=200)
        # Single axes, expected x-tick count
        assert len(fig.axes) == 1
        ax = fig.axes[0]
        # 5 item types → 5 bars at x positions [0, 1, 2, 3, 4]
        assert len(ax.get_xticks()) == 5


class TestFigurePerListFA:
    def test_constructs_figure_with_one_bar_per_list(self) -> None:
        battery = _small_battery()
        fig = figure_per_list_fa(battery, n_resamples=200)
        ax = fig.axes[0]
        n_lists = len(canonical_lists())
        assert len(ax.get_xticks()) == n_lists


class TestFigureEvidenceByIteration:
    def test_constructs_figure_with_one_line_per_item_type(self) -> None:
        batteries = {
            1: _small_battery(n_max=1),
            2: _small_battery(n_max=2),
            3: _small_battery(n_max=3),
        }
        fig = figure_evidence_by_iteration(batteries)
        ax = fig.axes[0]
        # 5 item types
        assert len(ax.get_lines()) == 5


class TestFigureBASRegression:
    def test_constructs_figure_with_synthetic_bas(self) -> None:
        """Use a synthetic BAS-like dict (random per-list values) just to
        check the figure constructs. Real BAS regression is tested by
        running with ConceptNet-derived values in the driver script."""
        battery = _small_battery()
        # Synthetic BAS values for each list index
        bas_values = {i: 0.3 + 0.05 * i for i in range(len(canonical_lists()))}
        fig = figure_bas_regression(battery, bas_values, n_resamples=200)
        assert len(fig.axes) == 1
        # Should have 15 data points + a regression line
        ax = fig.axes[0]
        assert len(ax.collections) >= 1 or len(ax.lines) >= 1


class TestFigureFrontierHeatmap:
    def test_constructs_heatmap(self) -> None:
        lists = canonical_lists()
        sweep = sweep_frontier(
            lists, _factory(lists),
            base_config=ExperimentConfig(threshold_recognition=0.3),
            ema_rates=(0.05, 0.20),
            novelty_thresholds=(0.5, 0.9),
            n_seeds_per_point=2,
        )
        fig = figure_frontier_heatmap(sweep, n_resamples=200)
        # Heatmap has axes + colorbar axes
        assert len(fig.axes) >= 1
