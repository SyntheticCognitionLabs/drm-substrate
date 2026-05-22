"""Parameter sweep runners for the DRM paper.

Two sweeps:

1. ``sweep_frontier`` — 2D grid over (semantic EMA rate × episodic novelty
   threshold), each grid point a small battery. Per §6.7: characterizes the
   fidelity-generalization Pareto manifold by relating per-grid-point lure
   FA rate to held-out-associate recognition rate.

2. ``run_ablation`` — convenience wrapper for the ablation conditions of
   §6.6. Currently supports "no_consolidation" (the A/B condition under our
   architecture). The "random orthogonal encoding" ablation (§6.6 condition
   D) is handled by simply swapping the encoder; no special runner needed.

Each sweep returns a structured result that downstream analysis can
flatten into a tabular form for plotting.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace

import numpy as np

from drm_paper.battery import BatteryResult, aggregate_stats, run_battery
from drm_paper.encoding import WordEncoder
from drm_paper.experiment import ExperimentConfig
from drm_paper.stimuli import DRMList


@dataclass
class FrontierPoint:
    """One grid point in the §6.7 sweep."""

    ema_alpha: float
    novelty_threshold: float
    battery: BatteryResult


@dataclass
class FrontierSweep:
    """Full frontier sweep result."""

    ema_rates: tuple[float, ...]
    novelty_thresholds: tuple[float, ...]
    points: list[FrontierPoint] = field(default_factory=list)

    def lookup(self, ema: float, novelty: float) -> FrontierPoint | None:
        for p in self.points:
            if p.ema_alpha == ema and p.novelty_threshold == novelty:
                return p
        return None


def sweep_frontier(
    drm_lists: tuple[DRMList, ...],
    encoder_factory: Callable[[int], WordEncoder],
    *,
    base_config: ExperimentConfig | None = None,
    ema_rates: tuple[float, ...] = (0.01, 0.05, 0.20, 0.50),
    novelty_thresholds: tuple[float, ...] = (0.5, 0.7, 0.85, 0.95),
    n_seeds_per_point: int = 5,
    master_seed: int = 0,
    encoder_name: str = "unknown",
) -> FrontierSweep:
    """Run the §6.7 2D sweep over (EMA rate × novelty threshold).

    For each grid point, run a small battery (default 5 seeds) and record
    the result. Downstream analysis pulls aggregate stats from each
    battery to plot the Pareto manifold (DRM rate vs held-out recognition).

    Total runs: len(ema_rates) × len(novelty_thresholds) × n_seeds_per_point.
    Defaults: 4 × 4 × 5 = 80 single-seed experiments.
    """
    if base_config is None:
        base_config = ExperimentConfig()

    points: list[FrontierPoint] = []
    for alpha in ema_rates:
        for novelty in novelty_thresholds:
            point_config = replace(
                base_config,
                ema_alpha=alpha,
                novelty_threshold=novelty,
            )
            # Vary master seed per grid point to decorrelate noise.
            point_master_seed = master_seed + hash((alpha, novelty)) & 0xFFFFFFFF
            battery = run_battery(
                drm_lists,
                encoder_factory,
                base_config=point_config,
                n_seeds=n_seeds_per_point,
                master_seed=point_master_seed,
                encoder_name=encoder_name,
            )
            points.append(
                FrontierPoint(
                    ema_alpha=alpha,
                    novelty_threshold=novelty,
                    battery=battery,
                )
            )

    return FrontierSweep(
        ema_rates=ema_rates,
        novelty_thresholds=novelty_thresholds,
        points=points,
    )


@dataclass
class FrontierTabularRow:
    """One row of frontier sweep tabular output, ready for plotting."""

    ema_alpha: float
    novelty_threshold: float
    lure_old_rate: float
    lure_ci_low: float
    lure_ci_high: float
    held_out_old_rate: float
    held_out_ci_low: float
    held_out_ci_high: float
    distractor_old_rate: float


def frontier_to_table(
    sweep: FrontierSweep,
    *,
    n_resamples: int = 5000,
    rng: np.random.Generator | None = None,
) -> list[FrontierTabularRow]:
    """Flatten a FrontierSweep into rows suitable for matplotlib heatmaps or
    scatter plots. The headline x and y are lure_old_rate (DRM intensity)
    and held_out_old_rate (gist generalization)."""
    rows: list[FrontierTabularRow] = []
    for point in sweep.points:
        agg = aggregate_stats(point.battery, n_resamples=n_resamples, rng=rng)
        lure = agg["critical_lure"]
        held = agg["held_out"]
        dist = agg["distractor"]
        rows.append(
            FrontierTabularRow(
                ema_alpha=point.ema_alpha,
                novelty_threshold=point.novelty_threshold,
                lure_old_rate=lure.mean_old_rate,
                lure_ci_low=lure.ci_low,
                lure_ci_high=lure.ci_high,
                held_out_old_rate=held.mean_old_rate,
                held_out_ci_low=held.ci_low,
                held_out_ci_high=held.ci_high,
                distractor_old_rate=dist.mean_old_rate,
            )
        )
    return rows


def run_ablation(
    name: str,
    drm_lists: tuple[DRMList, ...],
    encoder_factory: Callable[[int], WordEncoder],
    *,
    base_config: ExperimentConfig | None = None,
    n_seeds: int = 10,
    master_seed: int = 0,
    encoder_name: str = "unknown",
) -> BatteryResult:
    """Run one §6.6 ablation as a battery.

    Supported names:
        - "full"             — no ablation; the substrate as-designed.
        - "no_consolidation" — disable semantic-CAM writes during study.
          Collapses ablations A (no consolidation) and B (no semantic
          store) since both produce the same observable.

    For ablation D (random orthogonal encoding), simply pass a
    RandomOrthogonalEncoder factory directly; no special runner needed.
    Ablation C (pure key-value retrieval) is not yet implemented; see
    paper §6.6 discussion notes.
    """
    if base_config is None:
        base_config = ExperimentConfig()

    if name == "full":
        config = base_config
    elif name == "no_consolidation":
        config = replace(base_config, disable_consolidation=True)
    else:
        raise ValueError(
            f"unknown ablation '{name}'. Supported: 'full', 'no_consolidation'."
        )

    return run_battery(
        drm_lists,
        encoder_factory,
        base_config=config,
        n_seeds=n_seeds,
        master_seed=master_seed,
        encoder_name=f"{encoder_name}/{name}",
    )
