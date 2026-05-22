"""Multi-seed battery runner + bootstrap statistics.

Wraps ``run_drm_experiment`` to aggregate trial-level data across N seeds
with bootstrap CIs. Each seed varies all five sub-seeds in ExperimentConfig
plus the encoder seed deterministically from a master seed; per-seed
results stay individually inspectable.

This is the orchestration layer between the single-experiment runner
(``experiment.run_drm_experiment``) and the summary analysis
(``analysis.summary_stats`` / ``analysis.per_list_lure_fa_rate``).
Per-list FA rates for BAS regression are computed by aggregating
single-seed per-list rates across the battery.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace

import numpy as np

from drm_paper.analysis import ItemTypeStats, summary_stats
from drm_paper.encoding import WordEncoder
from drm_paper.experiment import (
    ExperimentConfig,
    ExperimentResult,
    ItemType,
    TrialResult,
    run_drm_experiment,
)
from drm_paper.stimuli import DRMList


@dataclass
class BatteryResult:
    """All trial-level data across N seeded experiments."""

    master_seed: int
    n_seeds: int
    encoder_name: str
    base_config: ExperimentConfig
    per_seed_results: list[ExperimentResult] = field(default_factory=list)

    def all_trials(self) -> list[TrialResult]:
        out: list[TrialResult] = []
        for r in self.per_seed_results:
            out.extend(r.trials)
        return out

    def trials_of_type(self, item_type: ItemType) -> list[TrialResult]:
        return [t for t in self.all_trials() if t.item_type == item_type]


@dataclass(frozen=True)
class AggregateStats:
    """Per-item-type stats aggregated across seeds with bootstrap CIs.

    Attributes:
        item_type: which class of test item these stats describe.
        n_seeds: number of seeds aggregated.
        n_trials_per_seed: mean trials of this type per seed.
        per_seed_old_rates: per-seed old (Remember + Know) rates.
        mean_old_rate: across-seeds mean.
        ci_low / ci_high: bootstrap percentile CI bounds on mean_old_rate.
        mean_remember_rate, mean_know_rate: parallel.
        mean_similarity, mean_episodic, mean_semantic, mean_evidence: parallels.
    """

    item_type: ItemType
    n_seeds: int
    n_trials_per_seed: float
    per_seed_old_rates: tuple[float, ...]
    mean_old_rate: float
    ci_low: float
    ci_high: float
    mean_remember_rate: float
    mean_know_rate: float
    mean_similarity: float
    mean_episodic_similarity: float
    mean_semantic_similarity: float
    mean_accumulated_evidence: float


def _derive_seeds(master_seed: int, i: int) -> tuple[int, int, int, int, int, int]:
    """Derive six independent integer seeds from a master seed and seed index.

    Returns: (list_order, item_order, distractor_sample, substrate, retrieval, encoder).
    """
    rng = np.random.default_rng(np.uint64((master_seed * 1_000_003 + i * 7919) & 0xFFFFFFFF))
    seeds = rng.integers(0, 2**31 - 1, size=6)
    return (
        int(seeds[0]), int(seeds[1]), int(seeds[2]),
        int(seeds[3]), int(seeds[4]), int(seeds[5]),
    )


def run_battery(
    drm_lists: tuple[DRMList, ...],
    encoder_factory: Callable[[int], WordEncoder],
    base_config: ExperimentConfig | None = None,
    *,
    n_seeds: int = 50,
    master_seed: int = 0,
    encoder_name: str = "unknown",
) -> BatteryResult:
    """Run N seeded DRM experiments and collect their results.

    Args:
        drm_lists: stimulus lists.
        encoder_factory: a callable seed -> WordEncoder. Constructs a fresh
            encoder per seed so that any randomness inside the encoder
            (random projection, base hypervectors) varies independently
            across seeds.
        base_config: experiment config; sub-seeds are derived per seed and
            replace the base config's seed fields. All other fields preserved.
        n_seeds: number of seeded runs.
        master_seed: deterministic master seed; the sub-seeds for each
            individual experiment derive from this.
        encoder_name: human-readable label saved on each per-seed result.
    """
    if n_seeds < 1:
        raise ValueError(f"n_seeds must be >= 1, got {n_seeds}")
    if base_config is None:
        base_config = ExperimentConfig()

    per_seed: list[ExperimentResult] = []

    for i in range(n_seeds):
        list_seed, item_seed, dist_seed, sub_seed, retr_seed, enc_seed = _derive_seeds(
            master_seed, i
        )
        config = replace(
            base_config,
            list_order_seed=list_seed,
            item_order_seed=item_seed,
            distractor_sample_seed=dist_seed,
            substrate_seed=sub_seed,
            retrieval_seed=retr_seed,
        )
        encoder = encoder_factory(enc_seed)
        result = run_drm_experiment(drm_lists, encoder, config, encoder_name=encoder_name)
        per_seed.append(result)

    return BatteryResult(
        master_seed=master_seed,
        n_seeds=n_seeds,
        encoder_name=encoder_name,
        base_config=base_config,
        per_seed_results=per_seed,
    )


def bootstrap_ci(
    values: list[float] | tuple[float, ...] | np.ndarray,
    *,
    confidence: float = 0.95,
    n_resamples: int = 10_000,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """Bootstrap percentile CI for the mean of ``values``.

    Returns (low, high) at the given confidence level. With <2 values,
    returns (mean, mean) — no spread to estimate.
    """
    if not (0.0 < confidence < 1.0):
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return (0.0, 0.0)
    if arr.size < 2:
        m = float(arr.mean())
        return (m, m)
    if rng is None:
        rng = np.random.default_rng(0)
    idx = rng.integers(0, arr.size, size=(n_resamples, arr.size))
    boot_means = arr[idx].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    low = float(np.quantile(boot_means, alpha))
    high = float(np.quantile(boot_means, 1.0 - alpha))
    return (low, high)


def aggregate_stats(
    battery: BatteryResult,
    *,
    confidence: float = 0.95,
    n_resamples: int = 10_000,
    rng: np.random.Generator | None = None,
) -> dict[ItemType, AggregateStats]:
    """Aggregate per-seed summary stats across the battery with bootstrap CIs."""
    item_types: tuple[ItemType, ...] = (
        "studied", "critical_lure", "mediated_lure", "held_out", "distractor"
    )
    out: dict[ItemType, AggregateStats] = {}

    for it in item_types:
        per_seed_stats: list[ItemTypeStats] = []
        for r in battery.per_seed_results:
            s = summary_stats(r)[it]
            per_seed_stats.append(s)

        # Filter out seeds that had no trials of this type (rare; defensive)
        nonempty = [s for s in per_seed_stats if s.n_trials > 0]
        if not nonempty:
            out[it] = AggregateStats(
                item_type=it, n_seeds=battery.n_seeds, n_trials_per_seed=0.0,
                per_seed_old_rates=(), mean_old_rate=0.0, ci_low=0.0, ci_high=0.0,
                mean_remember_rate=0.0, mean_know_rate=0.0,
                mean_similarity=0.0, mean_episodic_similarity=0.0,
                mean_semantic_similarity=0.0, mean_accumulated_evidence=0.0,
            )
            continue

        old_rates = [s.old_rate for s in nonempty]
        remember_rates = [s.remember_rate for s in nonempty]
        know_rates = [s.know_rate for s in nonempty]
        sims = [s.mean_overall_similarity for s in nonempty]
        ep_sims = [s.mean_episodic_similarity for s in nonempty]
        sm_sims = [s.mean_semantic_similarity for s in nonempty]
        # Pull per-trial evidence rather than from ItemTypeStats (it doesn't store it)
        evidences: list[float] = []
        for r in battery.per_seed_results:
            seed_trials = r.trials_of_type(it)
            if seed_trials:
                evidences.append(
                    float(np.mean([t.retrieval.accumulated_evidence for t in seed_trials]))
                )

        ci_low, ci_high = bootstrap_ci(
            old_rates, confidence=confidence, n_resamples=n_resamples, rng=rng
        )

        out[it] = AggregateStats(
            item_type=it,
            n_seeds=battery.n_seeds,
            n_trials_per_seed=float(np.mean([s.n_trials for s in nonempty])),
            per_seed_old_rates=tuple(old_rates),
            mean_old_rate=float(np.mean(old_rates)),
            ci_low=ci_low,
            ci_high=ci_high,
            mean_remember_rate=float(np.mean(remember_rates)),
            mean_know_rate=float(np.mean(know_rates)),
            mean_similarity=float(np.mean(sims)),
            mean_episodic_similarity=float(np.mean(ep_sims)),
            mean_semantic_similarity=float(np.mean(sm_sims)),
            mean_accumulated_evidence=float(np.mean(evidences)) if evidences else 0.0,
        )
    return out


def per_list_lure_fa_rate_with_ci(
    battery: BatteryResult,
    *,
    confidence: float = 0.95,
    n_resamples: int = 10_000,
    rng: np.random.Generator | None = None,
) -> dict[int, tuple[float, float, float]]:
    """For each list index, the cross-seed mean critical-lure FA rate and CI.

    Returns dict[list_index, (mean_fa, ci_low, ci_high)]. Pair this with
    published BAS values per list to do the headline BAS regression.
    """
    per_list_per_seed: dict[int, list[float]] = {}

    for r in battery.per_seed_results:
        per_list_hits: dict[int, list[int]] = {}
        for t in r.trials_of_type("critical_lure"):
            per_list_hits.setdefault(t.list_index, []).append(
                1 if t.judgment.judgment != "New" else 0
            )
        for li, hits in per_list_hits.items():
            per_list_per_seed.setdefault(li, []).append(sum(hits) / len(hits))

    out: dict[int, tuple[float, float, float]] = {}
    for li, rates in per_list_per_seed.items():
        low, high = bootstrap_ci(
            rates, confidence=confidence, n_resamples=n_resamples, rng=rng
        )
        out[li] = (float(np.mean(rates)), low, high)
    return out
