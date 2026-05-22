"""Analysis of DRM experiment results.

Headline summaries computed from an ``ExperimentResult``: recognition rates
by item type, Remember/Know dissociation table, per-list false-alarm rates
for BAS regression. Bootstrap CIs and significance testing are deferred —
this module provides the deterministic per-result summaries that the
multi-seed battery will aggregate.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Literal

from drm_paper.experiment import ExperimentResult, ItemType, TrialResult
from drm_paper.retrieval import Judgment


@dataclass(frozen=True)
class ItemTypeStats:
    """Summary stats for one item type across all trials.

    Attributes:
        n_trials: total trials of this type.
        n_old: trials judged Remember or Know.
        n_new: trials judged New.
        n_remember: trials judged Remember.
        n_know: trials judged Know.
        old_rate: n_old / n_trials.
        remember_rate: n_remember / n_trials.
        know_rate: n_know / n_trials.
        mean_overall_similarity: mean best_overall_similarity across trials.
        mean_episodic_similarity: mean best_episodic_similarity across trials.
        mean_semantic_similarity: mean best_semantic_similarity across trials.
    """

    item_type: ItemType
    n_trials: int
    n_old: int
    n_new: int
    n_remember: int
    n_know: int
    old_rate: float
    remember_rate: float
    know_rate: float
    mean_overall_similarity: float
    mean_episodic_similarity: float
    mean_semantic_similarity: float


def _summarize_one(item_type: ItemType, trials: list[TrialResult]) -> ItemTypeStats:
    n = len(trials)
    if n == 0:
        return ItemTypeStats(
            item_type=item_type,
            n_trials=0, n_old=0, n_new=0, n_remember=0, n_know=0,
            old_rate=0.0, remember_rate=0.0, know_rate=0.0,
            mean_overall_similarity=0.0,
            mean_episodic_similarity=0.0,
            mean_semantic_similarity=0.0,
        )
    n_remember = sum(1 for t in trials if t.judgment.judgment == "Remember")
    n_know = sum(1 for t in trials if t.judgment.judgment == "Know")
    n_new = sum(1 for t in trials if t.judgment.judgment == "New")
    n_old = n_remember + n_know
    sum_overall = sum(t.retrieval.best_overall_similarity for t in trials)
    sum_episodic = sum(t.retrieval.best_episodic_similarity for t in trials)
    sum_semantic = sum(t.retrieval.best_semantic_similarity for t in trials)
    return ItemTypeStats(
        item_type=item_type,
        n_trials=n,
        n_old=n_old,
        n_new=n_new,
        n_remember=n_remember,
        n_know=n_know,
        old_rate=n_old / n,
        remember_rate=n_remember / n,
        know_rate=n_know / n,
        mean_overall_similarity=sum_overall / n,
        mean_episodic_similarity=sum_episodic / n,
        mean_semantic_similarity=sum_semantic / n,
    )


def summary_stats(result: ExperimentResult) -> dict[ItemType, ItemTypeStats]:
    """Per-item-type summary stats covering recognition rates and R/K breakdown.

    Headline DRM finding to look for:
        - studied items: high Remember rate (own episodic trace)
        - critical lures: high Know rate, low Remember rate
          (semantic match without recollection)
        - distractors: low old_rate (New dominant)
        - held_out: moderate old_rate (gist-driven generalization)
    """
    item_types: tuple[ItemType, ...] = (
        "studied", "critical_lure", "mediated_lure", "held_out", "distractor"
    )
    return {it: _summarize_one(it, result.trials_of_type(it)) for it in item_types}


def per_list_lure_fa_rate(result: ExperimentResult) -> dict[int, float]:
    """Per-list false-alarm rate on the critical lure.

    For each list_index, returns the fraction of critical-lure trials at
    that list judged Remember or Know (i.e., false-alarmed as "old"). Used
    for BAS regression: per-list FA rate vs published BAS values.

    With the default ExperimentConfig (1 critical lure per list), this is
    binary per list. Aggregating across seeds (multi-seed battery) gives the
    smooth per-list rate that's actually informative.
    """
    per_list: dict[int, list[int]] = {}
    for t in result.trials_of_type("critical_lure"):
        per_list.setdefault(t.list_index, []).append(
            1 if t.judgment.judgment != "New" else 0
        )
    return {li: sum(hits) / len(hits) for li, hits in per_list.items() if hits}


def rk_dissociation_table(
    result: ExperimentResult,
) -> dict[ItemType, dict[Judgment, int]]:
    """Counts of (item_type, judgment) pairs across all trials.

    The classic DRM-paper table: each row is an item type, each column is
    Remember/Know/New. The signature pattern:

        item_type        Remember    Know    New
        studied              high     low    low
        critical_lure         low    high    low-moderate
        held_out          moderate    low    moderate
        distractor            low     low    high

    Returned as a nested dict; convert to pandas / printed table downstream
    as needed.
    """
    item_types: tuple[ItemType, ...] = (
        "studied", "critical_lure", "mediated_lure", "held_out", "distractor"
    )
    judgments: tuple[Judgment, ...] = ("Remember", "Know", "New")
    table: dict[ItemType, dict[Judgment, int]] = {}
    for it in item_types:
        c: Counter[Judgment] = Counter()
        for t in result.trials_of_type(it):
            c[t.judgment.judgment] += 1
        table[it] = {j: c[j] for j in judgments}
    return table
