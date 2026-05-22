"""Paper figures.

Each function takes battery / sweep results and returns a matplotlib Figure
object. Render-and-save is the caller's responsibility — the functions
themselves are pure construction. Save with::

    fig = figure_recognition_rates(battery)
    fig.savefig("drm_paper/results/figures/fig1.pdf", bbox_inches="tight")

Headline figures planned for the paper (per §7 of DRM_PAPER_OUTLINE.md):

    fig1: substrate schematic + DRM protocol (manual, not here)
    fig2: recognition rates by item type with R/K breakdown
    fig3: BAS regression — per-list FA against published BAS values
    fig4: FTT-AM dissociation — evidence growth across iteration depths
    fig5: encoding-richness scaling across four encoder conditions
    fig6: 2D fidelity-generalization frontier heatmap

This module currently implements fig2, fig3, fig4, fig6 from the substrate's
output. Fig1 is manually constructed; fig5 is built externally once the
ConceptNet/Nelson/Cooccurrence runs are all available.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from drm_paper.battery import (
    BatteryResult,
    aggregate_stats,
    per_list_lure_fa_rate_with_ci,
)
from drm_paper.experiment import ExperimentResult, ItemType
from drm_paper.sweeps import FrontierSweep, frontier_to_table


def figure_recognition_rates(
    battery: BatteryResult,
    *,
    title: str | None = None,
    n_resamples: int = 5000,
):
    """Figure 2: bar chart of recognition rates by item type with R/K split
    and bootstrap CIs.

    Stacked bars: Remember (dark) + Know (medium) = Old; New shown for context.
    Error bars on the Old rate.
    """
    import matplotlib.pyplot as plt

    agg = aggregate_stats(battery, n_resamples=n_resamples)
    item_order: tuple[ItemType, ...] = (
        "studied", "critical_lure", "mediated_lure", "held_out", "distractor"
    )
    labels = ["Studied", "Critical\nlure", "Mediated\nlure", "Held-out\nassociate", "Distractor"]
    n_items = len(item_order)

    remember = np.array([agg[it].mean_remember_rate for it in item_order])
    know = np.array([agg[it].mean_know_rate for it in item_order])
    new_ = np.array([1.0 - r - k for r, k in zip(remember, know)])
    old_rates = np.array([agg[it].mean_old_rate for it in item_order])
    ci_low = np.array([agg[it].ci_low for it in item_order])
    ci_high = np.array([agg[it].ci_high for it in item_order])
    yerr = np.vstack([old_rates - ci_low, ci_high - old_rates])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(n_items)

    # Stacked: Remember on bottom, Know on top, New transparent for context.
    ax.bar(x, remember, color="#1f4e79", label="Remember", edgecolor="white")
    ax.bar(x, know, bottom=remember, color="#7eb0d5", label="Know", edgecolor="white")
    ax.bar(
        x, new_, bottom=remember + know, color="lightgray",
        label="New", edgecolor="white", alpha=0.5,
    )

    # CIs on total "Old" rate
    ax.errorbar(
        x, old_rates, yerr=yerr, fmt="none", ecolor="black",
        capsize=4, capthick=1.0, linewidth=1.0,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Response rate")
    ax.set_ylim(0, 1.05)
    leg = ax.legend(loc="upper right", frameon=False)
    # The bars use white edges to separate stack segments visually; the
    # legend swatches inherit that, which disappears into the page. Force
    # the legend swatches alone to a black outline for contrast.
    for patch in leg.get_patches():
        patch.set_edgecolor("black")
        patch.set_linewidth(0.8)
    if title:
        ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    return fig


def figure_per_list_fa(
    battery: BatteryResult,
    *,
    title: str | None = None,
    n_resamples: int = 5000,
):
    """Figure 3: per-list critical-lure FA rate, ordered by mean rate, with CIs.

    Once published BAS values are available, this becomes a scatter against
    BAS for the headline regression. Until then it's an ordered bar chart
    that shows the per-list variance — i.e., that the substrate produces
    differentiated DRM rates list-by-list, not a uniform output.
    """
    import matplotlib.pyplot as plt

    fa = per_list_lure_fa_rate_with_ci(battery, n_resamples=n_resamples)
    # Pull list-index → critical lure name for x labels
    first_result = battery.per_seed_results[0]
    lure_names: dict[int, str] = {}
    for t in first_result.trials_of_type("critical_lure"):
        lure_names[t.list_index] = t.critical_lure

    items = sorted(fa.items(), key=lambda kv: kv[1][0], reverse=True)
    labels = [lure_names.get(li, str(li)) for li, _ in items]
    means = np.array([m for _, (m, _l, _h) in items])
    lows = np.array([l for _, (_m, l, _h) in items])
    highs = np.array([h for _, (_m, _l, h) in items])
    yerr = np.vstack([means - lows, highs - means])

    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(items))
    ax.bar(x, means, color="#7eb0d5", edgecolor="white")
    ax.errorbar(
        x, means, yerr=yerr, fmt="none", ecolor="black",
        capsize=3, capthick=1.0, linewidth=1.0,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Critical-lure false-alarm rate")
    ax.set_ylim(0, 1.05)
    if title:
        ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


def figure_evidence_by_iteration(
    batteries_by_n_max: dict[int, BatteryResult],
    *,
    title: str | None = None,
):
    """Figure 4: accumulated evidence by item type across iteration depths.

    Shows the AM-mode signature: evidence grows monotonically with N_max.
    The FTT-AM differential we want to surface is whether mediated-lure
    evidence grows faster (proportionally) than direct-lure evidence —
    if it does, iteration is preferentially rescuing the mediated case.
    """
    import matplotlib.pyplot as plt

    n_maxes = sorted(batteries_by_n_max.keys())
    item_order: tuple[ItemType, ...] = (
        "studied", "critical_lure", "mediated_lure", "held_out", "distractor"
    )
    labels = {
        "studied": "Studied",
        "critical_lure": "Critical lure",
        "mediated_lure": "Mediated lure",
        "held_out": "Held-out associate",
        "distractor": "Distractor",
    }
    colors = {
        "studied": "#1f4e79",
        "critical_lure": "#c0392b",
        "mediated_lure": "#e67e22",
        "held_out": "#7eb0d5",
        "distractor": "lightgray",
    }

    fig, ax = plt.subplots(figsize=(7, 4.5))

    for it in item_order:
        ys = []
        for n in n_maxes:
            agg = aggregate_stats(batteries_by_n_max[n], n_resamples=500)
            ys.append(agg[it].mean_accumulated_evidence)
        ax.plot(n_maxes, ys, marker="o", color=colors[it], label=labels[it])

    ax.set_xlabel("Iteration depth (N_max)")
    ax.set_ylabel("Accumulated evidence")
    ax.set_xticks(n_maxes)
    if title:
        ax.set_title(title)
    ax.legend(loc="upper left", frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


def figure_bas_regression(
    battery: BatteryResult,
    bas_values: dict[int, float],
    *,
    metric: Literal["fa_rate", "mean_similarity", "mean_evidence"] = "fa_rate",
    bas_label: str = "ConceptNet BAS proxy",
    y_label: str | None = None,
    title: str | None = None,
    n_resamples: int = 5000,
    label_points: bool = False,
):
    """Figure 3 (regression version): scatter per-list critical-lure FA rate
    against per-list BAS values, with a least-squares regression line and
    Pearson r².

    Args:
        battery: multi-seed battery result; per-list FA rates pulled from
            ``per_list_lure_fa_rate_with_ci``.
        bas_values: dict[list_index, BAS]. Use
            ``stimuli.compute_conceptnet_bas(...)`` for the proxy until
            published Stadler 1999 values are available.

    The headline regression for the paper: substrate FA rates correlate
    with the associative-strength predictor of each list, mirroring the
    human finding that BAS predicts DRM rates (Roediger et al. 2001).
    """
    import matplotlib.pyplot as plt

    first_result = battery.per_seed_results[0]
    lure_names: dict[int, str] = {
        t.list_index: t.critical_lure
        for t in first_result.trials_of_type("critical_lure")
    }

    # Compute the chosen per-list metric with bootstrap CIs.
    per_list_values: dict[int, list[float]] = {}
    for r in battery.per_seed_results:
        for t in r.trials_of_type("critical_lure"):
            if metric == "fa_rate":
                v = 1.0 if t.judgment.judgment != "New" else 0.0
            elif metric == "mean_similarity":
                v = t.retrieval.best_overall_similarity
            elif metric == "mean_evidence":
                v = t.retrieval.accumulated_evidence
            else:
                raise ValueError(f"unknown metric '{metric}'")
            per_list_values.setdefault(t.list_index, []).append(v)

    from drm_paper.battery import bootstrap_ci
    fa: dict[int, tuple[float, float, float]] = {}
    for li, vs in per_list_values.items():
        if not vs:
            continue
        mean = float(np.mean(vs))
        low, high = bootstrap_ci(vs, n_resamples=n_resamples)
        fa[li] = (mean, low, high)

    xs, ys, low_arr, high_arr, names = [], [], [], [], []
    for li, bas in bas_values.items():
        if li not in fa:
            continue
        mean, low, high = fa[li]
        xs.append(bas)
        ys.append(mean)
        low_arr.append(low)
        high_arr.append(high)
        names.append(lure_names.get(li, str(li)))

    xs_arr = np.array(xs)
    ys_arr = np.array(ys)
    yerr_lo = ys_arr - np.array(low_arr)
    yerr_hi = np.array(high_arr) - ys_arr

    # Pearson r and r² (handle degenerate constant case)
    if xs_arr.size >= 2 and float(xs_arr.std()) > 0 and float(ys_arr.std()) > 0:
        r = float(np.corrcoef(xs_arr, ys_arr)[0, 1])
        r2 = r * r
    else:
        r = float("nan")
        r2 = float("nan")

    # Least-squares regression
    if xs_arr.size >= 2 and float(xs_arr.std()) > 0:
        slope, intercept = np.polyfit(xs_arr, ys_arr, 1)
    else:
        slope = intercept = 0.0

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(
        xs_arr, ys_arr,
        yerr=np.vstack([yerr_lo, yerr_hi]),
        fmt="o", color="#1f4e79", ecolor="black",
        capsize=3, capthick=1.0, linewidth=1.0, markersize=7,
    )

    # Optionally annotate each point with the lure word. With 55 lists the
    # labels overlap into noise; default off. Set ``label_points=True`` for
    # a debug/exploration view.
    if label_points:
        for x, y, name in zip(xs_arr, ys_arr, names):
            ax.annotate(
                name, (x, y), xytext=(5, 5), textcoords="offset points",
                fontsize=8, color="black",
            )

    # Regression line
    if xs_arr.size >= 2:
        xline = np.linspace(xs_arr.min(), xs_arr.max(), 50)
        yline = slope * xline + intercept
        ax.plot(xline, yline, "--", color="gray", linewidth=1.0)

    ax.set_xlabel(bas_label)
    default_ylabels = {
        "fa_rate": "Substrate critical-lure FA rate",
        "mean_similarity": "Substrate mean similarity",
        "mean_evidence": "Substrate mean accumulated evidence",
    }
    ax.set_ylabel(y_label or default_ylabels[metric])
    if metric == "fa_rate":
        ax.set_ylim(-0.05, 1.10)
    if title:
        ax.set_title(title)
    if not np.isnan(r2):
        ax.text(
            0.05, 0.95, f"r² = {r2:.2f}  (r = {r:+.2f})",
            transform=ax.transAxes, fontsize=11,
            verticalalignment="top",
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="lightgray"),
        )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


def figure_ftt_am_dissociation(
    batteries_by_n_max: dict[int, BatteryResult],
    *,
    evidence_threshold: float,
    title: str | None = None,
):
    """Figure 4b: FTT--AM dissociation at the judgment level.

    Plots, for each item type, the fraction of trials whose accumulated
    evidence exceeds the given threshold, as a function of iteration depth
    N_max. The headline pattern to look for:

      - Distractor: stays near 0 at all N_max (no stepping stones).
      - Direct critical lure: high at all N_max (centroid match,
        gist abstraction succeeds even at single-step).
      - Mediated lure: low at N_max=1 (FTT regime — no direct gist match),
        rising at N_max>=2 as iterated retrieval accumulates activation
        through the centroid (AM regime).

    The flip in the mediated-lure curve is the substrate's mechanistic
    operationalization of the long-running FTT-vs-AM debate.
    """
    import matplotlib.pyplot as plt

    n_maxes = sorted(batteries_by_n_max.keys())
    item_order: tuple[ItemType, ...] = (
        "studied", "critical_lure", "mediated_lure", "held_out", "distractor"
    )
    labels = {
        "studied": "Studied",
        "critical_lure": "Critical lure",
        "mediated_lure": "Mediated lure",
        "held_out": "Held-out associate",
        "distractor": "Distractor",
    }
    colors = {
        "studied": "#1f4e79",
        "critical_lure": "#c0392b",
        "mediated_lure": "#e67e22",
        "held_out": "#7eb0d5",
        "distractor": "lightgray",
    }

    fig, ax = plt.subplots(figsize=(7, 4.5))

    for it in item_order:
        ys = []
        for n in n_maxes:
            trials = batteries_by_n_max[n].trials_of_type(it)
            if not trials:
                ys.append(0.0)
                continue
            rate = sum(
                1 for t in trials
                if t.retrieval.accumulated_evidence > evidence_threshold
            ) / len(trials)
            ys.append(rate)
        ax.plot(n_maxes, ys, marker="o", color=colors[it], label=labels[it])

    ax.set_xlabel("Iteration depth (N_max)")
    ax.set_ylabel(
        f"Recognition rate (evidence threshold = {evidence_threshold:.2f})"
    )
    ax.set_xticks(n_maxes)
    ax.set_ylim(-0.05, 1.05)
    if title:
        ax.set_title(title)
    ax.legend(loc="center right", frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


def figure_frontier_heatmap(
    sweep: FrontierSweep,
    *,
    metric: str = "lure_old_rate",
    title: str | None = None,
    n_resamples: int = 2000,
):
    """Figure 6: 2D heatmap over (EMA rate × novelty threshold).

    Args:
        metric: which row attribute of FrontierTabularRow to plot. Common:
            "lure_old_rate" (DRM intensity), "held_out_old_rate" (gist
            generalization), "distractor_old_rate" (false-alarm floor).
    """
    import matplotlib.pyplot as plt

    rows = frontier_to_table(sweep, n_resamples=n_resamples)
    ema_rates = sorted(set(r.ema_alpha for r in rows))
    novelties = sorted(set(r.novelty_threshold for r in rows))

    grid = np.full((len(ema_rates), len(novelties)), np.nan)
    for r in rows:
        i = ema_rates.index(r.ema_alpha)
        j = novelties.index(r.novelty_threshold)
        grid[i, j] = getattr(r, metric)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(
        grid, aspect="auto", origin="lower",
        cmap="viridis", vmin=0.0, vmax=1.0,
    )
    ax.set_xticks(np.arange(len(novelties)))
    ax.set_xticklabels([f"{n:.2f}" for n in novelties])
    ax.set_yticks(np.arange(len(ema_rates)))
    ax.set_yticklabels([f"{a:.2f}" for a in ema_rates])
    ax.set_xlabel("Novelty threshold")
    ax.set_ylabel("EMA rate α")

    # Annotate cells with the value
    for i in range(len(ema_rates)):
        for j in range(len(novelties)):
            v = grid[i, j]
            ax.text(
                j, i, f"{v:.2f}", ha="center", va="center",
                color="white" if v < 0.6 else "black", fontsize=9,
            )

    fig.colorbar(im, ax=ax, label=metric.replace("_", " "))
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig
