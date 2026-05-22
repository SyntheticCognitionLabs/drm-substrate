"""Run the 55-list DRM battery with the NelsonAssociationEncoder.

Mirrors the ConceptNet headline protocol (N_max = 1 single-step
best-similarity recognition, 20 seeds, 55 lists). Produces:

  - results/nelson_battery_summary.json: per-item-type recognition stats
    (mean, bootstrap CI) and per-list critical-lure FA rate.
  - results/figures/fig2_recognition_rates_nelson.{pdf,png}
  - results/figures/fig3c_bas_regression_nelson.{pdf,png}

Run from the repository root::

    .venv/bin/python drm_paper/run_nelson_battery.py [--n-seeds N]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from drm_paper.analysis import per_list_lure_fa_rate
from drm_paper.battery import (
    AggregateStats,
    BatteryResult,
    aggregate_stats,
    per_list_lure_fa_rate_with_ci,
    run_battery,
)
from drm_paper.encoding import make_nelson_factory
from drm_paper.experiment import ExperimentConfig
from drm_paper.figures import figure_bas_regression, figure_recognition_rates
from drm_paper.stimuli import load_stadler_full


DEFAULT_DATA = Path("drm_paper/data/stadler1999_55lists.json")
DEFAULT_NELSON = Path("drm_paper/data/nelson_norms.json")
DEFAULT_RESULTS_DIR = Path("drm_paper/results")
DEFAULT_FIGURES_DIR = DEFAULT_RESULTS_DIR / "figures"


def _vocab_from_lists(lists) -> set[str]:
    vocab: set[str] = set()
    for L in lists:
        vocab.add(L.critical_lure.lower())
        vocab.update(a.lower() for a in L.associates)
        if L.mediated_lure:
            vocab.add(L.mediated_lure.lower())
    return vocab


def _serialize_agg(it: str, agg: AggregateStats) -> dict:
    return {
        "item_type": it,
        "n_seeds": agg.n_seeds,
        "n_trials_per_seed": agg.n_trials_per_seed,
        "mean_old_rate": agg.mean_old_rate,
        "ci_low": agg.ci_low,
        "ci_high": agg.ci_high,
        "mean_remember_rate": agg.mean_remember_rate,
        "mean_know_rate": agg.mean_know_rate,
        "mean_similarity": agg.mean_similarity,
        "mean_episodic_similarity": agg.mean_episodic_similarity,
        "mean_semantic_similarity": agg.mean_semantic_similarity,
        "mean_accumulated_evidence": agg.mean_accumulated_evidence,
        "per_seed_old_rates": list(agg.per_seed_old_rates),
    }


def _per_list_metrics(battery: BatteryResult, lists) -> dict[int, dict]:
    """Per-list aggregate metrics for BAS-regression purposes.

    For each list index, returns:
        fa_rate:   mean across seeds of (critical_lure trials judged old / total)
        evidence:  mean across seeds of (mean accumulated evidence for critical-lure trials)
        similarity: mean across seeds of (mean best-similarity for critical-lure trials)
    """
    out: dict[int, dict] = {}
    for r in battery.per_seed_results:
        per_list_hits: dict[int, list[int]] = {}
        per_list_ev: dict[int, list[float]] = {}
        per_list_sim: dict[int, list[float]] = {}
        for t in r.trials_of_type("critical_lure"):
            per_list_hits.setdefault(t.list_index, []).append(
                1 if t.judgment.judgment != "New" else 0
            )
            per_list_ev.setdefault(t.list_index, []).append(
                float(t.retrieval.accumulated_evidence)
            )
            per_list_sim.setdefault(t.list_index, []).append(
                float(t.retrieval.best_overall_similarity)
            )
        for li in per_list_hits:
            d = out.setdefault(li, {"fa_rate": [], "evidence": [], "similarity": []})
            d["fa_rate"].append(sum(per_list_hits[li]) / len(per_list_hits[li]))
            d["evidence"].append(float(np.mean(per_list_ev[li])))
            d["similarity"].append(float(np.mean(per_list_sim[li])))

    # Collapse to per-list scalars (mean across seeds)
    collapsed: dict[int, dict] = {}
    for li, d in out.items():
        collapsed[li] = {
            "fa_rate": float(np.mean(d["fa_rate"])),
            "evidence": float(np.mean(d["evidence"])),
            "similarity": float(np.mean(d["similarity"])),
        }
    return collapsed


def _bas_regression(
    battery: BatteryResult, lists, n_resamples: int = 10_000
) -> dict:
    """Per-list critical-lure metric vs published BAS regression.

    Mirrors the analysis used for the ConceptNet headline. The headline
    metric is mean accumulated evidence (continuous, less noisy than the
    binary FA rate); we also report FA-rate and best-similarity for
    completeness.
    """
    per_list = _per_list_metrics(battery, lists)
    metric_data: dict[str, list[tuple[float, float]]] = {
        "fa_rate": [], "evidence": [], "similarity": [],
    }
    per_list_rows: list[tuple[int, str, float, float, float, float]] = []
    for li in sorted(per_list):
        bas = lists[li].published_bas
        if bas is None:
            continue
        d = per_list[li]
        metric_data["fa_rate"].append((bas, d["fa_rate"]))
        metric_data["evidence"].append((bas, d["evidence"]))
        metric_data["similarity"].append((bas, d["similarity"]))
        per_list_rows.append(
            (li, lists[li].critical_lure, bas,
             d["fa_rate"], d["evidence"], d["similarity"])
        )

    out: dict = {"n": len(per_list_rows), "per_list": per_list_rows}
    for metric_name, pairs in metric_data.items():
        xs = np.asarray([p[0] for p in pairs], dtype=np.float64)
        ys = np.asarray([p[1] for p in pairs], dtype=np.float64)
        if xs.size < 3 or np.std(xs) == 0 or np.std(ys) == 0:
            out[metric_name] = {"r": float("nan"), "r2": float("nan")}
        else:
            r = float(np.corrcoef(xs, ys)[0, 1])
            out[metric_name] = {"r": r, "r2": r * r}
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-seeds", type=int, default=20)
    parser.add_argument("--master-seed", type=int, default=0)
    parser.add_argument("--n-max", type=int, default=1)
    parser.add_argument(
        "--lists-path", type=Path, default=DEFAULT_DATA
    )
    parser.add_argument(
        "--nelson-path", type=Path, default=DEFAULT_NELSON
    )
    parser.add_argument(
        "--results-dir", type=Path, default=DEFAULT_RESULTS_DIR
    )
    parser.add_argument(
        "--figures-dir", type=Path, default=DEFAULT_FIGURES_DIR
    )
    parser.add_argument(
        "--load-cached", action="store_true",
        help="Skip the battery run and reload from "
             "results/nelson_battery_cache.pickle (fast iteration on "
             "regression / figure code)."
    )
    args = parser.parse_args()

    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    import pickle
    cache_path = args.results_dir / "nelson_battery_cache.pickle"

    if args.load_cached:
        if not cache_path.exists():
            raise FileNotFoundError(
                f"--load-cached set but cache file {cache_path} not found. "
                "Run the battery first (without --load-cached) to populate it."
            )
        with cache_path.open("rb") as f:
            cache = pickle.load(f)
        battery = cache["battery"]
        keep_lists = cache["keep_lists"]
        elapsed = cache["elapsed"]
        missing = cache["missing"]
        unresolved = cache["unresolved"]
        lists = load_stadler_full(args.lists_path)
        vocab = _vocab_from_lists(lists)
        print(
            f"Loaded cached battery: {battery.n_seeds} seeds, "
            f"{len(keep_lists)} lists, original elapsed {elapsed:.1f}s"
        )
        print(
            f"  unresolved (random fallback): {len(unresolved)} of {len(vocab)}"
        )
    else:
        lists = load_stadler_full(args.lists_path)
        vocab = _vocab_from_lists(lists)

        print(f"Lists: {len(lists)}; vocab size: {len(vocab)}")
        print(f"Loading Nelson norms from {args.nelson_path}")
        factory = make_nelson_factory(
            args.nelson_path, vocabulary=vocab, D=2000, K=64
        )

        # Probe missing words once via a seed-0 instance
        probe = factory(0)
        missing = probe.missing_words
        unresolved = probe.missing_after_fallback
        print(f"Nelson vocabulary: {probe.vocab_size:,} words")
        print(
            f"DRM-vocab missing from Nelson directly: "
            f"{len(missing)} of {len(vocab)}"
        )
        print(
            f"DRM-vocab missing after stem fallback:  "
            f"{len(unresolved)} of {len(vocab)} "
            f"({100*len(unresolved)/len(vocab):.1f}%)"
        )
        if unresolved:
            print(f"  unresolved (random fallback): {unresolved[:10]}...")

        # We keep all 55 lists. Words not in Nelson (after stem fallback)
        # receive a deterministic random sparse code -- valid hypervectors
        # with no semantic structure. This is the honest accounting: ~6% of
        # the DRM vocabulary falls outside Nelson, and we disclose the
        # fallback rate in the paper rather than dropping lists.
        keep_lists = lists

        base_config = ExperimentConfig(n_max=args.n_max)
        t0 = time.time()
        battery = run_battery(
            tuple(keep_lists),
            encoder_factory=factory,
            base_config=base_config,
            n_seeds=args.n_seeds,
            master_seed=args.master_seed,
            encoder_name="nelson",
        )
        elapsed = time.time() - t0
        print(
            f"\nBattery: {args.n_seeds} seeds across {len(keep_lists)} "
            f"lists in {elapsed:.1f}s"
        )

        # Persist battery to disk so downstream crashes don't lose the
        # run; lets us iterate on regression / figure code without
        # re-running the 16-min battery.
        with cache_path.open("wb") as f:
            pickle.dump(
                {"battery": battery, "keep_lists": keep_lists,
                 "elapsed": elapsed, "missing": missing,
                 "unresolved": unresolved}, f
            )
        print(f"Cached battery to {cache_path}")

    agg = aggregate_stats(battery)
    print("\n== Recognition rates by item type (Nelson encoder, N_max=1) ==")
    for it in ["studied", "critical_lure", "mediated_lure", "held_out", "distractor"]:
        s = agg[it]
        print(
            f"  {it:<14s} old={s.mean_old_rate:.3f} "
            f"[{s.ci_low:.3f},{s.ci_high:.3f}] "
            f"R={s.mean_remember_rate:.3f} K={s.mean_know_rate:.3f} "
            f"sim={s.mean_similarity:.3f}"
        )

    bas_reg = _bas_regression(battery, keep_lists, n_resamples=2_000)
    print(
        f"\nBAS regression vs Roediger 2001 BAS  (n={bas_reg['n']} lists):"
    )
    for m in ("fa_rate", "evidence", "similarity"):
        r = bas_reg[m]["r"]
        r2 = bas_reg[m]["r2"]
        print(f"  {m:<12s}  r={r:+.3f}  r2={r2:.3f}")

    # Save summary JSON
    summary = {
        "encoder": "nelson",
        "n_seeds": args.n_seeds,
        "master_seed": args.master_seed,
        "n_max": args.n_max,
        "elapsed_sec": elapsed,
        "n_lists_used": len(keep_lists),
        "n_lists_total": len(lists),
        "n_drm_vocab": len(vocab),
        "missing_from_nelson_direct": len(missing),
        "missing_after_stem_fallback": len(unresolved),
        "unresolved_words": list(unresolved),
        "recognition_by_item_type": {
            it: _serialize_agg(it, agg[it])
            for it in ["studied", "critical_lure", "mediated_lure",
                       "held_out", "distractor"]
        },
        "bas_regression": {
            "n_lists_in_regression": bas_reg["n"],
            "by_metric": {
                m: {"r": bas_reg[m]["r"], "r2": bas_reg[m]["r2"]}
                for m in ("fa_rate", "evidence", "similarity")
            },
            "per_list": [
                {
                    "list_index": li,
                    "critical_lure": lure,
                    "published_bas": bas,
                    "substrate_fa_rate": fa,
                    "substrate_mean_evidence": ev,
                    "substrate_mean_similarity": sim,
                }
                for (li, lure, bas, fa, ev, sim) in bas_reg["per_list"]
            ],
        },
    }
    summary_path = args.results_dir / "nelson_battery_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved summary to {summary_path}")

    # Figures
    print("Generating figures...")
    fig_recog = figure_recognition_rates(
        battery, n_resamples=2000,
        title="Recognition rates by item type "
              "(Nelson encoder, 55 lists, 20 seeds)",
    )
    fig_recog.savefig(
        args.figures_dir / "fig2_recognition_rates_nelson.pdf", bbox_inches="tight"
    )
    fig_recog.savefig(
        args.figures_dir / "fig2_recognition_rates_nelson.png",
        dpi=150, bbox_inches="tight"
    )

    bas_values = {
        li: keep_lists[li].published_bas
        for li in range(len(keep_lists))
        if keep_lists[li].published_bas is not None
    }
    if bas_values:
        fig_bas = figure_bas_regression(
            battery, bas_values, n_resamples=2000,
            metric="mean_evidence",
            bas_label="Roediger 2001 BAS",
            title="Per-list critical-lure evidence vs Roediger 2001 BAS\n(Nelson encoder)",
        )
        fig_bas.savefig(
            args.figures_dir / "fig3c_bas_regression_nelson.pdf",
            bbox_inches="tight"
        )
        fig_bas.savefig(
            args.figures_dir / "fig3c_bas_regression_nelson.png",
            dpi=150, bbox_inches="tight"
        )
    print("Done.")


if __name__ == "__main__":
    main()
