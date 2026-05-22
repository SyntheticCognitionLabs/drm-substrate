"""Run the 55-list ConceptNet similarity-content baseline battery.

Produces ``results/headline_battery_summary.json`` in the same per-item-
type-rates format as ``run_keyvalue_experiment.py``, so the key-value
runner can load it as the comparison condition for the dual-path
dissociation figure.

Same protocol as the headline DRM phenomenology run: ConceptNet
Numberbatch encoder, 55-list Roediger 2001 battery, N_max=1,
best-similarity recognition at threshold 0.30. 20 seeds with bootstrap
CIs.

Run from the repository root::

    .venv/bin/python drm_paper/run_headline_battery.py [--n-seeds N]
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np

from drm_paper.battery import aggregate_stats, run_battery
from drm_paper.encoding import make_conceptnet_factory
from drm_paper.experiment import ExperimentConfig
from drm_paper.figures import figure_bas_regression
from drm_paper.stimuli import DEFAULT_FOILS, load_stadler_full


DEFAULT_DATA = Path("drm_paper/data/stadler1999_55lists.json")
DEFAULT_NUMBERBATCH = Path("drm_paper/data/numberbatch-en-19.08.txt.gz")
DEFAULT_RESULTS_DIR = Path("drm_paper/results")
DEFAULT_FIGURES_DIR = DEFAULT_RESULTS_DIR / "figures"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-seeds", type=int, default=20)
    parser.add_argument("--master-seed", type=int, default=0)
    parser.add_argument("--lists-path", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--numberbatch-path", type=Path, default=DEFAULT_NUMBERBATCH)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES_DIR)
    parser.add_argument(
        "--load-cached", action="store_true",
        help="Skip the battery and reload from results/headline_battery_cache.pickle.",
    )
    args = parser.parse_args()

    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    cache_path = args.results_dir / "headline_battery_cache.pickle"

    drm_lists = tuple(load_stadler_full(args.lists_path))

    if args.load_cached:
        if not cache_path.exists():
            raise FileNotFoundError(
                f"--load-cached set but {cache_path} not found. Run without "
                "the flag first to populate the cache."
            )
        with cache_path.open("rb") as f:
            cache = pickle.load(f)
        battery = cache["battery"]
        elapsed = cache["elapsed"]
        print(
            f"Loaded cached headline battery: {battery.n_seeds} seeds, "
            f"{len(drm_lists)} lists, original elapsed {elapsed:.1f}s"
        )
    else:
        vocab = set(DEFAULT_FOILS)
        for dl in drm_lists:
            vocab.add(dl.critical_lure)
            vocab.update(dl.associates)
            if dl.mediated_lure:
                vocab.add(dl.mediated_lure)
        print(f"Lists: {len(drm_lists)}; vocab: {len(vocab)}")
        print(f"Loading ConceptNet from {args.numberbatch_path}")
        factory = make_conceptnet_factory(args.numberbatch_path, vocab)

        base_config = ExperimentConfig(n_max=1)
        t0 = time.time()
        battery = run_battery(
            drm_lists, factory, base_config,
            n_seeds=args.n_seeds, master_seed=args.master_seed,
            encoder_name="conceptnet_numberbatch",
        )
        elapsed = time.time() - t0
        print(f"\nBattery: {args.n_seeds} seeds in {elapsed:.1f}s")

        with cache_path.open("wb") as f:
            pickle.dump({"battery": battery, "elapsed": elapsed}, f)
        print(f"Cached battery to {cache_path}")

    agg = aggregate_stats(battery, n_resamples=5000)
    rates = {}
    for it in ("studied", "critical_lure", "mediated_lure", "held_out", "distractor"):
        s = agg[it]
        rates[it] = {
            "n_seeds": s.n_seeds,
            "mean": s.mean_old_rate,
            "ci_low": s.ci_low,
            "ci_high": s.ci_high,
            "mean_remember_rate": s.mean_remember_rate,
            "mean_know_rate": s.mean_know_rate,
        }

    print("\n== ConceptNet 55-list similarity-content rates ==")
    for it, r in rates.items():
        print(
            f"  {it:<14s} mean={r['mean']:.3f} "
            f"[{r['ci_low']:.3f}, {r['ci_high']:.3f}]"
        )

    summary = {
        "experiment": "similarity_content_baseline",
        "encoder": "conceptnet_numberbatch",
        "n_seeds": args.n_seeds,
        "master_seed": args.master_seed,
        "n_lists": len(drm_lists),
        "elapsed_sec": elapsed,
        "recognition_by_item_type": rates,
    }
    out_json = args.results_dir / "headline_battery_summary.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved summary to {out_json}")

    # ---- Figure regeneration: fig3c (BAS regression) + fig3d (human comparison) ----
    bas_values = {
        li: drm_lists[li].published_bas
        for li in range(len(drm_lists))
        if drm_lists[li].published_bas is not None
    }
    if bas_values:
        fig_bas = figure_bas_regression(
            battery, bas_values, metric="mean_evidence",
            bas_label="Roediger 2001 BAS",
            title="Per-list critical-lure evidence vs Roediger 2001 BAS "
                  "(ConceptNet, 55 lists, 20 seeds)",
            n_resamples=5000, label_points=False,
        )
        fig_bas.savefig(
            args.figures_dir / "fig3c_bas_regression_evidence.pdf",
            bbox_inches="tight",
        )
        fig_bas.savefig(
            args.figures_dir / "fig3c_bas_regression_evidence.png",
            dpi=150, bbox_inches="tight",
        )
        print(f"Saved fig3c to {args.figures_dir}")

    human_rates = {
        li: drm_lists[li].published_fa_rate
        for li in range(len(drm_lists))
        if drm_lists[li].published_fa_rate is not None
    }
    if human_rates:
        fig_human = figure_bas_regression(
            battery, human_rates, metric="mean_evidence",
            bas_label="Published human recognition rate (Roediger 2001)",
            title="Substrate per-list evidence vs human recognition "
                  "(ConceptNet, 55 lists, 20 seeds)",
            n_resamples=5000, label_points=False,
        )
        fig_human.savefig(
            args.figures_dir / "fig3d_human_evidence.pdf",
            bbox_inches="tight",
        )
        fig_human.savefig(
            args.figures_dir / "fig3d_human_evidence.png",
            dpi=150, bbox_inches="tight",
        )
        print(f"Saved fig3d to {args.figures_dir}")


if __name__ == "__main__":
    main()
