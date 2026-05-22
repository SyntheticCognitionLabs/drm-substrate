"""Multi-N_max sweep for the evidence-accumulation figure (fig4).

Runs the ConceptNet 55-list battery at each iteration depth
$N_{\\max} \\in \\{1, 2, 3, 5\\}$ and saves the resulting
``figure_evidence_by_iteration`` panel. The figure shows how
accumulated retrieval evidence grows with N_max for each item type;
the slopes are the mechanistic story behind the FTT--AM dissociation
(fig4b): mediated lures gain ground on direct lures as iteration
depth increases because spreading retrieval finds them via the
cluster centroid.

Caches the per-N_max batteries to disk so the figure can be
regenerated quickly without re-running the sweep.

Run::

    .venv/bin/python drm_paper/run_evidence_sweep.py [--n-seeds 10]
"""

from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import numpy as np

from drm_paper.battery import run_battery
from drm_paper.encoding import make_conceptnet_factory
from drm_paper.experiment import ExperimentConfig
from drm_paper.figures import figure_evidence_by_iteration
from drm_paper.stimuli import DEFAULT_FOILS, load_stadler_full


DEFAULT_DATA = Path("drm_paper/data/stadler1999_55lists.json")
DEFAULT_NUMBERBATCH = Path("drm_paper/data/numberbatch-en-19.08.txt.gz")
DEFAULT_RESULTS_DIR = Path("drm_paper/results")
DEFAULT_FIGURES_DIR = DEFAULT_RESULTS_DIR / "figures"
DEFAULT_N_MAX = (1, 2, 3, 5)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--master-seed", type=int, default=0)
    parser.add_argument("--lists-path", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--numberbatch-path", type=Path, default=DEFAULT_NUMBERBATCH)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES_DIR)
    parser.add_argument("--n-max-values", type=int, nargs="+", default=list(DEFAULT_N_MAX))
    parser.add_argument(
        "--load-cached", action="store_true",
        help="Skip the sweep and reload from results/evidence_sweep_cache.pickle.",
    )
    args = parser.parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    cache_path = args.results_dir / "evidence_sweep_cache.pickle"

    if args.load_cached:
        with cache_path.open("rb") as f:
            cache = pickle.load(f)
        batteries_by_n_max = cache["batteries_by_n_max"]
        print(f"Loaded cached sweep: N_max values = {sorted(batteries_by_n_max)}")
    else:
        drm_lists = tuple(load_stadler_full(args.lists_path))
        vocab = set(DEFAULT_FOILS)
        for dl in drm_lists:
            vocab.add(dl.critical_lure)
            vocab.update(dl.associates)
            if dl.mediated_lure:
                vocab.add(dl.mediated_lure)
        print(f"Lists: {len(drm_lists)}; vocab: {len(vocab)}")
        print(f"Loading ConceptNet from {args.numberbatch_path}")
        factory = make_conceptnet_factory(args.numberbatch_path, vocab)

        batteries_by_n_max = {}
        t_total = time.time()
        for n_max in args.n_max_values:
            t0 = time.time()
            cfg = ExperimentConfig(n_max=n_max)
            battery = run_battery(
                drm_lists, factory, cfg,
                n_seeds=args.n_seeds, master_seed=args.master_seed,
                encoder_name="conceptnet_numberbatch",
            )
            batteries_by_n_max[n_max] = battery
            print(
                f"  N_max={n_max}: {args.n_seeds} seeds, "
                f"{len(battery.all_trials())} trials, "
                f"{time.time() - t0:.1f}s"
            )
        print(f"\nTotal sweep: {time.time() - t_total:.1f}s")

        with cache_path.open("wb") as f:
            pickle.dump(
                {"batteries_by_n_max": batteries_by_n_max,
                 "master_seed": args.master_seed,
                 "n_seeds": args.n_seeds},
                f,
            )
        print(f"Cached sweep to {cache_path}")

    # Generate fig4 from the sweep
    fig = figure_evidence_by_iteration(
        batteries_by_n_max,
        title="Accumulated recognition evidence by item type vs "
              "iteration depth (ConceptNet, 55 lists, 10 seeds)",
    )
    fig.savefig(args.figures_dir / "fig4_evidence_by_iteration.pdf",
                bbox_inches="tight")
    fig.savefig(args.figures_dir / "fig4_evidence_by_iteration.png",
                dpi=150, bbox_inches="tight")
    print(f"Saved fig4 to {args.figures_dir}")


if __name__ == "__main__":
    main()
