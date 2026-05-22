"""Pure key-value DRM experiment: bind each studied associate to a per-list
role token, store the bound vectors, and recognize at test by attempting to
match a role-bound probe against stored vectors exactly.

This is the complement of the headline similarity-content path. The paper
claims (§\\ref{sec:discussion}) that DRM-style false memory is the empirical
signature of the similarity-content path. The clean way to support that
claim is to show that pure key-value retrieval over the same content
produces *no* false memory: studied items recognized perfectly, every other
item type rejected at floor.

Mechanics per seed:

  1. Generate one random permutation ``role_N`` per list (the "key" for
     that list).
  2. Study phase: for each list ``N`` and each studied associate ``w``,
     store ``bind(encode(w), role_N)`` in a flat memory bank. No similarity
     processing.
  3. Test phase: for each test item ``q`` whose nominal list is ``N``,
     compute ``probe = bind(encode(q), role_N)`` and check whether ``probe``
     matches any stored vector exactly (all ``K`` indices identical). Mark
     as "old" if so, "new" otherwise. For distractors (no list assignment)
     we probe under every role and mark "old" if any role gives an exact
     match; in practice none does, since distractors are not in any
     studied vocabulary.

Predicted result: studied items recognized at 1.0 (correct role recovers
the stored binding exactly); critical lures, mediated lures, held-out
associates, and distractors all at 0.0 (none of these is in the studied
vocabulary, so no role-bound probe matches any stored vector).

Contrast: under the similarity-content path (the headline ConceptNet run),
critical lures false-alarm at ~0.51 and mediated/held-out at intermediate
rates. The same memory contents, accessed through the two paths, produce
the predicted dissociation.

Outputs:

  - ``results/keyvalue_experiment_summary.json``: per-item-type rates
    under pure KV, alongside the previously-recorded similarity-content
    headline rates for direct comparison.
  - ``results/figures/fig_keyvalue_dissociation.{pdf,png}``: paired bar
    chart contrasting the two paths.

Run from the repository root::

    .venv/bin/python drm_paper/run_keyvalue_experiment.py [--n-seeds N]
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cgm.vsa.hypervector import SparseBinaryHV
from cgm.vsa.operations import bind, random_permutation

from drm_paper.encoding import WordEncoder, make_conceptnet_factory
from drm_paper.stimuli import DEFAULT_FOILS, DRMList, load_stadler_full


DEFAULT_DATA = Path("drm_paper/data/stadler1999_55lists.json")
DEFAULT_NUMBERBATCH = Path("drm_paper/data/numberbatch-en-19.08.txt.gz")
DEFAULT_RESULTS_DIR = Path("drm_paper/results")
DEFAULT_FIGURES_DIR = DEFAULT_RESULTS_DIR / "figures"

ITEM_TYPES = ("studied", "critical_lure", "mediated_lure", "held_out", "distractor")


@dataclass(frozen=True)
class KVTrial:
    list_index: int
    critical_lure: str
    item_word: str
    item_type: str
    recognized: bool


@dataclass(frozen=True)
class KVSeedResult:
    seed: int
    n_stored: int
    n_trials: int
    elapsed_sec: float
    trials: list[KVTrial]


def _sample_test_items(
    dl: DRMList,
    studied: list[str],
    held_out: list[str],
    foil_pool: list[str],
    rng: np.random.Generator,
    studied_per_list: int = 6,
    distractors_per_list: int = 6,
) -> list[tuple[str, str]]:
    """Return (word, item_type) tuples for this list's test items.

    Mirrors the test composition in ``run_drm_experiment``: studied subset,
    critical lure, mediated lure (if present), held-out associates,
    distractors drawn from the foil pool.
    """
    out: list[tuple[str, str]] = []
    studied_sample = list(rng.choice(studied, size=min(studied_per_list, len(studied)),
                                     replace=False))
    out.extend((w, "studied") for w in studied_sample)
    out.append((dl.critical_lure, "critical_lure"))
    if dl.mediated_lure:
        out.append((dl.mediated_lure, "mediated_lure"))
    out.extend((w, "held_out") for w in held_out)
    distractor_sample = list(rng.choice(foil_pool,
                                        size=min(distractors_per_list, len(foil_pool)),
                                        replace=False))
    out.extend((w, "distractor") for w in distractor_sample)
    return out


def run_kv_seed(
    drm_lists: tuple[DRMList, ...],
    encoder: WordEncoder,
    *,
    role_seed: int,
    study_seed: int,
    test_seed: int,
    foil_pool: tuple[str, ...] = DEFAULT_FOILS,
    held_out_per_list: int = 1,
    studied_per_list: int = 6,
    distractors_per_list: int = 6,
) -> KVSeedResult:
    """Single seed of the pure key-value DRM experiment."""
    t0 = time.time()
    rng_roles = np.random.default_rng(role_seed)
    rng_study = np.random.default_rng(study_seed)
    rng_test = np.random.default_rng(test_seed)

    D = encoder.D
    roles = [random_permutation(D, rng_roles) for _ in drm_lists]

    # ---- Study phase ----
    # stored_indices_set : set of tuple(indices) for fast exact-match lookup.
    # stored_meta        : parallel list of (list_index, word) for diagnostics.
    stored_indices_set: set[tuple[int, ...]] = set()
    stored_meta: list[tuple[int, str]] = []
    held_out_per_list_words: dict[int, list[str]] = {}
    studied_per_list_words: dict[int, list[str]] = {}
    for list_idx, dl in enumerate(drm_lists):
        associates = list(dl.associates)
        if len(associates) < held_out_per_list + 1:
            raise ValueError(
                f"list {list_idx} ({dl.critical_lure}) has too few "
                f"associates ({len(associates)}) for held_out_per_list="
                f"{held_out_per_list}"
            )
        # Choose which associates are held out (not studied) for the
        # gist-generalization probe at test time.
        perm = rng_study.permutation(len(associates))
        held_out_idxs = perm[:held_out_per_list].tolist()
        studied_idxs = perm[held_out_per_list:].tolist()
        held_out_words = [associates[i] for i in held_out_idxs]
        studied_words = [associates[i] for i in studied_idxs]
        held_out_per_list_words[list_idx] = held_out_words
        studied_per_list_words[list_idx] = studied_words

        for w in studied_words:
            encoded = encoder.encode(w)
            bound = bind(encoded, roles[list_idx])
            key = tuple(int(i) for i in bound.indices)
            stored_indices_set.add(key)
            stored_meta.append((list_idx, w))

    # ---- Test phase ----
    # Pure KV recognition rule: for a probe item q with associated list N,
    # compute bind(encode(q), role_N) and check exact set membership against
    # the stored indices set. Distractors (no list assignment) are tried
    # under every role.
    trials: list[KVTrial] = []
    for list_idx, dl in enumerate(drm_lists):
        items = _sample_test_items(
            dl,
            studied_per_list_words[list_idx],
            held_out_per_list_words[list_idx],
            list(foil_pool),
            rng_test,
            studied_per_list=studied_per_list,
            distractors_per_list=distractors_per_list,
        )
        for word, item_type in items:
            q_encoded = encoder.encode(word)
            recognized = False
            if item_type == "distractor":
                # No nominal list. Try every role.
                for r in roles:
                    probe = bind(q_encoded, r)
                    if tuple(int(i) for i in probe.indices) in stored_indices_set:
                        recognized = True
                        break
            else:
                # Probe with the test item's nominal list role.
                probe = bind(q_encoded, roles[list_idx])
                if tuple(int(i) for i in probe.indices) in stored_indices_set:
                    recognized = True
            trials.append(
                KVTrial(
                    list_index=list_idx,
                    critical_lure=dl.critical_lure,
                    item_word=word,
                    item_type=item_type,
                    recognized=recognized,
                )
            )

    elapsed = time.time() - t0
    return KVSeedResult(
        seed=role_seed,
        n_stored=len(stored_meta),
        n_trials=len(trials),
        elapsed_sec=elapsed,
        trials=trials,
    )


def aggregate_kv_rates(per_seed: list[KVSeedResult]) -> dict[str, dict]:
    """Mean recognition rate by item type across seeds, with bootstrap CIs."""
    by_type: dict[str, list[float]] = {it: [] for it in ITEM_TYPES}
    for seed_res in per_seed:
        bucket: dict[str, list[int]] = {it: [] for it in ITEM_TYPES}
        for t in seed_res.trials:
            bucket[t.item_type].append(1 if t.recognized else 0)
        for it, vals in bucket.items():
            if vals:
                by_type[it].append(float(np.mean(vals)))

    out: dict[str, dict] = {}
    for it, rates in by_type.items():
        if not rates:
            out[it] = {"n_seeds": 0, "mean": 0.0, "ci_low": 0.0, "ci_high": 0.0}
            continue
        arr = np.asarray(rates)
        if arr.size < 2:
            out[it] = {"n_seeds": arr.size, "mean": float(arr.mean()),
                       "ci_low": float(arr.mean()), "ci_high": float(arr.mean())}
            continue
        rng_boot = np.random.default_rng(0)
        idx = rng_boot.integers(0, arr.size, size=(5000, arr.size))
        boots = arr[idx].mean(axis=1)
        out[it] = {
            "n_seeds": int(arr.size),
            "mean": float(arr.mean()),
            "ci_low": float(np.quantile(boots, 0.025)),
            "ci_high": float(np.quantile(boots, 0.975)),
        }
    return out


def make_figure(
    kv_rates: dict[str, dict],
    sim_content_rates: dict[str, dict] | None,
    out_pdf: Path,
    out_png: Path,
    title: str | None = None,
) -> None:
    """Paired bar chart: similarity-content vs key-value rates by item type."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ["Studied", "Critical\nlure", "Mediated\nlure", "Held-out\nassociate",
              "Distractor"]
    keys = list(ITEM_TYPES)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    x = np.arange(len(labels))
    width = 0.36

    kv_means = [kv_rates[k]["mean"] for k in keys]
    kv_err_lo = [kv_rates[k]["mean"] - kv_rates[k]["ci_low"] for k in keys]
    kv_err_hi = [kv_rates[k]["ci_high"] - kv_rates[k]["mean"] for k in keys]

    if sim_content_rates is not None:
        sc_means = [sim_content_rates[k]["mean"] for k in keys]
        sc_err_lo = [sim_content_rates[k]["mean"] - sim_content_rates[k]["ci_low"]
                     for k in keys]
        sc_err_hi = [sim_content_rates[k]["ci_high"] - sim_content_rates[k]["mean"]
                     for k in keys]
        ax.bar(x - width/2, sc_means, width,
               yerr=np.array([sc_err_lo, sc_err_hi]),
               label="Similarity-content path",
               color="#2c5f7a", capsize=3)
        ax.bar(x + width/2, kv_means, width,
               yerr=np.array([kv_err_lo, kv_err_hi]),
               label="Pure key-value path",
               color="#c2784a", capsize=3)
    else:
        ax.bar(x, kv_means, width * 2,
               yerr=np.array([kv_err_lo, kv_err_hi]),
               label="Pure key-value path",
               color="#c2784a", capsize=3)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Old-response rate")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _vocab_from_lists(drm_lists: tuple[DRMList, ...],
                      foil_pool: tuple[str, ...]) -> set[str]:
    vocab: set[str] = set(foil_pool)
    for dl in drm_lists:
        vocab.add(dl.critical_lure)
        vocab.update(dl.associates)
        if dl.mediated_lure:
            vocab.add(dl.mediated_lure)
    return vocab


def _derive_seeds(master_seed: int, i: int) -> tuple[int, int, int, int]:
    rng = np.random.default_rng(
        np.uint64((master_seed * 1_000_003 + i * 7919) & 0xFFFFFFFF)
    )
    s = rng.integers(0, 2**31 - 1, size=4)
    return int(s[0]), int(s[1]), int(s[2]), int(s[3])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-seeds", type=int, default=20)
    parser.add_argument("--master-seed", type=int, default=0)
    parser.add_argument("--lists-path", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--numberbatch-path", type=Path, default=DEFAULT_NUMBERBATCH)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES_DIR)
    args = parser.parse_args()

    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    drm_lists = tuple(load_stadler_full(args.lists_path))
    vocab = _vocab_from_lists(drm_lists, DEFAULT_FOILS)
    print(f"Lists: {len(drm_lists)}; vocab: {len(vocab)}")
    print(f"Loading ConceptNet from {args.numberbatch_path}")
    factory = make_conceptnet_factory(args.numberbatch_path, vocab)

    per_seed: list[KVSeedResult] = []
    t0 = time.time()
    for i in range(args.n_seeds):
        role_s, study_s, test_s, enc_s = _derive_seeds(args.master_seed, i)
        encoder = factory(enc_s)
        # Sanity check: every word must be encodable. For ConceptNet, words
        # not in Numberbatch raise KeyError; we want to surface that early.
        missing = list(getattr(encoder, "missing_words", ()))
        # The shared foil pool may include words not in Numberbatch (rare).
        # Filter to encodable foils for this seed only.
        usable_foils = tuple(w for w in DEFAULT_FOILS if w not in missing)
        result = run_kv_seed(
            drm_lists, encoder,
            role_seed=role_s, study_seed=study_s, test_seed=test_s,
            foil_pool=usable_foils,
        )
        per_seed.append(result)
        print(
            f"  seed {i+1}/{args.n_seeds}: "
            f"stored={result.n_stored}, trials={result.n_trials}, "
            f"elapsed={result.elapsed_sec:.1f}s"
        )

    elapsed = time.time() - t0
    print(f"\nTotal: {args.n_seeds} seeds in {elapsed:.1f}s")

    kv_rates = aggregate_kv_rates(per_seed)
    print("\n== Pure key-value recognition rates ==")
    for it in ITEM_TYPES:
        r = kv_rates[it]
        print(
            f"  {it:<14s} mean={r['mean']:.3f} "
            f"[{r['ci_low']:.3f}, {r['ci_high']:.3f}]  n_seeds={r['n_seeds']}"
        )

    # Try to load similarity-content headline rates for comparison.
    sim_content_rates: dict[str, dict] | None = None
    headline_json = args.results_dir / "headline_battery_summary.json"
    if headline_json.exists():
        sc_payload = json.loads(headline_json.read_text())
        sim_content_rates = sc_payload.get("recognition_by_item_type")

    summary = {
        "experiment": "pure_key_value",
        "encoder": "conceptnet_numberbatch",
        "n_seeds": args.n_seeds,
        "master_seed": args.master_seed,
        "n_lists": len(drm_lists),
        "elapsed_sec": elapsed,
        "key_value_rates": kv_rates,
        "similarity_content_rates": sim_content_rates,
        "per_seed": [
            {
                "seed": r.seed, "n_stored": r.n_stored,
                "n_trials": r.n_trials, "elapsed_sec": r.elapsed_sec,
            }
            for r in per_seed
        ],
    }
    out_json = args.results_dir / "keyvalue_experiment_summary.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved summary to {out_json}")

    make_figure(
        kv_rates, sim_content_rates,
        out_pdf=args.figures_dir / "fig_keyvalue_dissociation.pdf",
        out_png=args.figures_dir / "fig_keyvalue_dissociation.png",
        title="Dual-path dissociation: similarity-content vs pure "
              "key-value retrieval (ConceptNet, 55 lists, 20 seeds)",
    )
    print("Figure written.")


if __name__ == "__main__":
    main()
