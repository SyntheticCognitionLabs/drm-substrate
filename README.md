# DRM substrate

Reproduction code for the manuscript *The architecture of false memory: a non-parametric account of DRM and the FTT–AM dissociation*, currently under double-blind review at *Open Mind: Discoveries in Cognitive Science*.

The substrate is a non-parametric memory architecture (sparse-binary vector-symbolic representations, content-addressable episodic and semantic stores, replay-driven consolidation, dual-path retrieval) that reproduces canonical DRM false-memory phenomenology without any learned parameters. All experiments reported in the manuscript can be regenerated from this repository.

## Layout

```
cgm/                          # the memory substrate (DRM-relevant subset)
    vsa/                      # sparse-binary hypervectors + operations
    memory/                   # content-addressable stores + HNSW index
drm_paper/                    # paper experiments
    stimuli.py                # 55-list Roediger 2001 / Stadler 1999 battery
    encoding.py               # four encoders: random, Brown, ConceptNet, Nelson
    retrieval.py              # iterated retrieval + Remember/Know judgment
    experiment.py             # single-seed DRM experiment runner
    battery.py                # multi-seed battery + bootstrap CIs
    figures.py                # paper figure generators
    sweeps.py                 # fidelity-generalization 2-d sweep
    analysis.py               # recognition-rate summary + R/K dissociation
    run_headline_battery.py   # ConceptNet 55-list battery (fig2, fig3c, fig3d)
    run_nelson_battery.py     # Nelson encoder 55-list battery
    run_evidence_sweep.py     # multi-N_max sweep (fig4)
    run_keyvalue_experiment.py # pure key-value dissociation (fig_keyvalue)
    data/
        build_stadler_data.py        # transcribes Roediger 2001 Appendices A+B
        build_mediated_lures.py      # auto-generates mediated lures via ConceptNet
        build_nelson_norms.py        # parses USF free-association norms
        stadler1999_55lists.json     # 55 DRM lists with per-list BAS values
        mediated_lures_55.json       # one mediated lure per list
    tests/                    # pytest suite (109 tests)
    results/figures/          # the 9 figures included in the manuscript
```

## Setup

Python 3.10+. The substrate has no GPU dependency.

```
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

The four external dependencies are `numpy`, `matplotlib`, `nltk`, and (for tests) `pytest`. They install automatically from `pyproject.toml`.

## External data

Two data files are required for the headline experiments but are not redistributed in this repository:

1. **ConceptNet Numberbatch** (English, 19.08 release, ~311 MB). Used as the primary perceptual encoder. Download:
    ```
    curl -L -o drm_paper/data/numberbatch-en-19.08.txt.gz \
      https://conceptnet.s3.amazonaws.com/downloads/2019/numberbatch/numberbatch-en-19.08.txt.gz
    ```

2. **Nelson free-association norms** (USF norms, redistributed at `github.com/teonbrooks/free_association`). Used by the Nelson-encoder ablation:
    ```
    curl -sL https://github.com/teonbrooks/free_association/archive/refs/heads/master.zip \
      -o /tmp/fa.zip
    unzip -q /tmp/fa.zip -d /tmp
    cp /tmp/free_association-master/free/free_association.txt \
       drm_paper/data/nelson_norms_raw.txt
    python drm_paper/data/build_nelson_norms.py
    ```

The DRM lists (`stadler1999_55lists.json`) and mediated lures (`mediated_lures_55.json`) ship with the repository.

## Reproducing the paper figures

Each headline figure has a single entry-point script.

| Figure | Command |
| --- | --- |
| `fig2_recognition_rates.{pdf,png}` (ConceptNet, 20 canonical lists) | see one-liner in §5.1 of the paper, or run `figure_recognition_rates` on a 20-list ConceptNet battery |
| `fig2_recognition_rates_nelson.{pdf,png}` | `python drm_paper/run_nelson_battery.py --n-seeds 20` |
| `fig3_per_list_fa.{pdf,png}` | bundled in `run_headline_battery.py` |
| `fig3c_bas_regression_evidence.{pdf,png}` | `python drm_paper/run_headline_battery.py --n-seeds 20` |
| `fig3c_bas_regression_nelson.{pdf,png}` | `python drm_paper/run_nelson_battery.py --n-seeds 20` |
| `fig3d_human_evidence.{pdf,png}` | `python drm_paper/run_headline_battery.py --n-seeds 20` |
| `fig4_evidence_by_iteration.{pdf,png}` | `python drm_paper/run_evidence_sweep.py --n-seeds 10` |
| `fig4b_ftt_am_dissociation.{pdf,png}` | uses the same multi-N_max sweep |
| `fig6_frontier_lure.{pdf,png}` | `python drm_paper/sweeps.py` |
| `fig_keyvalue_dissociation.{pdf,png}` | `python drm_paper/run_keyvalue_experiment.py --n-seeds 20` |

All runners cache the underlying batteries as `*.pickle` files in `drm_paper/results/`, so re-running with `--load-cached` regenerates figures in seconds.

## Tests

```
pytest drm_paper/
```

All 109 tests run in roughly 5 minutes on a laptop (the slowest individual test builds the Brown-corpus co-occurrence matrix on first run; subsequent runs are cached by NLTK).

## License

MIT. See LICENSE.

## Author

Anonymized for double-blind peer review. Will be deanonymized on acceptance.
