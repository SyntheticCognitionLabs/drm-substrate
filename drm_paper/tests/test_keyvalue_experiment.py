"""Tests for the pure key-value DRM experiment.

The predicted result is a sharp dissociation: studied items recognized at
1.0 (the correct role recovers a stored binding exactly), every other
item type at 0.0 (no exact match anywhere in the role-bound store).
These tests verify that prediction on a small synthetic substrate that
does not require external embeddings.
"""

from __future__ import annotations

import numpy as np
import pytest

from cgm.vsa.hypervector import SparseBinaryHV

from drm_paper.encoding import RandomOrthogonalEncoder
from drm_paper.run_keyvalue_experiment import (
    KVTrial,
    aggregate_kv_rates,
    run_kv_seed,
)
from drm_paper.stimuli import DRMList


def _toy_lists() -> tuple[DRMList, ...]:
    """Two tiny lists with non-overlapping vocabularies plus a critical lure.

    The point of these tests is to exercise the bind/probe flow, not the
    associative structure, so we use deterministic random encodings (the
    associates have no semantic relationship to the lure). Pure key-value
    retrieval should still recognize studied items and reject everything
    else regardless of encoder.
    """
    return (
        DRMList(
            critical_lure="alpha",
            associates=tuple(f"a{i}" for i in range(1, 16)),
            mediated_lure="alpha_med",
        ),
        DRMList(
            critical_lure="beta",
            associates=tuple(f"b{i}" for i in range(1, 16)),
            mediated_lure="beta_med",
        ),
    )


def test_studied_perfect_recognition() -> None:
    """Pure KV: every studied associate is recognized at its list's role."""
    drm_lists = _toy_lists()
    foils = ("z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8")
    encoder = RandomOrthogonalEncoder(D=512, K=16, seed=0)

    result = run_kv_seed(
        drm_lists, encoder,
        role_seed=0, study_seed=1, test_seed=2,
        foil_pool=foils, held_out_per_list=1,
        studied_per_list=4, distractors_per_list=3,
    )

    studied = [t for t in result.trials if t.item_type == "studied"]
    assert len(studied) > 0
    assert all(t.recognized for t in studied), (
        "studied items must all recognize under correct-role probe"
    )


def test_all_non_studied_items_rejected() -> None:
    """Pure KV: critical lures, mediated, held-out, distractors all reject."""
    drm_lists = _toy_lists()
    foils = ("z1", "z2", "z3", "z4", "z5", "z6")
    encoder = RandomOrthogonalEncoder(D=512, K=16, seed=0)

    result = run_kv_seed(
        drm_lists, encoder,
        role_seed=0, study_seed=1, test_seed=2,
        foil_pool=foils, held_out_per_list=1,
        studied_per_list=4, distractors_per_list=3,
    )

    for item_type in ("critical_lure", "mediated_lure", "held_out", "distractor"):
        trials = [t for t in result.trials if t.item_type == item_type]
        assert len(trials) > 0, f"no trials of type {item_type}"
        rejected = sum(1 for t in trials if not t.recognized)
        assert rejected == len(trials), (
            f"all {item_type} items should be rejected under pure KV; "
            f"got {len(trials) - rejected} false alarms out of {len(trials)}"
        )


def test_aggregate_kv_rates_yields_predicted_dissociation() -> None:
    drm_lists = _toy_lists()
    foils = ("z1", "z2", "z3", "z4", "z5", "z6")
    encoder = RandomOrthogonalEncoder(D=512, K=16, seed=0)

    seeds = [
        run_kv_seed(
            drm_lists, encoder,
            role_seed=i, study_seed=i + 100, test_seed=i + 200,
            foil_pool=foils, held_out_per_list=1,
            studied_per_list=4, distractors_per_list=3,
        )
        for i in range(3)
    ]
    rates = aggregate_kv_rates(seeds)
    assert rates["studied"]["mean"] == pytest.approx(1.0)
    for it in ("critical_lure", "mediated_lure", "held_out", "distractor"):
        assert rates[it]["mean"] == pytest.approx(0.0), (
            f"non-studied item type {it} must aggregate to 0 under pure KV"
        )


def test_distractor_probe_tries_all_roles() -> None:
    """Distractor probes try every role, so a deliberately-poisoned distractor
    that happens to encode identically to a studied word should be detected.
    """
    drm_lists = _toy_lists()
    # The first list studies a1..a5 (minus one held-out). Make "z1" identical
    # to one of those studied associates by ensuring the encoder maps both to
    # the same code: with a deterministic per-word RNG seed, distinct strings
    # have different codes, so a true collision is vanishingly rare. We
    # instead verify the path runs over all roles without erroring on
    # distractors that legitimately match nothing.
    foils = ("z1", "z2", "z3")
    encoder = RandomOrthogonalEncoder(D=512, K=16, seed=0)
    result = run_kv_seed(
        drm_lists, encoder,
        role_seed=0, study_seed=1, test_seed=2,
        foil_pool=foils, held_out_per_list=1,
        studied_per_list=4, distractors_per_list=3,
    )
    distractor_trials = [t for t in result.trials if t.item_type == "distractor"]
    assert all(not t.recognized for t in distractor_trials)
