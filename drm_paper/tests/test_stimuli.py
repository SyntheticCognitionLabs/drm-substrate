"""Tests for drm_paper.stimuli."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from drm_paper.stimuli import DRMList, all_words, canonical_lists, load_stadler_full


class TestDRMList:
    def test_valid(self) -> None:
        dl = DRMList(
            critical_lure="sleep",
            associates=tuple(["bed", "rest", "awake", "tired", "dream",
                              "wake", "snooze", "blanket", "doze", "slumber",
                              "snore", "nap", "peace", "yawn", "drowsy"]),
        )
        assert dl.critical_lure == "sleep"
        assert len(dl.associates) == 15

    def test_wrong_associate_count(self) -> None:
        with pytest.raises(ValueError, match="must have 15 associates"):
            DRMList(critical_lure="x", associates=("a", "b", "c"))

    def test_lure_in_associates(self) -> None:
        with pytest.raises(ValueError, match="must not appear in its own associates"):
            DRMList(
                critical_lure="sleep",
                associates=tuple(["sleep"] + [f"w{i}" for i in range(14)]),
            )

    def test_uppercase_lure_rejected(self) -> None:
        with pytest.raises(ValueError, match="must be lowercase"):
            DRMList(
                critical_lure="Sleep",
                associates=tuple([f"w{i}" for i in range(15)]),
            )

    def test_uppercase_associate_rejected(self) -> None:
        with pytest.raises(ValueError, match="must be lowercase"):
            DRMList(
                critical_lure="sleep",
                associates=tuple(["Bed"] + [f"w{i}" for i in range(14)]),
            )


class TestCanonicalLists:
    def test_nonempty(self) -> None:
        lists = canonical_lists()
        assert len(lists) >= 10, "canonical subset should have at least 10 lists for smoke tests"

    def test_each_has_15_associates(self) -> None:
        for dl in canonical_lists():
            assert len(dl.associates) == 15, (
                f"list for '{dl.critical_lure}' has {len(dl.associates)} associates"
            )

    def test_unique_lures(self) -> None:
        lures = [dl.critical_lure for dl in canonical_lists()]
        assert len(lures) == len(set(lures)), f"duplicate critical lures: {lures}"

    def test_all_lowercase(self) -> None:
        for dl in canonical_lists():
            assert dl.critical_lure == dl.critical_lure.lower()
            for w in dl.associates:
                assert w == w.lower(), f"non-lowercase associate '{w}' in '{dl.critical_lure}'"

    def test_lure_not_in_own_associates(self) -> None:
        for dl in canonical_lists():
            assert dl.critical_lure not in dl.associates

    def test_mediated_lures_present_and_distinct(self) -> None:
        """Every canonical list should have a mediated_lure, distinct from
        its critical_lure and not in its associates."""
        for dl in canonical_lists():
            assert dl.mediated_lure is not None, (
                f"canonical list for '{dl.critical_lure}' missing mediated_lure"
            )
            assert dl.mediated_lure != dl.critical_lure
            assert dl.mediated_lure not in dl.associates

    def test_mediated_lures_globally_unique(self) -> None:
        """No two lists share a mediated lure (avoids cross-list contamination)."""
        mediated = [
            dl.mediated_lure for dl in canonical_lists()
            if dl.mediated_lure is not None
        ]
        assert len(mediated) == len(set(mediated)), (
            f"duplicate mediated lures: {mediated}"
        )

    def test_mediated_lures_not_in_other_lists(self) -> None:
        """A mediated lure for list A should not appear in list B's
        associates or critical_lure — otherwise it would have a confounding
        episodic trace from studying list B."""
        lists = canonical_lists()
        for i, dl in enumerate(lists):
            if dl.mediated_lure is None:
                continue
            for j, other in enumerate(lists):
                if i == j:
                    continue
                assert dl.mediated_lure not in other.associates, (
                    f"mediated lure '{dl.mediated_lure}' (for '{dl.critical_lure}') "
                    f"appears in associates of '{other.critical_lure}'"
                )
                assert dl.mediated_lure != other.critical_lure


class TestAllWords:
    def test_aggregates_lures_and_associates(self) -> None:
        lists = canonical_lists()
        words = all_words(lists)

        # Should include every critical lure
        for dl in lists:
            assert dl.critical_lure in words
            for a in dl.associates:
                assert a in words

    def test_returns_set(self) -> None:
        words = all_words(canonical_lists())
        assert isinstance(words, set)


class TestLoadFromFile:
    def test_roundtrip(self, tmp_path: Path) -> None:
        data = [
            {
                "critical_lure": "test",
                "associates": [f"w{i}" for i in range(15)],
                "published_fa_rate": 0.7,
                "published_bas": 0.3,
                "notes": "test list",
            },
        ]
        path = tmp_path / "test.json"
        path.write_text(json.dumps(data))

        loaded = load_stadler_full(path)
        assert len(loaded) == 1
        assert loaded[0].critical_lure == "test"
        assert loaded[0].published_fa_rate == 0.7
        assert loaded[0].published_bas == 0.3
        assert loaded[0].notes == "test list"

    def test_optional_fields_default_none(self, tmp_path: Path) -> None:
        data = [
            {
                "critical_lure": "test",
                "associates": [f"w{i}" for i in range(15)],
            },
        ]
        path = tmp_path / "test.json"
        path.write_text(json.dumps(data))

        loaded = load_stadler_full(path)
        assert loaded[0].published_fa_rate is None
        assert loaded[0].published_bas is None
        assert loaded[0].notes == ""
