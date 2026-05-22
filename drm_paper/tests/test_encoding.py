"""Tests for drm_paper.encoding."""

from __future__ import annotations

import numpy as np
import pytest

from drm_paper.encoding import (
    DEFAULT_D,
    DEFAULT_K,
    ConceptNetNumberbatchEncoder,
    CooccurrenceEncoder,
    NelsonAssociationEncoder,
    RandomOrthogonalEncoder,
    SyntheticClusterEncoder,
    load_numberbatch,
    make_conceptnet_factory,
)
from drm_paper.stimuli import canonical_lists


class TestRandomOrthogonal:
    def test_encode_shape(self) -> None:
        enc = RandomOrthogonalEncoder()
        hv = enc.encode("doctor")
        assert hv.D == DEFAULT_D
        assert hv.K == DEFAULT_K
        assert len(hv.indices) == DEFAULT_K
        assert hv.indices.dtype == np.int32

    def test_encode_indices_sorted(self) -> None:
        enc = RandomOrthogonalEncoder()
        hv = enc.encode("doctor")
        assert np.all(np.diff(hv.indices) > 0), "indices must be sorted ascending"

    def test_same_word_same_hv(self) -> None:
        """Repeated encoding of the same word returns identical hypervectors."""
        enc = RandomOrthogonalEncoder()
        hv1 = enc.encode("doctor")
        hv2 = enc.encode("doctor")
        assert np.array_equal(hv1.indices, hv2.indices)

    def test_different_words_different_hv(self) -> None:
        enc = RandomOrthogonalEncoder()
        hv1 = enc.encode("doctor")
        hv2 = enc.encode("nurse")
        # Two random K=64 codes in D=2000 have expected overlap ~ K^2/D = 2.
        # They should not be identical.
        assert not np.array_equal(hv1.indices, hv2.indices)

    def test_seed_determinism(self) -> None:
        """Same seed -> same encoding for the same word across encoder instances."""
        enc_a = RandomOrthogonalEncoder(seed=42)
        enc_b = RandomOrthogonalEncoder(seed=42)
        hv_a = enc_a.encode("doctor")
        hv_b = enc_b.encode("doctor")
        assert np.array_equal(hv_a.indices, hv_b.indices)

    def test_different_seeds_different_encoding(self) -> None:
        enc_a = RandomOrthogonalEncoder(seed=42)
        enc_b = RandomOrthogonalEncoder(seed=7)
        hv_a = enc_a.encode("doctor")
        hv_b = enc_b.encode("doctor")
        # Different seeds should produce different encodings (with overwhelming probability)
        assert not np.array_equal(hv_a.indices, hv_b.indices)

    def test_no_semantic_structure(self) -> None:
        """Related words should not have systematically higher overlap than unrelated words.

        This is the architectural defining property of random orthogonal encoding:
        no semantic structure preserved. Tests that the chance-level overlap holds.
        """
        from cgm.vsa.operations import overlap

        enc = RandomOrthogonalEncoder(D=2000, K=64, seed=42)

        # "Related" pairs (semantically associated)
        related_pairs = [
            ("doctor", "nurse"),
            ("hot", "cold"),
            ("table", "chair"),
        ]
        # "Unrelated" pairs
        unrelated_pairs = [
            ("doctor", "elephant"),
            ("table", "ocean"),
            ("hot", "screwdriver"),
        ]

        related_overlaps = [
            overlap(enc.encode(a), enc.encode(b)) for a, b in related_pairs
        ]
        unrelated_overlaps = [
            overlap(enc.encode(a), enc.encode(b)) for a, b in unrelated_pairs
        ]

        # All overlaps should be near the chance floor K^2/D = 64^2/2000 ≈ 2.
        # No systematic difference between "related" and "unrelated".
        chance = (64 * 64) / 2000
        for o in related_overlaps + unrelated_overlaps:
            assert o < 4 * chance, (
                f"random orthogonal overlap {o} is far above chance floor {chance:.1f}; "
                "encoder may be leaking structure"
            )

    def test_invalid_D(self) -> None:
        with pytest.raises(ValueError, match="D must be positive"):
            RandomOrthogonalEncoder(D=0)

    def test_invalid_K(self) -> None:
        with pytest.raises(ValueError, match="K must be in"):
            RandomOrthogonalEncoder(D=100, K=200)
        with pytest.raises(ValueError, match="K must be in"):
            RandomOrthogonalEncoder(D=100, K=0)


_NELSON_PATH = "drm_paper/data/nelson_norms.json"


def _nelson_available() -> bool:
    from pathlib import Path
    return Path(_NELSON_PATH).exists()


@pytest.mark.skipif(not _nelson_available(), reason="Nelson norms JSON not present")
class TestNelsonEncoder:
    def test_encodes_in_vocabulary(self) -> None:
        from drm_paper.encoding import load_nelson_norms

        norms = load_nelson_norms(_NELSON_PATH)
        enc = NelsonAssociationEncoder(norms, seed=0)
        hv = enc.encode("doctor")
        assert hv.D == DEFAULT_D
        assert hv.K == DEFAULT_K
        assert hv.indices.shape == (DEFAULT_K,)

    def test_associates_more_similar_than_unrelated(self) -> None:
        """doctor-nurse (free-association neighbors) > doctor-mountain (unrelated)."""
        from cgm.vsa.operations import overlap

        from drm_paper.encoding import load_nelson_norms

        norms = load_nelson_norms(_NELSON_PATH)
        enc = NelsonAssociationEncoder(norms, seed=0)
        doctor = enc.encode("doctor")
        nurse = enc.encode("nurse")
        mountain = enc.encode("mountain")
        # chance floor ≈ K*K/D = 64*64/2000 ≈ 2.0
        related = overlap(doctor, nurse)
        unrelated = overlap(doctor, mountain)
        assert related > unrelated, (
            f"doctor-nurse overlap ({related}) should exceed doctor-mountain "
            f"overlap ({unrelated})"
        )

    def test_stem_fallback_resolves_plurals(self) -> None:
        """biscuits -> biscuit; almonds -> almond (stem fallback)."""
        from drm_paper.encoding import load_nelson_norms

        norms = load_nelson_norms(_NELSON_PATH)
        enc = NelsonAssociationEncoder(norms, seed=0, fallback="stem")
        # These plurals are not Nelson cues but their singular stems are.
        # Encoder should resolve via Porter stemmer.
        assert enc._resolve_word("biscuits") == "biscuit"
        # No exception:
        hv = enc.encode("biscuits")
        assert hv.K == DEFAULT_K

    def test_random_fallback_handles_oov(self) -> None:
        """OOV words get deterministic random sparse codes under stem/random fallback."""
        from drm_paper.encoding import load_nelson_norms

        norms = load_nelson_norms(_NELSON_PATH)
        enc = NelsonAssociationEncoder(norms, seed=0, fallback="random")
        hv1 = enc.encode("xqzzlbax_not_a_word")
        hv2 = enc.encode("xqzzlbax_not_a_word")
        assert hv1.D == DEFAULT_D
        assert hv1.K == DEFAULT_K
        # Same word encoded twice yields identical code
        assert (hv1.indices == hv2.indices).all()

    def test_strict_fallback_raises_on_oov(self) -> None:
        """fallback=strict raises KeyError for missing words."""
        from drm_paper.encoding import load_nelson_norms

        norms = load_nelson_norms(_NELSON_PATH)
        enc = NelsonAssociationEncoder(norms, seed=0, fallback="strict")
        with pytest.raises(KeyError, match="not in Nelson"):
            enc.encode("xqzzlbax_not_a_word")

    def test_factory_shares_norms_across_seeds(self) -> None:
        """make_nelson_factory shares the parsed dict; each seed gets a fresh projection."""
        from cgm.vsa.operations import overlap

        from drm_paper.encoding import make_nelson_factory

        factory = make_nelson_factory(_NELSON_PATH)
        enc_a = factory(0)
        enc_b = factory(1)
        # Both encode same word but with different projections; codes differ.
        a = enc_a.encode("doctor")
        b = enc_b.encode("doctor")
        # Should not be identical (different random projections)
        assert overlap(a, b) < DEFAULT_K, (
            "two encoders with different seeds should produce different codes"
        )

    def test_invalid_fallback(self) -> None:
        from drm_paper.encoding import load_nelson_norms

        norms = load_nelson_norms(_NELSON_PATH)
        with pytest.raises(ValueError, match="fallback must be"):
            NelsonAssociationEncoder(norms, fallback="invalid_mode")


_NUMBERBATCH_PATH = "drm_paper/data/numberbatch-en-19.08.txt.gz"


def _numberbatch_available() -> bool:
    from pathlib import Path
    return Path(_NUMBERBATCH_PATH).exists()


@pytest.mark.skipif(not _numberbatch_available(), reason="Numberbatch data not present")
class TestConceptNetNumberbatch:
    def test_encodes_vocabulary(self) -> None:
        from drm_paper.stimuli import DEFAULT_FOILS, all_words, canonical_lists
        lists = canonical_lists()
        vocab = all_words(lists) | set(DEFAULT_FOILS)
        # Also include mediated lures
        vocab |= {dl.mediated_lure for dl in lists if dl.mediated_lure is not None}

        enc = ConceptNetNumberbatchEncoder(_NUMBERBATCH_PATH, vocab, seed=42)
        # Encode a few words; verify shape
        hv = enc.encode("doctor")
        assert hv.D == DEFAULT_D
        assert hv.K == DEFAULT_K
        assert len(hv.indices) == DEFAULT_K

    def test_caching_consistent(self) -> None:
        from drm_paper.stimuli import canonical_lists, all_words
        vocab = all_words(canonical_lists())
        enc = ConceptNetNumberbatchEncoder(_NUMBERBATCH_PATH, vocab, seed=42)
        hv1 = enc.encode("doctor")
        hv2 = enc.encode("doctor")
        assert np.array_equal(hv1.indices, hv2.indices)

    def test_related_words_overlap_above_chance(self) -> None:
        """Related words should overlap above the chance floor."""
        from cgm.vsa.operations import overlap
        from drm_paper.stimuli import canonical_lists, all_words

        vocab = all_words(canonical_lists())
        enc = ConceptNetNumberbatchEncoder(_NUMBERBATCH_PATH, vocab, seed=42)

        # Related: doctor and nurse (both in the doctor list)
        related = overlap(enc.encode("doctor"), enc.encode("nurse"))
        # Unrelated: doctor and table (from chair list, unrelated to medical)
        unrelated = overlap(enc.encode("doctor"), enc.encode("table"))

        chance = DEFAULT_K * DEFAULT_K / DEFAULT_D  # ~2 for K=64, D=2000

        # Related pairs should have overlap above chance.
        assert related > 2 * chance, f"related overlap {related} too low (chance {chance:.1f})"
        # Unrelated may or may not be above chance — random projection adds noise.
        # The key check is related > unrelated on average; tested in another test.

    def test_seed_determinism(self) -> None:
        from drm_paper.stimuli import canonical_lists, all_words

        vocab = all_words(canonical_lists())
        enc_a = ConceptNetNumberbatchEncoder(_NUMBERBATCH_PATH, vocab, seed=42)
        enc_b = ConceptNetNumberbatchEncoder(_NUMBERBATCH_PATH, vocab, seed=42)
        assert np.array_equal(
            enc_a.encode("doctor").indices, enc_b.encode("doctor").indices
        )

    def test_unknown_word_raises(self) -> None:
        from drm_paper.stimuli import canonical_lists, all_words

        vocab = all_words(canonical_lists())
        enc = ConceptNetNumberbatchEncoder(_NUMBERBATCH_PATH, vocab, seed=42)
        with pytest.raises(KeyError, match="not in"):
            enc.encode("xyzzy_definitely_not_a_word")

    def test_loaded_dict_constructor_shares_embeddings(self) -> None:
        """Using a pre-loaded embeddings dict should give identical encodings
        to constructing from a path, just without the file-read overhead."""
        from drm_paper.stimuli import all_words, canonical_lists

        vocab = all_words(canonical_lists())
        embeddings = load_numberbatch(_NUMBERBATCH_PATH, vocab)

        enc_a = ConceptNetNumberbatchEncoder(embeddings, vocabulary=vocab, seed=42)
        enc_b = ConceptNetNumberbatchEncoder(_NUMBERBATCH_PATH, vocab, seed=42)

        # Same seed → identical projection → identical encodings
        for word in ("doctor", "nurse", "table"):
            assert np.array_equal(enc_a.encode(word).indices, enc_b.encode(word).indices)

    def test_factory_returns_distinct_encoders_per_seed(self) -> None:
        from drm_paper.stimuli import all_words, canonical_lists

        vocab = all_words(canonical_lists())
        factory = make_conceptnet_factory(_NUMBERBATCH_PATH, vocab)

        enc_a = factory(0)
        enc_b = factory(1)
        # Different seeds → different random projections → different encodings
        assert not np.array_equal(enc_a.encode("doctor").indices, enc_b.encode("doctor").indices)

    def test_factory_speeds_up_multiseed_construction(self) -> None:
        """make_conceptnet_factory loads embeddings once and reuses across
        encoder constructions. The N-th construction after the first should
        be substantially faster than the first."""
        import time

        from drm_paper.stimuli import all_words, canonical_lists

        vocab = all_words(canonical_lists())

        # First call includes the file load.
        t0 = time.perf_counter()
        factory = make_conceptnet_factory(_NUMBERBATCH_PATH, vocab)
        first_encoder = factory(0)
        first_total = time.perf_counter() - t0

        # Subsequent calls only build a new projection matrix (cheap).
        t1 = time.perf_counter()
        for s in range(5):
            factory(s)
        subsequent_total = time.perf_counter() - t1

        # Reuse should be ≥10× cheaper per construction than the first load.
        per_construction_first = first_total
        per_construction_reuse = subsequent_total / 5
        assert per_construction_reuse * 10 < per_construction_first, (
            f"factory reuse not fast enough: first={per_construction_first:.2f}s, "
            f"reuse_mean={per_construction_reuse:.2f}s"
        )

    def test_drm_list_related_words_more_similar_than_distractors(self) -> None:
        """Average overlap WITHIN a DRM list should exceed average overlap
        between random unrelated word pairs."""
        from cgm.vsa.operations import overlap
        from drm_paper.stimuli import DEFAULT_FOILS, all_words, canonical_lists

        lists = canonical_lists()
        vocab = all_words(lists) | set(DEFAULT_FOILS)
        enc = ConceptNetNumberbatchEncoder(_NUMBERBATCH_PATH, vocab, seed=42)

        # Within-list: lure vs each associate, averaged across all canonical lists
        within = []
        for dl in lists:
            lure_hv = enc.encode(dl.critical_lure)
            for a in dl.associates:
                within.append(overlap(lure_hv, enc.encode(a)))

        # Foil-foil: random foil pairs
        between = []
        foils = list(DEFAULT_FOILS)
        for i, w1 in enumerate(foils):
            for w2 in foils[i + 1 :]:
                between.append(overlap(enc.encode(w1), enc.encode(w2)))

        within_mean = float(np.mean(within))
        between_mean = float(np.mean(between))

        # The architectural property that produces DRM: related words have
        # substantially higher overlap than unrelated words under this encoder.
        assert within_mean > between_mean + 1.5, (
            f"within-list overlap {within_mean:.1f} should substantially exceed "
            f"between-foil overlap {between_mean:.1f}"
        )


class TestCooccurrence:
    """The CooccurrenceEncoder is the less-trained §6.2 comparator. Builds
    PPMI co-occurrence vectors from an NLTK corpus, then projects to
    sparse-binary. We expect it to preserve some semantic structure
    (related words should overlap above chance) but less than ConceptNet."""

    def test_encodes_vocabulary(self) -> None:
        from drm_paper.stimuli import all_words, canonical_lists
        vocab = all_words(canonical_lists())
        enc = CooccurrenceEncoder(vocab, seed=42)
        hv = enc.encode("doctor")
        assert hv.D == DEFAULT_D
        assert hv.K == DEFAULT_K

    def test_unknown_word_raises(self) -> None:
        from drm_paper.stimuli import all_words, canonical_lists
        vocab = all_words(canonical_lists())
        enc = CooccurrenceEncoder(vocab, seed=42)
        with pytest.raises(KeyError, match="not in"):
            enc.encode("xyzzy_definitely_not_a_word")

    def test_caching_consistent(self) -> None:
        from drm_paper.stimuli import all_words, canonical_lists
        vocab = all_words(canonical_lists())
        enc = CooccurrenceEncoder(vocab, seed=42)
        hv1 = enc.encode("doctor")
        hv2 = enc.encode("doctor")
        assert np.array_equal(hv1.indices, hv2.indices)

    def test_seed_determinism(self) -> None:
        from drm_paper.stimuli import all_words, canonical_lists
        vocab = all_words(canonical_lists())
        enc_a = CooccurrenceEncoder(vocab, seed=42)
        enc_b = CooccurrenceEncoder(vocab, seed=42)
        # Same seed should give identical encoding for any word
        assert np.array_equal(
            enc_a.encode("doctor").indices,
            enc_b.encode("doctor").indices,
        )


class TestSyntheticClusterEncoder:
    def test_encodes_list_words(self) -> None:
        lists = canonical_lists()
        enc = SyntheticClusterEncoder(lists, seed=42)
        for dl in lists:
            assert isinstance(enc.encode(dl.critical_lure).indices, np.ndarray)
            for a in dl.associates:
                assert isinstance(enc.encode(a).indices, np.ndarray)

    def test_unknown_word_raises(self) -> None:
        lists = canonical_lists()
        enc = SyntheticClusterEncoder(lists, seed=42)
        with pytest.raises(KeyError, match="not in"):
            enc.encode("xyzzy_definitely_not_a_word")

    def test_extra_words_orthogonal_to_lists(self) -> None:
        lists = canonical_lists()
        enc = SyntheticClusterEncoder(
            lists, extra_words=("xyzzy_distractor",), seed=42
        )
        # Should be encodable without error
        hv = enc.encode("xyzzy_distractor")
        assert hv.K == DEFAULT_K

    def test_lure_associates_overlap(self) -> None:
        """Words in the same DRM list (lure + associates) should have
        substantial mutual overlap; words across lists should have chance-floor
        overlap. This is what makes the substrate's DRM prediction work.
        """
        from cgm.vsa.operations import overlap

        lists = canonical_lists()
        enc = SyntheticClusterEncoder(lists, drop_frac=0.35, seed=42)

        # Within-cluster: lure vs each associate
        first = lists[0]
        lure_hv = enc.encode(first.critical_lure)
        within_overlaps = [overlap(lure_hv, enc.encode(a)) for a in first.associates]
        # With drop_frac=0.35, associate has ~(1-0.35) * K = ~42/64 active bits in common with lure
        assert np.mean(within_overlaps) >= DEFAULT_K * 0.5, (
            f"within-cluster overlap mean {np.mean(within_overlaps):.1f} "
            f"too low (expected at least {DEFAULT_K * 0.5:.0f})"
        )

        # Cross-cluster: lure of list 0 vs lure of list 1
        cross = overlap(lure_hv, enc.encode(lists[1].critical_lure))
        chance_floor = DEFAULT_K * DEFAULT_K / DEFAULT_D  # ≈ 2 for K=64, D=2000
        assert cross < 4 * chance_floor, (
            f"cross-cluster overlap {cross} much higher than chance floor {chance_floor:.1f}"
        )

    def test_same_seed_deterministic(self) -> None:
        lists = canonical_lists()
        enc_a = SyntheticClusterEncoder(lists, seed=42)
        enc_b = SyntheticClusterEncoder(lists, seed=42)
        for dl in lists:
            assert np.array_equal(
                enc_a.encode(dl.critical_lure).indices,
                enc_b.encode(dl.critical_lure).indices,
            )

    def test_invalid_drop_frac(self) -> None:
        lists = canonical_lists()
        with pytest.raises(ValueError, match="drop_frac"):
            SyntheticClusterEncoder(lists, drop_frac=0.0)
        with pytest.raises(ValueError, match="drop_frac"):
            SyntheticClusterEncoder(lists, drop_frac=1.0)

    def test_invalid_mediated_drop_frac(self) -> None:
        lists = canonical_lists()
        with pytest.raises(ValueError, match="mediated_drop_frac"):
            SyntheticClusterEncoder(lists, mediated_drop_frac=0.0)
        with pytest.raises(ValueError, match="mediated_drop_frac"):
            SyntheticClusterEncoder(lists, mediated_drop_frac=1.0)
        # Must exceed drop_frac
        with pytest.raises(ValueError, match="must exceed"):
            SyntheticClusterEncoder(
                lists, drop_frac=0.5, mediated_drop_frac=0.5
            )

    def test_mediated_lure_encodable(self) -> None:
        """Mediated lures should be encodable and have moderate overlap
        with the cluster centroid — less than associates have, but more
        than chance."""
        from cgm.vsa.operations import overlap

        lists = canonical_lists()
        enc = SyntheticClusterEncoder(
            lists, drop_frac=0.35, mediated_drop_frac=0.55, seed=42
        )

        first = lists[0]
        assert first.mediated_lure is not None  # guaranteed by fixture
        lure_hv = enc.encode(first.critical_lure)
        mediated_hv = enc.encode(first.mediated_lure)
        associate_hv = enc.encode(first.associates[0])

        # Mediated lure shares fewer bits with the cluster base than associates do
        assoc_overlap = overlap(lure_hv, associate_hv)
        med_overlap = overlap(lure_hv, mediated_hv)
        chance_floor = DEFAULT_K * DEFAULT_K / DEFAULT_D

        assert med_overlap < assoc_overlap, (
            f"mediated overlap {med_overlap} should be less than "
            f"associate overlap {assoc_overlap}"
        )
        assert med_overlap > 4 * chance_floor, (
            f"mediated overlap {med_overlap} should be well above chance "
            f"floor {chance_floor:.1f}"
        )
