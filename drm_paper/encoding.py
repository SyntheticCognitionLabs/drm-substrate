"""Word encoders for the DRM paper.

The substrate operates on sparse-binary hypervectors. Each encoding scheme
maps a word string to a stable ``SparseBinaryHV`` with shared dimension D
and active-bit count K. Per §6.2 of DRM_PAPER_OUTLINE.md, four schemes are
used to test the contribution of semantic structure in the input
representation:

1. ``RandomOrthogonalEncoder`` — per-word random sparse codes (no semantic
   structure). Used as the ablation condition; predicts DRM ≈ 0.
2. ``ConceptNetNumberbatchEncoder`` — sparse projection of pretrained
   ConceptNet Numberbatch embeddings. Primary encoder.
3. ``NelsonAssociationEncoder`` — sparse-binary projection of the Nelson,
   McEvoy & Schreiber (1998) free-association graph (USF norms). The
   encoding most directly tied to the canonical DRM-predictor BAS, since
   both are derived from the same association data.
4. ``CooccurrenceEncoder`` — sparse codes from corpus co-occurrence
   statistics (NLTK Brown). Less-trained comparator.

All encoders are deterministic given their constructor seed. Each word maps
to the same hypervector across calls (internal caching enforces this).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Protocol

import numpy as np

from cgm.vsa.hypervector import SparseBinaryHV

DEFAULT_D = 2000
DEFAULT_K = 64


class WordEncoder(Protocol):
    """Encoder interface: word string -> sparse-binary hypervector."""

    @property
    def D(self) -> int: ...

    @property
    def K(self) -> int: ...

    def encode(self, word: str) -> SparseBinaryHV: ...


def _stable_word_seed(seed: int, word: str) -> int:
    """Map (seed, word) to a stable 64-bit integer for RNG seeding.

    Uses SHA-256 so the mapping is independent of Python's PYTHONHASHSEED
    and stable across processes / interpreters. Without this, the same word
    would map to different hypervectors across separate runs even with the
    same constructor seed.
    """
    h = hashlib.sha256(f"{seed}:{word}".encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big")


class RandomOrthogonalEncoder:
    """Per-word random sparse codes with no semantic structure.

    Each word maps to K randomly-chosen indices in [0, D), with the choice
    deterministic given the encoder's seed and the word string. Two unrelated
    words have expected overlap ``K * K / D`` (the chance floor); two related
    words have the same expected overlap (no semantic structure preserved).

    Used as the §6.2 ablation condition. Predicts DRM ≈ 0 because no two
    associates share enough semantic features for a meaningful consolidated
    gist centroid to form.
    """

    def __init__(self, D: int = DEFAULT_D, K: int = DEFAULT_K, seed: int = 0) -> None:
        if D <= 0:
            raise ValueError(f"D must be positive, got {D}")
        if K < 1 or K > D:
            raise ValueError(f"K must be in [1, {D}], got {K}")

        self._D = D
        self._K = K
        self._seed = seed
        self._cache: dict[str, SparseBinaryHV] = {}

    @property
    def D(self) -> int:
        return self._D

    @property
    def K(self) -> int:
        return self._K

    def encode(self, word: str) -> SparseBinaryHV:
        if word not in self._cache:
            rng = np.random.default_rng(_stable_word_seed(self._seed, word))
            indices = np.sort(rng.choice(self._D, self._K, replace=False)).astype(np.int32)
            self._cache[word] = SparseBinaryHV(D=self._D, indices=indices)
        return self._cache[word]

    def encode_batch(self, words: list[str]) -> list[SparseBinaryHV]:
        return [self.encode(w) for w in words]


def load_numberbatch(
    path: Path | str,
    vocabulary: set[str] | tuple[str, ...] | list[str],
) -> dict[str, np.ndarray]:
    """Stream the gzipped ConceptNet Numberbatch file once and extract dense
    embeddings for every word in ``vocabulary``.

    Format (gzipped UTF-8 text):
        <line 1>: "<vocab_size> <dim>"
        <line N>: "<word> <v1> <v2> ... <v_dim>"

    Returns dict[word, dense float32 array]. Words not in the file are
    silently absent; callers handle missing-word reporting.

    Hoisting this out of the encoder lets callers load once and reuse the
    dict across encoder instances — eliminates ~6.5s file read per seed
    in multi-seed batteries.
    """
    import gzip

    embeddings: dict[str, np.ndarray] = {}
    vocab_set: set[str] = set(vocabulary)
    if not vocab_set:
        return embeddings

    with gzip.open(path, "rt", encoding="utf-8") as f:
        _header = f.readline()
        for line in f:
            space = line.find(" ")
            if space < 0:
                continue
            word = line[:space]
            if word in vocab_set:
                vec = np.fromstring(line[space + 1:], sep=" ", dtype=np.float32)
                embeddings[word] = vec
                if len(embeddings) == len(vocab_set):
                    break
    return embeddings


class ConceptNetNumberbatchEncoder:
    """Sparse-binary projection of ConceptNet Numberbatch embeddings.

    Pipeline:
      1. Dense embedding for the word (from a pre-loaded dict or loaded
         on demand from a file path).
      2. Fixed Gaussian random projection matrix 300 → D (seeded).
      3. Project, take top-K indices by absolute value, build SparseBinaryHV.
      4. Cache results per encoder instance.

    The fixed random projection preserves semantic similarity (Johnson-
    Lindenstrauss style) while delivering sparse-binary output compatible
    with the substrate. ConceptNet Numberbatch's dense space encodes
    semantic relatedness (synonymy + association + co-occurrence) into
    cosine similarity; the projection inherits that property.

    Construction supports two modes:
      - ``ConceptNetNumberbatchEncoder(embeddings_dict, seed=...)`` — uses
        pre-loaded dict. Faster for multi-seed batteries (share one load).
      - ``ConceptNetNumberbatchEncoder(path, vocabulary=..., seed=...)`` —
        loads from file at construction. Use only for single-experiment runs;
        prefer ``make_conceptnet_factory`` for multi-seed work.

    ``missing_words`` reports vocabulary entries not found in the embedding
    dict. Empty when loading without a vocabulary check.
    """

    def __init__(
        self,
        source: dict[str, np.ndarray] | Path | str,
        vocabulary: set[str] | tuple[str, ...] | list[str] | None = None,
        *,
        D: int = DEFAULT_D,
        K: int = DEFAULT_K,
        seed: int = 0,
    ) -> None:
        if D <= 0:
            raise ValueError(f"D must be positive, got {D}")
        if K < 1 or K > D:
            raise ValueError(f"K must be in [1, {D}], got {K}")

        self._D = D
        self._K = K
        self._seed = seed

        if isinstance(source, dict):
            self._embeddings: dict[str, np.ndarray] = source
            if vocabulary is not None:
                self.missing_words: tuple[str, ...] = tuple(
                    sorted(set(vocabulary) - set(source.keys()))
                )
            else:
                self.missing_words = ()
        else:
            if vocabulary is None:
                raise ValueError(
                    "vocabulary required when loading ConceptNet from a path; "
                    "supply a set of words to extract from the file."
                )
            vocab_set = set(vocabulary)
            self._embeddings = load_numberbatch(source, vocab_set)
            self.missing_words = tuple(
                sorted(vocab_set - set(self._embeddings.keys()))
            )

        rng = np.random.default_rng(seed)
        dim_in = (
            next(iter(self._embeddings.values())).shape[0]
            if self._embeddings else 300
        )
        self._projection: np.ndarray = rng.standard_normal((dim_in, D)).astype(np.float32)
        self._cache: dict[str, SparseBinaryHV] = {}

    @property
    def D(self) -> int:
        return self._D

    @property
    def K(self) -> int:
        return self._K

    def encode(self, word: str) -> SparseBinaryHV:
        if word in self._cache:
            return self._cache[word]
        if word not in self._embeddings:
            raise KeyError(
                f"word '{word}' not in ConceptNetNumberbatchEncoder vocabulary or "
                "missing from Numberbatch file. Add it to vocabulary at "
                "construction time, or check spelling against Numberbatch entries."
            )

        dense = self._embeddings[word]
        projected = dense @ self._projection  # shape (D,)
        # Top-K by absolute value — equivalent to thresholding at the K-th
        # largest |value|. Indices are sorted ascending for SparseBinaryHV.
        top_k = np.argpartition(np.abs(projected), -self._K)[-self._K :]
        indices = np.sort(top_k).astype(np.int32)
        hv = SparseBinaryHV(D=self._D, indices=indices)
        self._cache[word] = hv
        return hv


def load_nelson_norms(path: Path | str) -> dict:
    """Load the Nelson norms JSON produced by ``build_nelson_norms.py``.

    Returns the full payload dict (``metadata``, ``by_cue``, ``by_target``).
    Hoisted out of the encoder so callers can pre-load once and share the
    structure across encoder instances (analogous to ``load_numberbatch``).
    """
    import json

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Nelson norms JSON not found at {p}. Build it with: "
            "`python drm_paper/data/build_nelson_norms.py`"
        )
    return json.loads(p.read_text())


class NelsonAssociationEncoder:
    """Sparse-binary codes from the Nelson, McEvoy & Schreiber (1998)
    USF Free Association Norms.

    Pipeline:
      1. Each Nelson word is indexed in ``[0, V)`` (``V \\approx 10\\,617``;
         the union of every word that appears as a cue or as a target).
      2. For the word being encoded, a sparse feature vector is built over
         ``V`` dimensions: ``f[i]`` is the maximum association strength
         (FSG, forward strength) between the word and word ``i``, taking
         either direction (forward: this word as cue -> word ``i`` as
         target; backward: word ``i`` as cue -> this word as target).
      3. A fixed Gaussian random matrix ``V -> D`` (seeded) projects the
         feature vector down to ``D`` dimensions (Johnson-Lindenstrauss).
      4. Top-``K`` indices by absolute value form the sparse-binary code.

    Words that share many associates in the Nelson graph end up with
    overlapping sparse-binary codes. This is the canonical "associative
    structure" encoder: overlap in the substrate's similarity metric tracks
    overlap in human free-association response patterns.

    The data file is generated by ``drm_paper/data/build_nelson_norms.py``
    from the raw USF flat-file at
    ``drm_paper/data/nelson_norms_raw.txt`` (redistributed at
    https://github.com/teonbrooks/free_association). The encoder is
    deterministic in ``seed`` and caches per-word results.

    Construction supports two modes (mirroring ConceptNetNumberbatchEncoder):
      - ``NelsonAssociationEncoder(norms_dict, seed=...)`` -- uses a
        pre-loaded payload (from ``load_nelson_norms``). Use this for
        multi-seed batteries via ``make_nelson_factory``.
      - ``NelsonAssociationEncoder(path, seed=...)`` -- loads from JSON
        at construction. Slower for multi-seed work because the load is
        repeated each time.

    ``fallback`` controls how out-of-vocabulary words are handled. Nelson
    contains ~10\\,617 base lemma forms; DRM associate lists often include
    morphological variants ("biscuits", "sluggish") and the technical
    distractor pool is largely outside Nelson. With ``fallback="stem"``,
    on a vocabulary miss the encoder retries with a Porter-stem lookup
    against the Nelson vocabulary, picking the most-connected matching
    base form. With ``fallback="random"``, missing words receive a
    deterministic random sparse code (per-word, seeded by the encoder
    seed and the word string), giving valid hypervectors with no
    semantic structure. With ``fallback="strict"``, missing words raise
    ``KeyError``. ``stem`` falls through to ``random`` if no stem match
    exists.

    ``missing_after_fallback`` reports vocabulary entries that fell through
    to the random fallback (an honest accounting of where the encoder has
    no semantic information). ``missing_words`` reports vocabulary entries
    missing from Nelson before any fallback (informational).
    """

    def __init__(
        self,
        source: dict | Path | str,
        vocabulary: set[str] | tuple[str, ...] | list[str] | None = None,
        *,
        D: int = DEFAULT_D,
        K: int = DEFAULT_K,
        seed: int = 0,
        fallback: str = "stem",
    ) -> None:
        if D <= 0:
            raise ValueError(f"D must be positive, got {D}")
        if K < 1 or K > D:
            raise ValueError(f"K must be in [1, {D}], got {K}")
        if fallback not in ("strict", "stem", "random"):
            raise ValueError(
                f"fallback must be one of strict/stem/random, got {fallback!r}"
            )

        self._D = D
        self._K = K
        self._seed = seed
        self._fallback = fallback

        if isinstance(source, dict):
            self._data = source
        else:
            self._data = load_nelson_norms(source)

        by_cue = self._data["by_cue"]
        by_target = self._data["by_target"]
        vocab_set = set(by_cue.keys()) | set(by_target.keys())
        vocab_list = sorted(vocab_set)
        self._word_to_idx: dict[str, int] = {w: i for i, w in enumerate(vocab_list)}
        V = len(vocab_list)
        self._V = V
        self._vocab_set = vocab_set

        # Build stem -> best base-word index. "Best" = the Nelson word with
        # the most cue-direction associates (i.e., the most-normed lemma).
        # Used by the stem fallback.
        self._stem_to_word: dict[str, str] = {}
        if fallback == "stem":
            try:
                from nltk.stem import PorterStemmer
                self._stemmer: object = PorterStemmer()
                stem_best: dict[str, tuple[int, str]] = {}
                for w in vocab_list:
                    stem = self._stemmer.stem(w)
                    n_responses = len(by_cue.get(w, []))
                    cur = stem_best.get(stem)
                    if cur is None or n_responses > cur[0]:
                        stem_best[stem] = (n_responses, w)
                self._stem_to_word = {
                    stem: word for stem, (_n, word) in stem_best.items()
                }
            except ImportError:
                # nltk not available; degrade to random fallback silently
                self._fallback = "random"
                self._stemmer = None
        else:
            self._stemmer = None

        if vocabulary is not None:
            req = {w.lower() for w in vocabulary}
            self.missing_words: tuple[str, ...] = tuple(sorted(req - vocab_set))
            # Pre-compute missing_after_fallback: which req words have no
            # stem match either. With fallback=random these all fall through;
            # with fallback=stem only the truly-unresolvable ones do.
            if self._fallback == "stem" and self._stemmer is not None:
                unresolved = []
                for w in self.missing_words:
                    stem = self._stemmer.stem(w)
                    if stem not in self._stem_to_word:
                        unresolved.append(w)
                self.missing_after_fallback: tuple[str, ...] = tuple(sorted(unresolved))
            elif self._fallback == "strict":
                self.missing_after_fallback = self.missing_words
            else:
                self.missing_after_fallback = self.missing_words
        else:
            self.missing_words = ()
            self.missing_after_fallback = ()

        rng = np.random.default_rng(seed)
        self._projection: np.ndarray = rng.standard_normal((V, D)).astype(np.float32)
        self._cache: dict[str, SparseBinaryHV] = {}

    @property
    def D(self) -> int:
        return self._D

    @property
    def K(self) -> int:
        return self._K

    @property
    def vocab_size(self) -> int:
        return self._V

    def _random_fallback(self, word: str) -> SparseBinaryHV:
        """Deterministic random sparse code for a word missing from Nelson.

        Seeded by ``(encoder_seed, word)`` via SHA-256 so the same word
        always maps to the same code in a given encoder instance and
        across processes. Used when a word has no Nelson presence and
        cannot be resolved via stemming.
        """
        rng = np.random.default_rng(_stable_word_seed(self._seed, word))
        indices = np.sort(
            rng.choice(self._D, self._K, replace=False)
        ).astype(np.int32)
        hv = SparseBinaryHV(D=self._D, indices=indices)
        return hv

    def _resolve_word(self, w: str) -> str | None:
        """Return a Nelson-vocabulary word to use for encoding ``w``.

        Direct match if ``w`` is in Nelson; otherwise consult the stem
        index if ``fallback == "stem"``. Returns ``None`` when no
        resolution exists (caller falls through to random or raises).
        """
        if w in self._word_to_idx:
            return w
        if self._fallback == "stem" and self._stemmer is not None:
            stem = self._stemmer.stem(w)
            return self._stem_to_word.get(stem)
        return None

    def encode(self, word: str) -> SparseBinaryHV:
        w = word.lower()
        if w in self._cache:
            return self._cache[w]

        resolved = self._resolve_word(w)
        if resolved is None:
            if self._fallback == "strict":
                raise KeyError(
                    f"word '{word}' not in Nelson norms vocabulary "
                    f"({self._V} unique words). Construct with "
                    "fallback='stem' or fallback='random' to handle "
                    "out-of-vocabulary words."
                )
            hv = self._random_fallback(w)
            self._cache[w] = hv
            return hv

        feat: dict[int, float] = {}
        for row in self._data["by_cue"].get(resolved, []):
            target = row[0]
            fsg = row[1]
            if target in self._word_to_idx:
                i = self._word_to_idx[target]
                if fsg > feat.get(i, 0.0):
                    feat[i] = fsg
        for row in self._data["by_target"].get(resolved, []):
            cue = row[0]
            fsg = row[1]
            if cue in self._word_to_idx:
                i = self._word_to_idx[cue]
                if fsg > feat.get(i, 0.0):
                    feat[i] = fsg

        if not feat:
            hv = self._random_fallback(w)
            self._cache[w] = hv
            return hv

        indices = np.fromiter(feat.keys(), dtype=np.int32, count=len(feat))
        values = np.fromiter(feat.values(), dtype=np.float32, count=len(feat))
        rows = self._projection[indices]  # (n_feat, D)
        projected: np.ndarray = values @ rows  # (D,)

        top_k = np.argpartition(np.abs(projected), -self._K)[-self._K :]
        sorted_indices = np.sort(top_k).astype(np.int32)
        hv = SparseBinaryHV(D=self._D, indices=sorted_indices)
        self._cache[w] = hv
        return hv


def make_nelson_factory(
    path: Path | str,
    vocabulary: set[str] | tuple[str, ...] | list[str] | None = None,
    *,
    D: int = DEFAULT_D,
    K: int = DEFAULT_K,
    fallback: str = "stem",
):
    """Pre-load Nelson norms once and return a per-seed encoder factory.

    Mirrors ``make_conceptnet_factory``. The factory creates fresh encoders
    (each with its own random-projection matrix) but shares the parsed
    norms dict across seeds, eliminating the ~3 MB JSON load per seed.
    The ``fallback`` policy is propagated to every encoder instance.
    """
    norms = load_nelson_norms(path)

    def factory(seed: int) -> NelsonAssociationEncoder:
        return NelsonAssociationEncoder(
            norms, vocabulary=vocabulary, D=D, K=K, seed=seed, fallback=fallback,
        )

    return factory


class SyntheticClusterEncoder:
    """Development encoder: structured concept clusters from DRM lists.

    For development and validation purposes only. **Not for paper results.**

    Each DRMList defines a "concept cluster". The critical lure is the
    cluster's prototype (a random sparse-binary base vector); all of the
    list's associates are encodings derived by perturbing that base — drop
    ``drop_frac`` of the base's bits and replace with random non-base bits.
    Result: associates have similarity ~(1 - drop_frac) with the lure and
    ~(1 - drop_frac)^2 with each other. Words in different lists have base
    vectors with chance-floor overlap.

    This encoder mimics ConceptNet Numberbatch's structural property
    (semantically related words have overlapping codes) without requiring
    external data, letting us validate the substrate's DRM prediction
    end-to-end on the laptop. The paper's primary results require
    ConceptNetNumberbatchEncoder once that's wired up.

    Encodes only the words present in the constructor's ``drm_lists``.
    Encoding an unknown word raises KeyError. To extend the vocabulary,
    construct with the full list of DRMLists plus any extra words to
    pre-register via ``extra_words``.
    """

    def __init__(
        self,
        drm_lists: "tuple[object, ...]",  # tuple[DRMList, ...] — typed in stimuli
        *,
        D: int = DEFAULT_D,
        K: int = DEFAULT_K,
        drop_frac: float = 0.35,
        mediated_drop_frac: float = 0.55,
        seed: int = 0,
        extra_words: tuple[str, ...] = (),
    ) -> None:
        if not (0.0 < drop_frac < 1.0):
            raise ValueError(f"drop_frac must be in (0, 1), got {drop_frac}")
        if not (0.0 < mediated_drop_frac < 1.0):
            raise ValueError(
                f"mediated_drop_frac must be in (0, 1), got {mediated_drop_frac}"
            )
        if mediated_drop_frac <= drop_frac:
            raise ValueError(
                f"mediated_drop_frac ({mediated_drop_frac}) must exceed "
                f"drop_frac ({drop_frac}) — mediated lures must be FURTHER "
                f"from the cluster centroid than studied associates, otherwise "
                f"the FTT-AM differential prediction collapses."
            )
        self._D = D
        self._K = K
        self._drop_frac = drop_frac
        self._mediated_drop_frac = mediated_drop_frac
        self._cache: dict[str, SparseBinaryHV] = {}

        rng = np.random.default_rng(seed)

        # Each list gets a base vector (the lure's encoding). Associates are
        # close perturbations; mediated lures are FAR perturbations of the
        # SAME base — close enough to the direct lure that iterated retrieval
        # can spread to find them, but not close enough to the cluster
        # centroid for single-step retrieval to recognize them directly.
        for dl in drm_lists:
            critical_lure = getattr(dl, "critical_lure", None)
            associates = getattr(dl, "associates", None)
            if critical_lure is None or associates is None:
                raise TypeError(
                    f"SyntheticClusterEncoder expected DRMList-shaped objects "
                    f"(.critical_lure, .associates), got {type(dl).__name__}"
                )

            base = self._random_hv(rng)
            self._cache[critical_lure] = base
            for w in associates:
                self._cache[w] = self._perturb(base, self._drop_frac, rng)

            mediated_lure = getattr(dl, "mediated_lure", None)
            if mediated_lure is not None:
                self._cache[mediated_lure] = self._perturb(
                    base, self._mediated_drop_frac, rng
                )

        # Pre-register extra orthogonal words (e.g. distractors not in any list)
        for w in extra_words:
            if w not in self._cache:
                self._cache[w] = self._random_hv(rng)

    @property
    def D(self) -> int:
        return self._D

    @property
    def K(self) -> int:
        return self._K

    def _random_hv(self, rng: np.random.Generator) -> SparseBinaryHV:
        indices = np.sort(rng.choice(self._D, self._K, replace=False)).astype(np.int32)
        return SparseBinaryHV(D=self._D, indices=indices)

    def _perturb(
        self,
        base: SparseBinaryHV,
        drop_frac: float,
        rng: np.random.Generator,
    ) -> SparseBinaryHV:
        n_drop = int(base.K * drop_frac)
        n_keep = base.K - n_drop
        kept = rng.choice(base.indices, size=n_keep, replace=False)
        non_base = np.setdiff1d(
            np.arange(base.D, dtype=np.int32), base.indices, assume_unique=True
        )
        added = rng.choice(non_base, size=n_drop, replace=False)
        new = np.sort(np.concatenate([kept, added])).astype(np.int32)
        return SparseBinaryHV(D=base.D, indices=new)

    def encode(self, word: str) -> SparseBinaryHV:
        if word not in self._cache:
            raise KeyError(
                f"word '{word}' not in SyntheticClusterEncoder vocabulary. "
                "Construct the encoder with all relevant DRMLists or pass "
                "the word in `extra_words`."
            )
        return self._cache[word]


class CooccurrenceEncoder:
    """Sparse-binary codes from corpus co-occurrence statistics.

    Pipeline (when implemented):
      1. Read a small text corpus (e.g., Wikipedia subset, ~100 MB).
      2. Build a sentence-window co-occurrence matrix for the target
         vocabulary.
      3. Reduce via PPMI weighting + optional SVD.
      4. Project to D dimensions, top-K binarize.

    Less-trained than ConceptNet Numberbatch (which is itself derived from
    GloVe/word2vec + ConceptNet relations), but more structured than random
    orthogonal codes. Tests the dose-response of encoding richness on DRM
    rate per §6.2.

    Requires a text corpus (default suggestion: 100 MB Wikipedia subset).
    """

    def __init__(
        self,
        vocabulary: set[str] | tuple[str, ...] | list[str],
        *,
        corpus: str = "brown",
        D: int = DEFAULT_D,
        K: int = DEFAULT_K,
        seed: int = 0,
        window_size: int = 5,
    ) -> None:
        """Build a co-occurrence-based encoder from an NLTK corpus.

        Args:
            vocabulary: words to encode (only these get co-occurrence counts).
            corpus: NLTK corpus identifier. "brown" (default, ~1M words,
                balanced) is downloaded automatically. Other options:
                "reuters", "gutenberg". Must already be available via
                ``nltk.download``.
            window_size: number of tokens on each side to count
                co-occurrences within. Default 5.
        """
        if D <= 0:
            raise ValueError(f"D must be positive, got {D}")
        if K < 1 or K > D:
            raise ValueError(f"K must be in [1, {D}], got {K}")

        self._D = D
        self._K = K
        self._seed = seed
        self._window_size = window_size
        self._corpus = corpus
        self._cache: dict[str, SparseBinaryHV] = {}

        vocab_list = sorted(set(w.lower() for w in vocabulary))
        self._vocab_list = vocab_list
        self._vocab_index = {w: i for i, w in enumerate(vocab_list)}
        V = len(vocab_list)

        # Build co-occurrence matrix from NLTK corpus.
        # Restrict to within-vocabulary tokens; window-based counts.
        import nltk

        try:
            nltk.data.find(f"corpora/{corpus}")
        except LookupError:
            nltk.download(corpus, quiet=True)

        from nltk.corpus import reuters, brown, gutenberg
        corpora = {"brown": brown, "reuters": reuters, "gutenberg": gutenberg}
        if corpus not in corpora:
            raise ValueError(
                f"unknown corpus '{corpus}'. Supported: {list(corpora.keys())}"
            )

        counts = np.zeros((V, V), dtype=np.float64)
        vocab_set = set(vocab_list)
        for sent in corpora[corpus].sents():
            # Lower-case tokens, keep only those in vocabulary
            tokens = [t.lower() for t in sent]
            indexed = [
                (i, t) for i, t in enumerate(tokens) if t in vocab_set
            ]
            for a_pos, (a_idx, a) in enumerate(indexed):
                ai = self._vocab_index[a]
                for _b_pos, (b_idx, b) in enumerate(indexed):
                    if a_idx == b_idx:
                        continue
                    if abs(a_idx - b_idx) > window_size:
                        continue
                    bi = self._vocab_index[b]
                    counts[ai, bi] += 1.0

        # PPMI weighting: positive pointwise mutual information.
        # P(a,b) / (P(a) P(b)) — clamp negative log to 0.
        total = counts.sum()
        if total > 0:
            row_sums = counts.sum(axis=1, keepdims=True) + 1e-12
            col_sums = counts.sum(axis=0, keepdims=True) + 1e-12
            expected = row_sums @ col_sums / total
            with np.errstate(divide="ignore", invalid="ignore"):
                pmi = np.log((counts + 1e-12) / (expected + 1e-12))
            ppmi = np.maximum(pmi, 0.0)
        else:
            ppmi = counts

        self._ppmi = ppmi.astype(np.float32)

        # Random projection V -> D for sparsification.
        rng = np.random.default_rng(seed)
        self._projection: np.ndarray = rng.standard_normal(
            (V, D)
        ).astype(np.float32)

    @property
    def D(self) -> int:
        return self._D

    @property
    def K(self) -> int:
        return self._K

    def encode(self, word: str) -> SparseBinaryHV:
        w = word.lower()
        if w in self._cache:
            return self._cache[w]
        if w not in self._vocab_index:
            raise KeyError(
                f"word '{word}' not in CooccurrenceEncoder vocabulary. "
                "Pass it via the constructor's `vocabulary` parameter."
            )
        idx = self._vocab_index[w]
        dense = self._ppmi[idx]  # V-dim PPMI row
        projected = dense @ self._projection  # D-dim
        top_k = np.argpartition(np.abs(projected), -self._K)[-self._K :]
        indices = np.sort(top_k).astype(np.int32)
        hv = SparseBinaryHV(D=self._D, indices=indices)
        self._cache[w] = hv
        return hv


def make_conceptnet_factory(
    path: Path | str,
    vocabulary: set[str] | tuple[str, ...] | list[str],
    *,
    D: int = DEFAULT_D,
    K: int = DEFAULT_K,
):
    """Pre-load Numberbatch embeddings once and return a per-seed encoder factory.

    Use this in battery and sweep call sites instead of constructing an
    encoder from-file per seed. The factory creates fresh encoders (each
    with its own random-projection matrix) but shares the underlying dense
    embeddings dict.

    Example::

        factory = make_conceptnet_factory(path, vocab)
        battery = run_battery(lists, factory, n_seeds=50, ...)

    Returns:
        Callable[[int], ConceptNetNumberbatchEncoder]: seed -> encoder.
    """
    embeddings = load_numberbatch(path, vocabulary)

    def factory(seed: int) -> ConceptNetNumberbatchEncoder:
        return ConceptNetNumberbatchEncoder(embeddings, D=D, K=K, seed=seed)

    return factory
