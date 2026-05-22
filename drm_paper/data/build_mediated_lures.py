"""Auto-generate mediated lures for the full 55-list battery.

For each critical lure, search ConceptNet Numberbatch for the word with
highest cosine similarity to the lure, subject to filters:

  - Not in this list's studied associates
  - Not the critical lure of any list
  - Not in any other list's studied associates
  - Not in DEFAULT_FOILS
  - At least 3 characters long (filter trivial fragments)
  - Letters only (no numerals or punctuation)
  - Not a stop word
  - Cosine to lure in a sensible mediated range (0.30 <= cos <= 0.80):
    high enough to be plausibly one-hop, not so high it's a synonym

Output: drm_paper/data/mediated_lures_55.json
        {critical_lure: mediated_lure_word}

Run from repo root::

    .venv/bin/python drm_paper/data/build_mediated_lures.py
"""

from __future__ import annotations

import gzip
import json
import re
from pathlib import Path

import numpy as np


# Common English stopwords to exclude — these have semantic-poor
# embeddings that occasionally rank high spuriously.
STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "must", "shall", "can", "this",
    "that", "these", "those", "i", "you", "he", "she", "it", "we", "they",
    "what", "which", "who", "whom", "whose", "where", "when", "why", "how",
    "all", "each", "every", "some", "any", "no", "none", "one", "two",
    "yes", "not", "as", "so", "very", "much", "many", "more", "most",
    "less", "only", "other", "such", "than", "then", "if", "though",
}


WORD_RE = re.compile(r"^[a-z]+$")  # single word, lowercase letters only


def cosine_top_k(query: np.ndarray, matrix: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (indices, similarities) of top-k rows in matrix by cosine to query."""
    q_norm = query / (np.linalg.norm(query) + 1e-12)
    m_norm = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12)
    sims = m_norm @ q_norm
    top = np.argpartition(-sims, min(k, len(sims) - 1))[:k]
    top_sorted = top[np.argsort(-sims[top])]
    return top_sorted, sims[top_sorted]


def main() -> None:
    repo = Path(__file__).parent.parent.parent
    lists_path = repo / "drm_paper" / "data" / "stadler1999_55lists.json"
    nb_path = repo / "drm_paper" / "data" / "numberbatch-en-19.08.txt.gz"
    out_path = repo / "drm_paper" / "data" / "mediated_lures_55.json"

    lists = json.loads(lists_path.read_text())
    critical_lures = {item["critical_lure"] for item in lists}
    all_associates: set[str] = set()
    for item in lists:
        all_associates.update(item["associates"])
    from drm_paper.stimuli import DEFAULT_FOILS
    foils = set(DEFAULT_FOILS)
    excluded = critical_lures | all_associates | foils | STOPWORDS

    # Restrict candidates to common English words. Use NLTK's Brown corpus
    # frequency distribution as the inclusion gate — captures realistic
    # English vocabulary without exotic ConceptNet compounds.
    print("Loading Brown-corpus common-vocabulary gate...")
    import nltk
    try:
        nltk.data.find("corpora/brown")
    except LookupError:
        nltk.download("brown", quiet=True)
    from collections import Counter
    from nltk.corpus import brown
    word_counts: Counter[str] = Counter(w.lower() for w in brown.words())
    # Words appearing >= 5 times in Brown — ~10k common English words.
    common_english = {w for w, c in word_counts.items() if c >= 5 and WORD_RE.match(w)}
    print(f"  {len(common_english)} common single-word English candidates")

    # Stream Numberbatch and keep only words that ALSO appear in the
    # common-English gate. This filters out ConceptNet's exotic compound
    # entries and rare technical terms.
    print("Streaming Numberbatch for candidate vocabulary...")
    vocab: list[str] = []
    vectors: list[np.ndarray] = []
    with gzip.open(nb_path, "rt", encoding="utf-8") as f:
        _header = f.readline()
        for line in f:
            space = line.find(" ")
            if space < 0:
                continue
            word = line[:space]
            if word in excluded:
                continue
            if word not in common_english:
                continue
            if len(word) < 3:
                continue
            vec = np.fromstring(line[space + 1 :], sep=" ", dtype=np.float32)
            vocab.append(word)
            vectors.append(vec)
    matrix = np.stack(vectors).astype(np.float32)
    print(f"  loaded {len(vocab)} candidate words")

    # Load embeddings for the critical lures.
    print("Loading lure embeddings...")
    lure_to_vec: dict[str, np.ndarray] = {}
    with gzip.open(nb_path, "rt", encoding="utf-8") as f:
        _header = f.readline()
        for line in f:
            space = line.find(" ")
            if space < 0:
                continue
            word = line[:space]
            if word in critical_lures:
                lure_to_vec[word] = np.fromstring(
                    line[space + 1 :], sep=" ", dtype=np.float32
                )
                if len(lure_to_vec) == len(critical_lures):
                    break

    # Reject morphological variants of the lure or its studied associates
    # (e.g., spider -> crawling when "crawl" is in spider's associates).
    # Mediated lures must be distinct concepts, not inflected forms of
    # already-studied items.
    from nltk.stem import PorterStemmer
    stemmer = PorterStemmer()
    lure_stems = {lure: stemmer.stem(lure) for lure in critical_lures}
    per_list_associate_stems: dict[str, set[str]] = {}
    for item in lists:
        per_list_associate_stems[item["critical_lure"]] = {
            stemmer.stem(a) for a in item["associates"]
        }

    print("Selecting mediated lures...")
    mediated: dict[str, dict] = {}
    used: set[str] = set()
    for lure in sorted(critical_lures):
        if lure not in lure_to_vec:
            print(f"  WARNING: lure '{lure}' not in ConceptNet")
            continue
        q = lure_to_vec[lure]
        top_idx, top_sim = cosine_top_k(q, matrix, k=500)
        chosen = None
        for idx, sim in zip(top_idx, top_sim):
            cand = vocab[idx]
            if cand in used:
                continue
            if not (0.30 <= sim <= 0.70):
                continue
            cand_stem = stemmer.stem(cand)
            # Reject candidates with the same stem as the lure
            # (filters out plural/comparative/gerund forms).
            if cand_stem == lure_stems[lure]:
                continue
            # Reject candidates sharing a stem with any studied associate
            # for THIS list — these are too directly tied to the studied gist.
            if cand_stem in per_list_associate_stems[lure]:
                continue
            # Reject candidates whose lure is a substring or vice versa
            # — catches simple morphological variants the stemmer misses.
            if lure in cand or cand in lure:
                continue
            chosen = (cand, float(sim))
            break
        if chosen is None:
            print(f"  WARNING: no candidate found for '{lure}'")
            continue
        mediated[lure] = {"word": chosen[0], "cosine_to_lure": chosen[1]}
        used.add(chosen[0])
        print(f"  {lure:>12} -> {chosen[0]:<20} (cos={chosen[1]:.3f})")

    out_path.write_text(json.dumps(mediated, indent=2))
    print(f"\nWrote {len(mediated)} mediated lures to {out_path}")


if __name__ == "__main__":
    main()
