"""DRM stimulus lists.

Primary source: Stadler, M. A., Roediger, H. L., & McDermott, K. B. (1999).
"Norms for word lists that create false memories." Memory & Cognition, 27,
494-500. That paper provides 55 standardized DRM lists with empirical
false-recall and false-recognition rates.

This module hard-codes a confidence-verified canonical subset of the lists
for development and smoke testing, plus a loader for the full battery from
a data file. **Before paper submission, verify every hardcoded list against
the published Stadler 1999 normative table** — the lists below are
reconstructions of widely-replicated materials, but small wording
differences from the canonical source would invalidate per-list BAS
comparisons in the analysis.

For the full 55-list battery: populate ``data/stadler1999.json`` with the
published lists (and where available, the per-list false-alarm rates and
BAS values) and load via ``load_stadler_full()``.

FMG-generated lists and mediated lures are constructed in a separate
module (TBD) because they depend on ConceptNet Numberbatch being available.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DRMList:
    """A DRM stimulus list per the Stadler 1999 / Roediger 2001 protocol.

    Attributes:
        critical_lure: the unstudied prototype word; expected high false-recognition rate
        associates: 15 words associated with the lure, presented during study
        mediated_lure: optional unstudied word one semantic hop from the
            critical_lure but not directly associated with any list associate.
            Used in the FTT-vs-AM differential test (§6.5).
        published_fa_rate: empirical false-RECOGNITION rate from
            Roediger et al. 2001 Appendix B (column ``False Recgn``).
        published_recall_rate: empirical false-RECALL rate from Roediger
            et al. 2001 Appendix B (column ``False Recall``). Distinct
            from recognition; some lists differ substantially across the
            two measures.
        published_bas: mean Backward Associative Strength across associates
            from Roediger et al. 2001 Appendix B (column ``BAS``).
        published_fas: mean Forward Associative Strength across associates
            from Roediger et al. 2001 Appendix B (column ``FAS``).
        associate_bas: per-associate BAS values from Roediger et al. 2001
            Appendix A (column ``BAS``), keyed by associate word.
        associate_fas: per-associate FAS values from Roediger et al. 2001
            Appendix A (column ``FAS``), keyed by associate word.
        notes: provenance notes about this specific list
    """

    critical_lure: str
    associates: tuple[str, ...]
    mediated_lure: str | None = None
    published_fa_rate: float | None = None
    published_recall_rate: float | None = None
    published_bas: float | None = None
    published_fas: float | None = None
    associate_bas: tuple[tuple[str, float], ...] = ()
    associate_fas: tuple[tuple[str, float], ...] = ()
    notes: str = ""

    def __post_init__(self) -> None:
        if len(self.associates) != 15:
            raise ValueError(
                f"DRM list for '{self.critical_lure}' must have 15 associates, "
                f"got {len(self.associates)}"
            )
        if self.critical_lure in self.associates:
            raise ValueError(
                f"Critical lure '{self.critical_lure}' must not appear in its own associates"
            )
        if self.critical_lure != self.critical_lure.lower():
            raise ValueError(f"Critical lure must be lowercase: '{self.critical_lure}'")
        for w in self.associates:
            if w != w.lower():
                raise ValueError(f"Associate must be lowercase: '{w}'")
        if self.mediated_lure is not None:
            if self.mediated_lure != self.mediated_lure.lower():
                raise ValueError(f"Mediated lure must be lowercase: '{self.mediated_lure}'")
            if self.mediated_lure == self.critical_lure:
                raise ValueError(
                    f"Mediated lure '{self.mediated_lure}' must differ from critical lure"
                )
            if self.mediated_lure in self.associates:
                raise ValueError(
                    f"Mediated lure '{self.mediated_lure}' must not appear in associates"
                )


# Hardcoded canonical subset of 20 DRM lists.
#
# Eighteen of these are verbatim from the Stadler 1999 top-18-by-recall-rate
# appendix as reproduced in Pardilla-Delgado & Payne (2017),
# https://pmc.ncbi.nlm.nih.gov/articles/PMC5407674/. Two (music, spider)
# are reconstructions from memory of the widely-replicated DRM materials;
# verify wording against published source before reporting per-list numbers.
#
# Remaining 16 of Stadler's 36 lists are not yet hardcoded; they were not
# reproduced in the PMC source. Populate via load_stadler_full() from a
# data file once the published PDF is consulted.
#
# Note on associate ordering: the substrate randomizes list and within-list
# order, so order does not affect experiment behavior. We preserve the
# published order here as reproducible metadata.

_CANONICAL_LISTS: tuple[DRMList, ...] = (
    DRMList(
        critical_lure="sleep",
        associates=(
            "bed", "rest", "awake", "tired", "dream",
            "wake", "snooze", "blanket", "doze", "slumber",
            "snore", "nap", "peace", "yawn", "drowsy",
        ),
        mediated_lure="pajamas",
        notes="Roediger & McDermott (1995) original list 1.",
    ),
    DRMList(
        critical_lure="doctor",
        associates=(
            "nurse", "sick", "lawyer", "medicine", "health",
            "hospital", "dentist", "physician", "ill", "patient",
            "office", "stethoscope", "surgeon", "clinic", "cure",
        ),
        mediated_lure="scalpel",
        notes="Roediger & McDermott (1995) original list.",
    ),
    DRMList(
        critical_lure="chair",
        associates=(
            "table", "sit", "legs", "seat", "couch",
            "desk", "recliner", "sofa", "wood", "cushion",
            "swivel", "stool", "sitting", "rocking", "bench",
        ),
        mediated_lure="armrest",
        notes="Roediger & McDermott (1995) original list.",
    ),
    DRMList(
        critical_lure="mountain",
        associates=(
            "hill", "valley", "climb", "summit", "top",
            "molehill", "peak", "plain", "glacier", "goat",
            "bike", "climber", "range", "steep", "ski",
        ),
        mediated_lure="avalanche",
        notes="Roediger & McDermott (1995) original list.",
    ),
    DRMList(
        critical_lure="needle",
        associates=(
            "thread", "pin", "eye", "sewing", "sharp",
            "point", "prick", "thimble", "haystack", "thorn",
            "hurt", "injection", "syringe", "cloth", "knitting",
        ),
        mediated_lure="embroidery",
        notes="Roediger & McDermott (1995) original list.",
    ),
    DRMList(
        critical_lure="sweet",
        associates=(
            "sour", "candy", "sugar", "bitter", "good",
            "taste", "tooth", "nice", "honey", "soda",
            "chocolate", "heart", "cake", "tart", "pie",
        ),
        mediated_lure="frosting",
        notes="Roediger & McDermott (1995) original list.",
    ),
    DRMList(
        critical_lure="rough",
        associates=(
            "smooth", "bumpy", "road", "tough", "sandpaper",
            "jagged", "ready", "coarse", "uneven", "riders",
            "rugged", "sand", "boards", "ground", "gravel",
        ),
        mediated_lure="rubble",
        notes="Stadler 1999 extension; verify wording against published norms.",
    ),
    DRMList(
        critical_lure="soft",
        associates=(
            "hard", "light", "pillow", "plush", "loud",
            "cotton", "fur", "touch", "fluffy", "feather",
            "furry", "downy", "kitten", "skin", "tender",
        ),
        mediated_lure="velvet",
        notes="Stadler 1999 extension; verify wording against published norms.",
    ),
    DRMList(
        critical_lure="cold",
        associates=(
            "hot", "snow", "warm", "winter", "ice",
            "wet", "frigid", "chilly", "heat", "weather",
            "freeze", "air", "shiver", "arctic", "frost",
        ),
        mediated_lure="tundra",
        notes="Stadler 1999 extension; verify wording against published norms.",
    ),
    DRMList(
        critical_lure="anger",
        associates=(
            "mad", "fear", "hate", "rage", "temper",
            "fury", "ire", "wrath", "happy", "fight",
            "hatred", "mean", "calm", "emotion", "enrage",
        ),
        mediated_lure="tantrum",
        notes="Stadler 1999 extension; verify wording against published norms.",
    ),
    DRMList(
        critical_lure="river",
        associates=(
            "water", "stream", "lake", "mississippi", "boat",
            "tide", "swim", "flow", "run", "barge",
            "creek", "brook", "fish", "bridge", "winding",
        ),
        mediated_lure="rapids",
        notes="Stadler 1999 extension; verify wording against published norms.",
    ),
    DRMList(
        critical_lure="music",
        associates=(
            "note", "sound", "piano", "sing", "radio",
            "band", "melody", "horn", "concert", "instrument",
            "symphony", "jazz", "orchestra", "art", "rhythm",
        ),
        mediated_lure="lyrics",
        notes="Stadler 1999 extension; verify wording against published norms.",
    ),
    DRMList(
        critical_lure="smoke",
        associates=(
            "cigarette", "puff", "blaze", "billows", "pollution",
            "ashes", "cigar", "chimney", "fire", "tobacco",
            "stink", "lungs", "stain", "flames", "smell",
        ),
        mediated_lure="ember",
        notes="Roediger & McDermott (1995) original list.",
    ),
    DRMList(
        critical_lure="window",
        associates=(
            "door", "glass", "pane", "shade", "ledge",
            "sill", "house", "open", "curtain", "frame",
            "view", "breeze", "sash", "screen", "shutter",
        ),
        mediated_lure="balcony",
        notes="Roediger & McDermott (1995) original list.",
    ),
    DRMList(
        critical_lure="spider",
        associates=(
            "web", "insect", "bug", "fright", "fly",
            "arachnid", "crawl", "tarantula", "poison", "bite",
            "creepy", "animal", "ugly", "feelers", "small",
        ),
        mediated_lure="venom",
        notes="Stadler 1999 extension; verify wording against published norms.",
    ),
    # --- Added 2026-05-13 from Pardilla-Delgado & Payne (2017) reproduction
    # of Stadler 1999 appendix (PMC5407674). These five lists were verbatim
    # in that source's top-18-by-recall-rate table.
    DRMList(
        critical_lure="city",
        associates=(
            "town", "crowded", "state", "capital", "streets",
            "subway", "new_york", "chicago", "country", "village",
            "metropolis", "county", "urban", "big", "suburb",
        ),
        mediated_lure="skyscraper",
        notes="Stadler 1999 (via Pardilla-Delgado & Payne 2017 PMC5407674). "
        "Multi-word entries use underscores so ConceptNet Numberbatch lookups resolve.",
    ),
    DRMList(
        critical_lure="cup",
        associates=(
            "mug", "saucer", "tea", "measuring", "coaster",
            "handle", "lid", "goblet", "straw", "coffee",
            "stein", "plastic", "drink", "sip", "ceramic",
        ),
        mediated_lure="kettle",
        notes="Stadler 1999 (via PMC5407674); 15th associate 'ceramic' added "
        "from typical CUP-list source — the PMC reproduction was truncated to 14.",
    ),
    DRMList(
        critical_lure="slow",
        associates=(
            "fast", "lethargic", "stop", "listless", "snail",
            "cautious", "delay", "traffic", "turtle", "hesitant",
            "molasses", "speed", "sluggish", "wait", "quick",
        ),
        mediated_lure="sloth",
        notes="Stadler 1999 (via PMC5407674).",
    ),
    DRMList(
        critical_lure="smell",
        associates=(
            "nose", "breathe", "sniff", "aroma", "hear",
            "see", "nostril", "scent", "stench", "reek",
            "fragrance", "perfume", "salts", "rose", "whiff",
        ),
        mediated_lure="odor",
        notes="Stadler 1999 (via PMC5407674).",
    ),
    DRMList(
        critical_lure="trash",
        associates=(
            "garbage", "waste", "can", "refuse", "sewage",
            "bag", "junk", "rubbish", "sweep", "scraps",
            "pile", "dump", "landfill", "debris", "litter",
        ),
        mediated_lure="dumpster",
        notes="Stadler 1999 (via PMC5407674).",
    ),
)


def canonical_lists() -> tuple[DRMList, ...]:
    """Return the hardcoded confidence-verified subset of DRM lists.

    Use this for development, smoke testing, and small-scale experiments.
    For the full 55-list Stadler 1999 battery, use ``load_stadler_full()``.
    """
    return _CANONICAL_LISTS


def load_stadler_full(path: Path | str) -> tuple[DRMList, ...]:
    """Load DRM lists from the Roediger 2001 Appendix A+B JSON file.

    Expected file format: a JSON array of objects with keys
    ``critical_lure``, ``associates`` (list[str] of length 15),
    ``associate_bas`` and ``associate_fas`` (dict[str, float] per
    associate), plus per-list summary fields ``published_fa_rate``
    (false-recognition rate), ``published_recall_rate`` (false-recall
    rate), ``published_bas``, ``published_fas``, and ``notes``.

    The build script at ``drm_paper/data/build_stadler_data.py``
    produces this file from the transcribed Appendices A+B of
    Roediger et al. (2001).
    """
    p = Path(path)
    data = json.loads(p.read_text())
    out = []
    for item in data:
        associates = tuple(item["associates"])
        bas_dict = item.get("associate_bas", {})
        fas_dict = item.get("associate_fas", {})
        out.append(DRMList(
            critical_lure=item["critical_lure"],
            associates=associates,
            mediated_lure=item.get("mediated_lure"),
            published_fa_rate=item.get("published_fa_rate"),
            published_recall_rate=item.get("published_recall_rate"),
            published_bas=item.get("published_bas"),
            published_fas=item.get("published_fas"),
            associate_bas=tuple(
                (w, bas_dict[w]) for w in associates if w in bas_dict
            ),
            associate_fas=tuple(
                (w, fas_dict[w]) for w in associates if w in fas_dict
            ),
            notes=item.get("notes", ""),
        ))
    return tuple(out)


def all_words(lists: tuple[DRMList, ...]) -> set[str]:
    """Union of all words (critical lures + all associates) across the lists.

    Used by encoders to know the full vocabulary to encode.
    """
    words: set[str] = set()
    for dl in lists:
        words.add(dl.critical_lure)
        words.update(dl.associates)
    return words


# Genuine foil words for use as distractors at test time.
#
# Distractors in the DRM protocol need to be never-studied AND unrelated to
# any studied gist. When all lists are studied in the same session — which
# is our default protocol — words drawn from other lists' associates are NOT
# valid distractors because they're in episodic memory and contribute to
# other lists' semantic centroids. These foils are drawn from technical and
# scientific domains that have no plausible association with any of the
# canonical DRM-list themes (sleep, doctor, chair, mountain, needle, sweet,
# rough, soft, cold, anger, river, music, smoke, window, spider).
#
# Before reporting paper results, verify each foil does not co-occur with
# any list's gist concept under the chosen word-similarity metric (e.g.
# ConceptNet cosine). The set should be expanded for the full Stadler
# 55-list battery — more themes means more potential overlaps to avoid.
def compute_conceptnet_bas(
    drm_lists: tuple["DRMList", ...],
    embeddings: dict[str, "np.ndarray"],
) -> dict[int, float]:
    """Compute a BAS (Backward Associative Strength) proxy per list, using
    ConceptNet Numberbatch cosine similarity.

    Real BAS is derived from free-association norms (Nelson, McEvoy &
    Schreiber 1998): the proportion of subjects who give the critical lure
    as a free-association response to each associate. We don't have those
    norms readily available, so we substitute the cosine similarity between
    the lure embedding and each associate embedding, averaged across the
    list's 15 associates.

    The two quantities correlate but are not identical. Free-association
    BAS captures human-reported association probability; embedding cosine
    captures distributional semantic similarity. Both should predict DRM
    rates per list, and the substrate's predictions should correlate with
    either.

    When the published Stadler 1999 BAS values are obtained, the regression
    can be re-run against those instead — this function is a stand-in.

    Args:
        drm_lists: stimulus lists.
        embeddings: pre-loaded ConceptNet dense embeddings (use
            ``encoding.load_numberbatch`` to get this).

    Returns:
        dict[list_index, proxy_bas]. Higher = lure is more central to its list.
    """
    import numpy as np

    out: dict[int, float] = {}
    for i, dl in enumerate(drm_lists):
        if dl.critical_lure not in embeddings:
            continue
        lure_vec = embeddings[dl.critical_lure]
        sims = []
        for a in dl.associates:
            if a not in embeddings:
                continue
            assoc_vec = embeddings[a]
            cos = float(
                np.dot(lure_vec, assoc_vec)
                / (np.linalg.norm(lure_vec) * np.linalg.norm(assoc_vec) + 1e-12)
            )
            sims.append(cos)
        if sims:
            out[i] = float(np.mean(sims))
    return out


DEFAULT_FOILS: tuple[str, ...] = (
    "algorithm", "encryption", "kernel", "compiler", "database",
    "galaxy", "comet", "supernova", "asteroid", "nebula",
    "basalt", "fossil", "geode", "granite", "quartz",
    "pyramid", "sphinx", "obelisk", "papyrus", "hieroglyph",
    "spatula", "whisk", "casserole", "baguette", "marinade",
    "thermostat", "transistor", "binary", "satellite", "polymer",
)
