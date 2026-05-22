"""Parse the Nelson, McEvoy & Schreiber (1998) USF Free Association Norms
into a compact JSON for use by ``NelsonAssociationEncoder``.

Input:
    drm_paper/data/nelson_norms_raw.txt
    (the flat-file dump from the USF norms, redistributed at
     https://github.com/teonbrooks/free_association as
     ``free/free_association.txt``; ~10 MB, 72,184 cue-target rows.)

Output:
    drm_paper/data/nelson_norms.json
    {
        "metadata": {...},
        "by_cue":    {<cue>: [[target, fsg, bsg, normed_bool], ...]},
        "by_target": {<target>: [[cue,   fsg, bsg], ...]}
    }

The two indices are redundant but pre-built so the encoder doesn't have
to invert the cue->target table at load time. Words are lowercased and
single-word entries only (multi-word targets like "PINK PANTHER" are
kept under the lowercased phrase). FSG = forward strength = P(target|cue);
BSG = backward strength = P(cue|target) when target was also normed as
a cue, else null. ``normed_bool`` mirrors the NORMED? column.

Re-runnable; idempotent. Run::

    python drm_paper/data/build_nelson_norms.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

RAW_PATH = Path(__file__).parent / "nelson_norms_raw.txt"
OUT_PATH = Path(__file__).parent / "nelson_norms.json"

# The USF file uses "�" (U+FFFD or Latin-1 0xAB depending on encoding) as
# a sentinel for "not available" (target not separately normed as a cue,
# so its backward-strength row is missing). We accept several variants
# in case the file was re-encoded somewhere along the line.
_NA_TOKENS = {"�", "\xab", "?", ""}


def _parse_float(token: str) -> float | None:
    """Parse a number from a USF cell. Returns None for any NA sentinel."""
    s = token.strip()
    if not s or s in _NA_TOKENS or s.startswith("�"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def build(raw_path: Path = RAW_PATH, out_path: Path = OUT_PATH) -> dict:
    """Parse the flat file and write JSON. Returns the in-memory dict too."""
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Nelson norms raw file not found at {raw_path}. "
            "Download from https://github.com/teonbrooks/free_association "
            "(file free/free_association.txt) and place at this path."
        )

    by_cue: dict[str, list[tuple[str, float, float | None, bool]]] = defaultdict(list)
    by_target: dict[str, list[tuple[str, float, float | None]]] = defaultdict(list)

    n_rows_seen = 0
    n_rows_kept = 0
    n_bsg_present = 0
    cues_seen: set[str] = set()
    targets_seen: set[str] = set()

    # Use latin-1 to be permissive about the sentinel byte; the file's
    # actual content is ASCII otherwise.
    with raw_path.open("r", encoding="latin-1") as f:
        header = f.readline()  # skip header line
        if "CUE" not in header.upper():
            raise ValueError(
                f"Unexpected header in {raw_path}: {header[:80]!r}. "
                "Expected a comma-separated row starting with CUE,TARGET,..."
            )

        for line in f:
            n_rows_seen += 1
            cells = [c.strip() for c in line.split(",")]
            if len(cells) < 7:
                continue  # malformed; skip
            cue = cells[0].lower()
            target = cells[1].lower()
            if not cue or not target:
                continue
            normed_str = cells[2].upper()
            fsg = _parse_float(cells[5])
            bsg = _parse_float(cells[6])
            if fsg is None:
                continue  # forward strength missing = unusable row

            normed = normed_str == "YES"
            by_cue[cue].append((target, fsg, bsg, normed))
            if bsg is not None:
                by_target[target].append((cue, fsg, bsg))
                n_bsg_present += 1
            else:
                by_target[target].append((cue, fsg, None))

            cues_seen.add(cue)
            targets_seen.add(target)
            n_rows_kept += 1

    # Sort each cue's responses by FSG descending for predictable iteration.
    for cue in by_cue:
        by_cue[cue].sort(key=lambda row: -row[1])
    for tgt in by_target:
        by_target[tgt].sort(key=lambda row: -row[1])

    vocab = cues_seen | targets_seen

    payload = {
        "metadata": {
            "source": (
                "Nelson, D. L., McEvoy, C. L., & Schreiber, T. A. (1998). "
                "The University of South Florida word association, rhyme, "
                "and word fragment norms."
            ),
            "via": "github.com/teonbrooks/free_association",
            "n_rows_seen": n_rows_seen,
            "n_rows_kept": n_rows_kept,
            "n_cues": len(cues_seen),
            "n_targets": len(targets_seen),
            "n_unique_words": len(vocab),
            "n_bsg_present": n_bsg_present,
        },
        "by_cue": {c: by_cue[c] for c in sorted(by_cue)},
        "by_target": {t: by_target[t] for t in sorted(by_target)},
    }

    out_path.write_text(json.dumps(payload, separators=(",", ":")))
    return payload


def main() -> None:
    payload = build()
    meta = payload["metadata"]
    print(f"Parsed {meta['n_rows_kept']:,} / {meta['n_rows_seen']:,} rows")
    print(f"  unique cues:    {meta['n_cues']:,}")
    print(f"  unique targets: {meta['n_targets']:,}")
    print(f"  unique words:   {meta['n_unique_words']:,}")
    print(f"  with BSG:       {meta['n_bsg_present']:,}")
    print(f"Wrote {OUT_PATH} ({OUT_PATH.stat().st_size / 1024:.0f} KB)")

    # Spot-check: doctor list canonical lure should have many associates.
    doctor_responses = payload["by_cue"].get("doctor", [])
    print(f"\nSpot check — 'doctor' as cue: {len(doctor_responses)} responses")
    for target, fsg, bsg, normed in doctor_responses[:10]:
        bsg_str = f"{bsg:.3f}" if bsg is not None else "  --"
        print(f"  {target:<20s} fsg={fsg:.3f} bsg={bsg_str} normed={normed}")


if __name__ == "__main__":
    main()
