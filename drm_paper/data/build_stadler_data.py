"""Build the canonical DRM-list data file from Roediger, Watson, McDermott
& Gallo (2001) Appendices A and B.

Source: Roediger, H. L., Watson, J. M., McDermott, K. B., & Gallo, D. A.
(2001). Factors that determine false recall: A multiple regression
analysis. Psychonomic Bulletin & Review, 8(3), 385-407. (PDF in
drm_paper/data/Factors that determine false recall.pdf)

Appendix A gives, for each of 55 critical-item lists, the 15 associates
with their backward (BAS) and forward (FAS) associative strengths.
Appendix B gives, per list, the empirical false-recall and false-
recognition rates plus mean BAS, FAS, length, log-frequency,
concreteness, connectivity, veridical-recall, raw-frequency,
orthographic-distinctiveness, and Coltheart's N (orthographic
neighborhood size).

Asterisked BAS/FAS values in Appendix A were obtained by the authors
using norming procedures similar to Nelson et al. 1999 (i.e., not from
the published Nelson norms). We mark these with the ``*`` suffix in the
raw data and strip the asterisk in the numeric value.

Output: drm_paper/data/stadler1999_55lists.json

Run from repo root::

    .venv/bin/python drm_paper/data/build_stadler_data.py
"""

from __future__ import annotations

import json
from pathlib import Path


# ---------------------------------------------------------------------------
# Appendix B (page 406): per-list summary statistics.
# Columns: false_recall, false_recognition, length, log_freq, concrete,
#          fas_mean, bas_mean, connectivity, veridical_recall, raw_freq,
#          ortho_dist, colt_n
# Mutton row in source has "n.a." for concrete; we store None.
# ---------------------------------------------------------------------------

PER_LIST_DATA: dict[str, dict] = {
    "anger":     {"false_recall": .490, "false_recognition": .790, "length": 5, "log_freq": 1.69, "concrete": 3.75, "fas_mean": .044, "bas_mean": .157, "connectivity": 1.73, "veridical_recall": .500, "raw_freq":  48, "ortho_dist": 4.30, "colt_n":  3},
    "army":      {"false_recall": .250, "false_recognition": .530, "length": 4, "log_freq": 2.12, "concrete": 6.53, "fas_mean": .044, "bas_mean": .135, "connectivity": 2.53, "veridical_recall": .610, "raw_freq": 132, "ortho_dist": 4.12, "colt_n":  2},
    "beautiful": {"false_recall": .030, "false_recognition": .440, "length": 9, "log_freq": 2.11, "concrete": 3.89, "fas_mean": .049, "bas_mean": .038, "connectivity": 2.13, "veridical_recall": .710, "raw_freq": 127, "ortho_dist": 6.38, "colt_n":  0},
    "bitter":    {"false_recall": .010, "false_recognition": .260, "length": 6, "log_freq": 1.73, "concrete": 4.05, "fas_mean": .059, "bas_mean": .011, "connectivity": 1.40, "veridical_recall": .680, "raw_freq":  53, "ortho_dist": 5.78, "colt_n": 10},
    "black":     {"false_recall": .340, "false_recognition": .490, "length": 5, "log_freq": 2.31, "concrete": 4.66, "fas_mean": .052, "bas_mean": .130, "connectivity": 1.47, "veridical_recall": .600, "raw_freq": 203, "ortho_dist": 4.41, "colt_n":  3},
    "bread":     {"false_recall": .310, "false_recognition": .640, "length": 5, "log_freq": 1.62, "concrete": 6.18, "fas_mean": .049, "bas_mean": .200, "connectivity": 1.33, "veridical_recall": .550, "raw_freq":  41, "ortho_dist": 3.97, "colt_n":  5},
    "butterfly": {"false_recall": .010, "false_recognition": .260, "length": 9, "log_freq": 0.40, "concrete": 5.91, "fas_mean": .033, "bas_mean": .045, "connectivity": 1.93, "veridical_recall": .700, "raw_freq":   2, "ortho_dist": 7.44, "colt_n":  0},
    "cabbage":   {"false_recall": .050, "false_recognition": .440, "length": 7, "log_freq": 0.65, "concrete": 6.07, "fas_mean": .051, "bas_mean": .011, "connectivity": 2.53, "veridical_recall": .650, "raw_freq":   4, "ortho_dist": 6.99, "colt_n":  0},
    "car":       {"false_recall": .350, "false_recognition": .420, "length": 3, "log_freq": 2.44, "concrete": 6.35, "fas_mean": .027, "bas_mean": .347, "connectivity": 1.73, "veridical_recall": .660, "raw_freq": 274, "ortho_dist": 2.08, "colt_n": 19},
    "carpet":    {"false_recall": .150, "false_recognition": .490, "length": 6, "log_freq": 1.13, "concrete": 5.68, "fas_mean": .039, "bas_mean": .037, "connectivity": 0.60, "veridical_recall": .610, "raw_freq":  13, "ortho_dist": 4.37, "colt_n":  0},
    "chair":     {"false_recall": .540, "false_recognition": .740, "length": 5, "log_freq": 1.82, "concrete": 6.12, "fas_mean": .063, "bas_mean": .303, "connectivity": 1.93, "veridical_recall": .640, "raw_freq":  66, "ortho_dist": 3.60, "colt_n":  3},
    "citizen":   {"false_recall": .100, "false_recognition": .600, "length": 7, "log_freq": 1.48, "concrete": 4.51, "fas_mean": .046, "bas_mean": .003, "connectivity": 2.27, "veridical_recall": .680, "raw_freq":  30, "ortho_dist": 6.26, "colt_n":  0},
    "city":      {"false_recall": .460, "false_recognition": .640, "length": 4, "log_freq": 2.59, "concrete": 5.41, "fas_mean": .042, "bas_mean": .185, "connectivity": 1.80, "veridical_recall": .650, "raw_freq": 393, "ortho_dist": 4.38, "colt_n":  3},
    "cold":      {"false_recall": .440, "false_recognition": .840, "length": 4, "log_freq": 2.23, "concrete": 4.67, "fas_mean": .043, "bas_mean": .353, "connectivity": 2.27, "veridical_recall": .610, "raw_freq": 171, "ortho_dist": 3.60, "colt_n": 14},
    "command":   {"false_recall": .080, "false_recognition": .550, "length": 7, "log_freq": 1.86, "concrete": None, "fas_mean": .035, "bas_mean": .009, "connectivity": 0.67, "veridical_recall": .580, "raw_freq":  72, "ortho_dist": 5.82, "colt_n":  1},
    "cottage":   {"false_recall": .080, "false_recognition": .380, "length": 7, "log_freq": 1.29, "concrete": 5.93, "fas_mean": .053, "bas_mean": .003, "connectivity": 1.00, "veridical_recall": .660, "raw_freq":  19, "ortho_dist": 6.52, "colt_n":  0},
    "cup":       {"false_recall": .450, "false_recognition": .820, "length": 3, "log_freq": 1.66, "concrete": 5.35, "fas_mean": .048, "bas_mean": .109, "connectivity": 1.53, "veridical_recall": .530, "raw_freq":  45, "ortho_dist": 2.66, "colt_n":  7},
    "doctor":    {"false_recall": .600, "false_recognition": .710, "length": 6, "log_freq": 2.00, "concrete": 5.75, "fas_mean": .053, "bas_mean": .245, "connectivity": 2.60, "veridical_recall": .570, "raw_freq": 100, "ortho_dist": 5.00, "colt_n":  0},
    "flag":      {"false_recall": .310, "false_recognition": .600, "length": 4, "log_freq": 1.22, "concrete": 6.20, "fas_mean": .043, "bas_mean": .109, "connectivity": 1.00, "veridical_recall": .630, "raw_freq":  16, "ortho_dist": 4.25, "colt_n":  5},
    "foot":      {"false_recall": .350, "false_recognition": .620, "length": 4, "log_freq": 1.85, "concrete": 3.46, "fas_mean": .049, "bas_mean": .177, "connectivity": 1.40, "veridical_recall": .640, "raw_freq":  70, "ortho_dist": 5.18, "colt_n": 10},
    "fruit":     {"false_recall": .200, "false_recognition": .450, "length": 5, "log_freq": 1.55, "concrete": 6.00, "fas_mean": .040, "bas_mean": .202, "connectivity": 1.67, "veridical_recall": .710, "raw_freq":  35, "ortho_dist": 4.58, "colt_n":  0},
    "girl":      {"false_recall": .320, "false_recognition": .580, "length": 4, "log_freq": 2.34, "concrete": 6.83, "fas_mean": .052, "bas_mean": .097, "connectivity": 1.27, "veridical_recall": .670, "raw_freq": 220, "ortho_dist": 4.15, "colt_n":  2},
    "health":    {"false_recall": .110, "false_recognition": .550, "length": 6, "log_freq": 2.02, "concrete": 3.54, "fas_mean": .028, "bas_mean": .020, "connectivity": 1.27, "veridical_recall": .620, "raw_freq": 105, "ortho_dist": 4.97, "colt_n":  2},
    "high":      {"false_recall": .260, "false_recognition": .720, "length": 4, "log_freq": 2.70, "concrete": 3.62, "fas_mean": .049, "bas_mean": .087, "connectivity": 1.47, "veridical_recall": .580, "raw_freq": 497, "ortho_dist": 5.18, "colt_n":  3},
    "justice":   {"false_recall": .300, "false_recognition": .760, "length": 7, "log_freq": 2.06, "concrete": 2.18, "fas_mean": .044, "bas_mean": .026, "connectivity": 1.40, "veridical_recall": .590, "raw_freq": 114, "ortho_dist": 5.67, "colt_n":  1},
    "king":      {"false_recall": .100, "false_recognition": .270, "length": 4, "log_freq": 1.95, "concrete": 5.54, "fas_mean": .059, "bas_mean": .230, "connectivity": 2.07, "veridical_recall": .650, "raw_freq":  88, "ortho_dist": 4.40, "colt_n":  9},
    "lamp":      {"false_recall": .140, "false_recognition": .630, "length": 4, "log_freq": 1.27, "concrete": 6.09, "fas_mean": .063, "bas_mean": .006, "connectivity": 0.73, "veridical_recall": .610, "raw_freq":  18, "ortho_dist": 3.51, "colt_n":  9},
    "lion":      {"false_recall": .230, "false_recognition": .330, "length": 4, "log_freq": 1.24, "concrete": 6.14, "fas_mean": .040, "bas_mean": .136, "connectivity": 0.63, "veridical_recall": .630, "raw_freq":  17, "ortho_dist": 4.00, "colt_n":  6},
    "long":      {"false_recall": .030, "false_recognition": .340, "length": 4, "log_freq": 2.88, "concrete": 3.68, "fas_mean": .045, "bas_mean": .039, "connectivity": 0.47, "veridical_recall": .600, "raw_freq": 755, "ortho_dist": 3.36, "colt_n": 10},
    "man":       {"false_recall": .240, "false_recognition": .610, "length": 3, "log_freq": 3.08, "concrete": 6.14, "fas_mean": .047, "bas_mean": .115, "connectivity": 0.80, "veridical_recall": .560, "raw_freq":1207, "ortho_dist": 2.13, "colt_n": 33},
    "mountain":  {"false_recall": .420, "false_recognition": .690, "length": 8, "log_freq": 1.53, "concrete": 6.25, "fas_mean": .037, "bas_mean": .154, "connectivity": 0.87, "veridical_recall": .600, "raw_freq":  33, "ortho_dist": 5.62, "colt_n":  1},
    "music":     {"false_recall": .340, "false_recognition": .690, "length": 5, "log_freq": 2.34, "concrete": 5.15, "fas_mean": .020, "bas_mean": .227, "connectivity": 1.60, "veridical_recall": .590, "raw_freq": 216, "ortho_dist": 3.71, "colt_n":  0},
    "mutton":    {"false_recall": .010, "false_recognition": .110, "length": 6, "log_freq": 0.93, "concrete": 5.34, "fas_mean": .014, "bas_mean": .002, "connectivity": 2.00, "veridical_recall": .650, "raw_freq":   6, "ortho_dist": 6.37, "colt_n":  4},
    "needle":    {"false_recall": .520, "false_recognition": .680, "length": 6, "log_freq": 1.19, "concrete": 5.79, "fas_mean": .063, "bas_mean": .203, "connectivity": 1.93, "veridical_recall": .600, "raw_freq":  15, "ortho_dist": 6.29, "colt_n":  0},
    "pen":       {"false_recall": .350, "false_recognition": .570, "length": 3, "log_freq": 1.27, "concrete": 5.59, "fas_mean": .044, "bas_mean": .135, "connectivity": 1.33, "veridical_recall": .630, "raw_freq":  18, "ortho_dist": 2.15, "colt_n": 19},
    "river":     {"false_recall": .420, "false_recognition": .670, "length": 5, "log_freq": 2.22, "concrete": 5.83, "fas_mean": .036, "bas_mean": .147, "connectivity": 2.47, "veridical_recall": .640, "raw_freq": 165, "ortho_dist": 3.68, "colt_n":  6},
    "rough":     {"false_recall": .530, "false_recognition": .830, "length": 5, "log_freq": 1.62, "concrete": 4.48, "fas_mean": .033, "bas_mean": .122, "connectivity": 1.27, "veridical_recall": .560, "raw_freq":  44, "ortho_dist": 4.74, "colt_n":  8},
    "rubber":    {"false_recall": .320, "false_recognition": .670, "length": 6, "log_freq": 1.19, "concrete": 6.04, "fas_mean": .017, "bas_mean": .033, "connectivity": 0.67, "veridical_recall": .530, "raw_freq":  15, "ortho_dist": 5.89, "colt_n":  2},
    "shirt":     {"false_recall": .270, "false_recognition": .540, "length": 5, "log_freq": 1.44, "concrete": 6.05, "fas_mean": .044, "bas_mean": .186, "connectivity": 1.40, "veridical_recall": .640, "raw_freq":  27, "ortho_dist": 3.84, "colt_n":  4},
    "sleep":     {"false_recall": .610, "false_recognition": .800, "length": 5, "log_freq": 1.82, "concrete": 4.74, "fas_mean": .047, "bas_mean": .431, "connectivity": 1.80, "veridical_recall": .610, "raw_freq":  65, "ortho_dist": 5.15, "colt_n":  5},
    "slow":      {"false_recall": .420, "false_recognition": .690, "length": 4, "log_freq": 1.78, "concrete": 2.89, "fas_mean": .047, "bas_mean": .172, "connectivity": 1.13, "veridical_recall": .530, "raw_freq":  60, "ortho_dist": 3.38, "colt_n": 10},
    "smell":     {"false_recall": .600, "false_recognition": .840, "length": 5, "log_freq": 1.54, "concrete": 4.40, "fas_mean": .015, "bas_mean": .290, "connectivity": 1.47, "veridical_recall": .580, "raw_freq":  34, "ortho_dist": 4.32, "colt_n":  5},
    "smoke":     {"false_recall": .540, "false_recognition": .730, "length": 5, "log_freq": 1.62, "concrete": 5.16, "fas_mean": .042, "bas_mean": .167, "connectivity": 1.80, "veridical_recall": .640, "raw_freq":  41, "ortho_dist": 5.05, "colt_n":  2},
    "soft":      {"false_recall": .460, "false_recognition": .810, "length": 4, "log_freq": 1.79, "concrete": 4.10, "fas_mean": .041, "bas_mean": .179, "connectivity": 0.80, "veridical_recall": .590, "raw_freq":  61, "ortho_dist": 3.70, "colt_n":  4},
    "spider":    {"false_recall": .370, "false_recognition": .580, "length": 6, "log_freq": 0.40, "concrete": 5.95, "fas_mean": .044, "bas_mean": .159, "connectivity": 1.33, "veridical_recall": .620, "raw_freq":   2, "ortho_dist": 5.78, "colt_n":  1},
    "stove":     {"false_recall": .180, "false_recognition": .700, "length": 5, "log_freq": 1.19, "concrete": 5.75, "fas_mean": .058, "bas_mean": .035, "connectivity": 2.00, "veridical_recall": .600, "raw_freq":  15, "ortho_dist": 4.29, "colt_n":  7},
    "sweet":     {"false_recall": .540, "false_recognition": .780, "length": 5, "log_freq": 1.85, "concrete": 4.53, "fas_mean": .054, "bas_mean": .172, "connectivity": 2.07, "veridical_recall": .630, "raw_freq":  70, "ortho_dist": 5.47, "colt_n":  6},
    "swift":     {"false_recall": .140, "false_recognition": .350, "length": 5, "log_freq": 1.51, "concrete": 3.31, "fas_mean": .058, "bas_mean": .006, "connectivity": 1.73, "veridical_recall": .530, "raw_freq":  32, "ortho_dist": 5.43, "colt_n":  1},
    "thief":     {"false_recall": .230, "false_recognition": .700, "length": 5, "log_freq": 0.93, "concrete": 4.83, "fas_mean": .056, "bas_mean": .100, "connectivity": 4.27, "veridical_recall": .610, "raw_freq":   8, "ortho_dist": 5.34, "colt_n":  1},
    "trash":     {"false_recall": .490, "false_recognition": .780, "length": 5, "log_freq": 0.40, "concrete": 5.76, "fas_mean": .055, "bas_mean": .140, "connectivity": 2.60, "veridical_recall": .540, "raw_freq":   2, "ortho_dist": 4.00, "colt_n":  2},
    "trouble":   {"false_recall": .080, "false_recognition": .540, "length": 7, "log_freq": 2.13, "concrete": 2.25, "fas_mean": .026, "bas_mean": .010, "connectivity": 1.07, "veridical_recall": .580, "raw_freq": 134, "ortho_dist": 5.53, "colt_n":  1},
    "whiskey":   {"false_recall": .030, "false_recognition": .530, "length": 7, "log_freq": 1.24, "concrete": 6.00, "fas_mean": .042, "bas_mean": .022, "connectivity": 4.93, "veridical_recall": .720, "raw_freq":  17, "ortho_dist": 7.70, "colt_n":  1},
    "whistle":   {"false_recall": .090, "false_recognition": .190, "length": 7, "log_freq": 0.65, "concrete": 5.58, "fas_mean": .038, "bas_mean": .005, "connectivity": 1.13, "veridical_recall": .600, "raw_freq":   4, "ortho_dist": 6.52, "colt_n":  0},
    "window":    {"false_recall": .650, "false_recognition": .840, "length": 6, "log_freq": 2.08, "concrete": 6.27, "fas_mean": .058, "bas_mean": .184, "connectivity": 0.67, "veridical_recall": .630, "raw_freq": 119, "ortho_dist": 5.73, "colt_n":  1},
    "wish":      {"false_recall": .290, "false_recognition": .800, "length": 4, "log_freq": 2.04, "concrete": 2.66, "fas_mean": .053, "bas_mean": .012, "connectivity": 0.87, "veridical_recall": .580, "raw_freq": 110, "ortho_dist": 4.15, "colt_n":  6},
}


# ---------------------------------------------------------------------------
# Appendix A: per-list associates with BAS and FAS values.
#
# Each entry is (word, bas, fas). Asterisked source values are stored
# without the asterisk; we lose the "authors' supplemental norming"
# annotation but keep all numeric values. Multi-word entries from the
# source ("New York", "Air Force", "United States") use underscores so
# ConceptNet Numberbatch lookups resolve.
# ---------------------------------------------------------------------------

ASSOCIATES: dict[str, list[tuple[str, float, float]]] = {
    "anger": [
        ("mad", .393, .412), ("fear", .020, .059), ("hate", .028, .109),
        ("rage", .541, .042), ("temper", .182, .000), ("fury", .306, .000),
        ("ire", .179, .000), ("wrath", .128, .000), ("happy", .000, .042),
        ("fight", .034, .000), ("hatred", .070, .000), ("mean", .090, .000),
        ("calm", .000, .000), ("emotion", .000, .000), ("enrage", .378, .000),
    ],
    "army": [
        ("navy", .543, .500), ("soldier", .287, .027), ("united_states", .000, .000),
        ("rifle", .000, .000), ("air_force", .133, .014), ("draft", .122, .000),
        ("military", .266, .027), ("marines", .283, .047), ("march", .041, .000),
        ("infantry", .284, .000), ("captain", .014, .000), ("war", .000, .041),
        ("uniform", .048, .000), ("pilot", .000, .000), ("combat", .000, .000),
    ],
    "beautiful": [
        ("ugly", .028, .229), ("pretty", .095, .389), ("girls", .033, .049),
        ("woman", .016, .014), ("homely", .000, .000), ("lovely", .182, .000),
        ("nice", .000, .000), ("picture", .000, .000), ("lady", .010, .000),
        ("mountain", .000, .000), ("snow", .000, .000), ("scene", .024, .000),
        ("music", .000, .000), ("day", .000, .000), ("gorgeous", .184, .056),
    ],
    "bitter": [
        ("sweet", .020, .435), ("sour", .115, .254), ("taste", .024, .065),
        ("chocolate", .000, .000), ("rice", .000, .000), ("cold", .000, .101),
        ("lemon", .000, .022), ("angry", .000, .000), ("hard", .000, .000),
        ("mad", .000, .000), ("acid", .000, .014), ("almonds", .000, .000),
        ("herbs", .000, .000), ("grape", .000, .000), ("fruit", .000, .000),
    ],
    "black": [
        ("white", .655, .557), ("dark", .111, .100), ("cat", .000, .043),
        ("charred", .023, .000), ("night", .000, .021), ("funeral", .034, .000),
        ("color", .074, .050), ("grief", .000, .000), ("blue", .028, .000),
        ("death", .016, .014), ("ink", .020, .000), ("bottom", .000, .000),
        ("coal", .288, .000), ("brown", .338, .000), ("gray", .365, .000),
    ],
    "bread": [
        ("butter", .364, .487), ("food", .000, .045), ("eat", .000, .026),
        ("sandwich", .067, .026), ("rye", .791, .000), ("jam", .054, .000),
        ("milk", .012, .000), ("flour", .142, .000), ("jelly", .053, .019),
        ("dough", .310, .058), ("crust", .243, .000), ("slice", .048, .019),
        ("wine", .000, .000), ("loaf", .552, .051), ("toast", .364, .000),
    ],
    "butterfly": [
        ("moth", .267, .109), ("insect", .000, .079), ("wing", .000, .030),
        ("bird", .000, .042), ("fly", .000, .091), ("yellow", .000, .018),
        ("net", .000, .030), ("flower", .000, .036), ("bug", .000, .012),
        ("cocoon", .412, .024), ("summer", .000, .000), ("color", .000, .030),
        ("bee", .000, .000), ("stomach", .000, .000), ("worm", .000, .000),
    ],
    "cabbage": [
        ("head", .000, .022), ("lettuce", .021, .281), ("vegetable", .000, .137),
        ("food", .000, .022), ("salad", .000, .022), ("green", .000, .079),
        ("garden", .000, .000), ("leaf", .000, .029), ("sauerkraut", .042, .000),
        ("smell", .000, .000), ("slaw", .041, .043), ("patch", .066, .115),
        ("plant", .000, .000), ("carrots", .000, .000), ("soup", .000, .014),
    ],
    "car": [
        ("truck", .264, .111), ("bus", .252, .022), ("train", .058, .011),
        ("automobile", .709, .133), ("vehicle", .740, .000), ("drive", .480, .122),
        ("jeep", .240, .000), ("ford", .331, .000), ("race", .043, .011),
        ("keys", .360, .000), ("garage", .519, .000), ("highway", .115, .000),
        ("sedan", .510, .000), ("van", .448, .000), ("taxi", .129, .000),
    ],
    "carpet": [
        ("rug", .468, .248), ("floor", .074, .159), ("soft", .000, .062),
        ("red", .000, .048), ("sweeper", .000, .000), ("tack", .000, .000),
        ("walk", .000, .000), ("bag", .000, .021), ("room", .000, .000),
        ("blue", .000, .021), ("chair", .000, .000), ("thick", .000, .000),
        ("deep", .000, .000), ("magic", .014, .028), ("wool", .000, .000),
    ],
    "chair": [
        ("table", .756, .314), ("sit", .183, .212), ("legs", .000, .013),
        ("seat", .543, .109), ("couch", .288, .109), ("desk", .290, .019),
        ("recliner", .547, .000), ("sofa", .132, .077), ("wood", .012, .013),
        ("cushion", .086, .019), ("swivel", .593, .000), ("stool", .320, .032),
        ("sitting", .096, .000), ("rocking", .593, .019), ("bench", .109, .013),
    ],
    "citizen": [
        ("united_states", .000, .191), ("man", .000, .000), ("person", .000, .191),
        ("american", .034, .086), ("country", .000, .059), ("alien", .000, .079),
        ("people", .000, .000), ("vote", .000, .013), ("me", .000, .000),
        ("patriot", .013, .020), ("flag", .000, .000), ("foreigner", .000, .000),
        ("france", .000, .000), ("immigrant", .000, .013), ("member", .000, .033),
    ],
    "city": [
        ("town", .529, .307), ("crowded", .000, .010), ("state", .117, .132),
        ("capital", .095, .000), ("streets", .054, .046), ("subway", .000, .000),
        ("country", .068, .020), ("new_york", .383, .066), ("village", .020, .000),
        ("metropolis", .536, .000), ("big", .000, .025), ("chicago", .152, .000),
        ("suburb", .265, .010), ("county", .195, .010), ("urban", .358, .000),
    ],
    "cold": [
        ("hot", .676, .413), ("snow", .199, .033), ("warm", .364, .033),
        ("winter", .277, .022), ("ice", .364, .098), ("wet", .108, .011),
        ("frigid", .570, .000), ("chilly", .395, .000), ("heat", .169, .000),
        ("weather", .032, .011), ("freeze", .461, .011), ("air", .000, .000),
        ("shiver", .669, .011), ("arctic", .642, .000), ("frost", .370, .000),
    ],
    "command": [
        ("order", .000, .288), ("army", .000, .034), ("obey", .140, .062),
        ("officer", .000, .027), ("performance", .000, .014), ("do", .000, .014),
        ("tell", .000, .055), ("general", .000, .027), ("shout", .000, .000),
        ("halt", .000, .000), ("voice", .000, .000), ("soldier", .000, .000),
        ("harsh", .000, .000), ("attention", .000, .000), ("sharp", .000, .000),
    ],
    "cottage": [
        ("house", .000, .381), ("lake", .000, .063), ("cheese", .000, .206),
        ("home", .000, .040), ("white", .000, .000), ("cabin", .020, .016),
        ("small", .000, .016), ("door", .000, .000), ("fence", .000, .000),
        ("vines", .000, .000), ("woods", .000, .040), ("ivy", .000, .000),
        ("roses", .000, .000), ("cozy", .000, .000), ("hut", .030, .032),
    ],
    "cup": [
        ("mug", .268, .025), ("saucer", .527, .418), ("tea", .054, .056),
        ("measuring", .385, .000), ("coaster", .096, .000), ("lid", .000, .000),
        ("handle", .014, .000), ("coffee", .051, .105), ("straw", .029, .000),
        ("goblet", .118, .000), ("soup", .000, .062), ("stein", .014, .000),
        ("drink", .011, .049), ("plastic", .075, .000), ("sip", .000, .000),
    ],
    "doctor": [
        ("nurse", .547, .379), ("sick", .031, .051), ("lawyer", .149, .101),
        ("medicine", .152, .066), ("health", .049, .020), ("hospital", .027, .015),
        ("dentist", .214, .020), ("physician", .804, .040), ("ill", .000, .025),
        ("patient", .365, .025), ("office", .014, .010), ("stethoscope", .520, .000),
        ("surgeon", .479, .040), ("clinic", .300, .000), ("cure", .028, .010),
    ],
    "flag": [
        ("banner", .687, .000), ("american", .200, .269), ("symbol", .014, .021),
        ("stars", .000, .048), ("anthem", .062, .000), ("stripes", .177, .014),
        ("pole", .157, .193), ("wave", .000, .103), ("raised", .000, .000),
        ("national", .027, .000), ("checkered", .247, .000), ("emblem", .048, .000),
        ("sign", .000, .000), ("freedom", .021, .000), ("pendant", .000, .000),
    ],
    "foot": [
        ("shoe", .321, .337), ("hand", .158, .122), ("toe", .605, .235),
        ("kick", .039, .000), ("sandals", .209, .000), ("soccer", .000, .000),
        ("yard", .126, .000), ("walk", .016, .020), ("ankle", .364, .000),
        ("arm", .000, .000), ("boot", .142, .000), ("inch", .473, .020),
        ("sock", .172, .000), ("knee", .032, .000), ("mouth", .000, .000),
    ],
    "fruit": [
        ("apple", .154, .223), ("vegetable", .220, .082), ("orange", .194, .174),
        ("kiwi", .709, .000), ("citrus", .426, .000), ("ripe", .151, .000),
        ("pear", .347, .000), ("banana", .215, .065), ("berry", .298, .000),
        ("cherry", .168, .000), ("basket", .084, .011), ("juice", .035, .027),
        ("salad", .000, .000), ("bowl", .028, .000), ("cocktail", .000, .011),
    ],
    "girl": [
        ("boy", .701, .738), ("dolls", .199, .000), ("female", .098, .013),
        ("young", .000, .000), ("dress", .063, .000), ("pretty", .149, .027),
        ("hair", .000, .000), ("niece", .026, .000), ("dance", .000, .000),
        ("beautiful", .049, .000), ("cute", .035, .000), ("date", .056, .000),
        ("aunt", .000, .000), ("daughter", .042, .000), ("sister", .041, .000),
    ],
    "health": [
        ("sickness", .220, .140), ("good", .000, .133), ("happiness", .000, .014),
        ("wealth", .022, .021), ("ill", .014, .000), ("doctor", .020, .049),
        ("service", .000, .000), ("strong", .000, .000), ("hospital", .000, .014),
        ("disease", .000, .021), ("body", .027, .028), ("vigor", .000, .000),
        ("center", .000, .000), ("pain", .000, .000), ("robust", .000, .000),
    ],
    "high": [
        ("low", .777, .655), ("clouds", .000, .000), ("up", .041, .034),
        ("tall", .000, .020), ("tower", .079, .000), ("jump", .072, .000),
        ("above", .057, .000), ("building", .000, .000), ("noon", .033, .000),
        ("cliff", .028, .000), ("sky", .017, .014), ("over", .000, .000),
        ("airplane", .000, .014), ("dive", .028, .000), ("elevate", .174, .000),
    ],
    "justice": [
        ("peace", .000, .151), ("law", .031, .171), ("courts", .090, .158),
        ("judge", .014, .096), ("right", .000, .021), ("liberty", .113, .021),
        ("government", .000, .000), ("jury", .000, .000), ("truth", .082, .000),
        ("blind", .000, .000), ("fair", .030, .027), ("supreme", .021, .000),
        ("crime", .014, .021), ("department", .000, .000), ("trial", .000, .000),
    ],
    "king": [
        ("queen", .730, .772), ("england", .000, .000), ("crown", .471, .016),
        ("prince", .134, .016), ("george", .020, .000), ("dictator", .023, .000),
        ("palace", .159, .000), ("throne", .759, .000), ("chess", .092, .000),
        ("rule", .014, .031), ("subjects", .000, .000), ("monarch", .317, .039),
        ("royal", .315, .016), ("leader", .034, .000), ("reign", .383, .000),
    ],
    "lamp": [
        ("light", .020, .769), ("shade", .028, .058), ("table", .000, .019),
        ("bulb", .014, .045), ("post", .000, .026), ("black", .000, .000),
        ("cord", .000, .000), ("desk", .034, .019), ("bright", .000, .000),
        ("lighter", .000, .000), ("read", .000, .000), ("on", .000, .000),
        ("bed", .000, .000), ("burn", .000, .013), ("stand", .000, .000),
    ],
    "lion": [
        ("tiger", .308, .362), ("circus", .011, .000), ("jungle", .034, .000),
        ("tamer", .489, .021), ("den", .097, .021), ("cub", .063, .074),
        ("africa", .014, .021), ("mane", .200, .021), ("cage", .035, .000),
        ("feline", .000, .000), ("roar", .614, .032), ("fierce", .112, .021),
        ("bears", .034, .021), ("hunt", .000, .000), ("pride", .029, .000),
    ],
    "long": [
        ("short", .222, .536), ("fellow", .000, .000), ("narrow", .021, .000),
        ("john", .000, .017), ("time", .000, .034), ("far", .041, .000),
        ("hair", .103, .078), ("island", .000, .000), ("road", .020, .000),
        ("thin", .000, .011), ("underwear", .000, .000), ("distance", .150, .000),
        ("line", .031, .000), ("low", .000, .000), ("rope", .000, .000),
    ],
    "man": [
        ("woman", .595, .660), ("husband", .018, .000), ("uncle", .000, .000),
        ("lady", .070, .013), ("mouse", .371, .000), ("male", .131, .000),
        ("father", .048, .000), ("strong", .020, .013), ("friend", .000, .000),
        ("beard", .055, .000), ("person", .122, .000), ("handsome", .144, .000),
        ("muscle", .048, .013), ("suit", .074, .000), ("old", .034, .000),
    ],
    "mountain": [
        ("hill", .428, .265), ("valley", .195, .020), ("climb", .291, .092),
        ("summit", .108, .000), ("top", .000, .041), ("molehill", .256, .031),
        ("peak", .248, .020), ("plain", .000, .000), ("glacier", .028, .000),
        ("goat", .033, .000), ("bike", .000, .000), ("climber", .603, .031),
        ("range", .000, .051), ("steep", .061, .000), ("ski", .034, .000),
    ],
    "music": [
        ("note", .132, .068), ("sound", .205, .020), ("piano", .230, .020),
        ("sing", .033, .088), ("radio", .270, .041), ("band", .432, .020),
        ("melody", .243, .000), ("horn", .014, .000), ("concert", .395, .000),
        ("instrument", .148, .000), ("symphony", .329, .000), ("jazz", .367, .000),
        ("orchestra", .309, .000), ("art", .020, .020), ("rhythm", .277, .000),
    ],
    "mutton": [
        ("lamb", .024, .133), ("sheep", .000, .027), ("meat", .000, .047),
        ("chops", .000, .000), ("beef", .000, .000), ("veal", .000, .000),
        ("collar", .000, .000), ("leg", .000, .000), ("eat", .000, .000),
        ("fat", .000, .000), ("coat", .000, .000), ("stew", .000, .000),
        ("fur", .000, .000), ("pork", .000, .000), ("steak", .000, .000),
    ],
    "needle": [
        ("thread", .758, .424), ("pin", .289, .212), ("eye", .000, .000),
        ("sewing", .181, .224), ("sharp", .030, .024), ("point", .024, .024),
        ("prick", .108, .012), ("thimble", .218, .000), ("haystack", .418, .030),
        ("thorn", .028, .000), ("hurt", .000, .000), ("injection", .331, .000),
        ("syringe", .520, .000), ("cloth", .000, .000), ("knitting", .135, .000),
    ],
    "pen": [
        ("pencil", .476, .594), ("write", .128, .065), ("fountain", .071, .000),
        ("leak", .000, .000), ("quill", .635, .000), ("felt", .047, .000),
        ("bic", .372, .000), ("scribble", .020, .000), ("cross", .013, .000),
        ("crayon", .000, .000), ("tip", .000, .000), ("marker", .257, .000),
        ("red", .000, .000), ("cap", .000, .000), ("letter", .000, .000),
    ],
    "river": [
        ("water", .000, .071), ("stream", .321, .118), ("lake", .142, .118),
        ("mississippi", .654, .031), ("boat", .000, .055), ("tide", .000, .000),
        ("swim", .000, .016), ("flow", .283, .063), ("run", .000, .016),
        ("barge", .047, .000), ("creek", .397, .000), ("brook", .161, .016),
        ("fish", .000, .016), ("bridge", .197, .000), ("winding", .000, .016),
    ],
    "rough": [
        ("smooth", .416, .352), ("bumpy", .150, .028), ("road", .000, .000),
        ("tough", .192, .048), ("sandpaper", .429, .000), ("jagged", .128, .000),
        ("ready", .000, .000), ("coarse", .291, .014), ("uneven", .019, .000),
        ("riders", .027, .000), ("rugged", .174, .014), ("sand", .000, .000),
        ("boards", .000, .000), ("ground", .000, .000), ("gravel", .000, .000),
    ],
    "rubber": [
        ("elastic", .035, .000), ("bounce", .018, .000), ("gloves", .033, .041),
        ("tire", .062, .095), ("ball", .000, .041), ("eraser", .026, .000),
        ("springy", .000, .000), ("foam", .116, .000), ("galoshes", .063, .000),
        ("soles", .000, .041), ("latex", .107, .014), ("glue", .000, .000),
        ("flexible", .041, .000), ("resilient", .000, .027), ("stretch", .000, .000),
    ],
    "shirt": [
        ("blouse", .647, .135), ("sleeves", .347, .038), ("pants", .185, .269),
        ("tie", .074, .103), ("button", .240, .064), ("shorts", .252, .013),
        ("iron", .010, .000), ("polo", .177, .000), ("collar", .342, .032),
        ("vest", .143, .000), ("pocket", .058, .000), ("jersey", .174, .000),
        ("belt", .000, .000), ("linen", .000, .000), ("cuffs", .143, .000),
    ],
    "sleep": [
        ("bed", .638, .092), ("rest", .475, .163), ("awake", .618, .143),
        ("tired", .493, .092), ("dream", .247, .194), ("wake", .304, .000),
        ("snooze", .520, .020), ("blanket", .024, .000), ("doze", .682, .000),
        ("slumber", .514, .000), ("snore", .439, .000), ("nap", .730, .000),
        ("peace", .000, .000), ("yawn", .235, .000), ("drowsy", .551, .000),
    ],
    "slow": [
        ("fast", .598, .527), ("lethargic", .142, .000), ("stop", .000, .034),
        ("listless", .000, .000), ("snail", .486, .020), ("cautious", .027, .000),
        ("delay", .059, .000), ("traffic", .020, .000), ("turtle", .372, .115),
        ("hesitant", .034, .000), ("speed", .061, .014), ("quick", .272, .000),
        ("sluggish", .340, .000), ("wait", .000, .000), ("molasses", .170, .000),
    ],
    "smell": [
        ("nose", .108, .116), ("breathe", .000, .000), ("sniff", .442, .043),
        ("aroma", .678, .000), ("hear", .000, .000), ("see", .000, .000),
        ("nostril", .000, .000), ("whiff", .577, .000), ("scent", .625, .029),
        ("reek", .510, .000), ("stench", .562, .000), ("fragrance", .389, .000),
        ("perfume", .393, .036), ("salts", .028, .000), ("rose", .034, .000),
    ],
    "smoke": [
        ("cigarette", .449, .323), ("puff", .240, .000), ("blaze", .000, .000),
        ("billows", .061, .000), ("pollution", .068, .000), ("ashes", .052, .000),
        ("cigar", .507, .000), ("chimney", .240, .000), ("fire", .018, .291),
        ("tobacco", .338, .000), ("stink", .000, .000), ("pipe", .419, .016),
        ("lungs", .119, .000), ("flames", .000, .000), ("stain", .000, .000),
    ],
    "soft": [
        ("hard", .564, .509), ("light", .000, .012), ("pillow", .236, .018),
        ("plush", .178, .000), ("loud", .333, .000), ("cotton", .166, .018),
        ("fur", .061, .000), ("touch", .061, .012), ("fluffy", .266, .000),
        ("feather", .045, .024), ("furry", .061, .000), ("downy", .221, .000),
        ("kitten", .033, .000), ("skin", .161, .018), ("tender", .297, .000),
    ],
    "spider": [
        ("web", .845, .246), ("insect", .000, .127), ("bug", .040, .127),
        ("fright", .000, .000), ("fly", .000, .016), ("arachnid", .704, .079),
        ("crawl", .000, .024), ("tarantula", .744, .000), ("poison", .000, .000),
        ("bite", .000, .000), ("creepy", .058, .040), ("animal", .000, .000),
        ("ugly", .000, .000), ("feelers", .000, .000), ("small", .000, .000),
    ],
    "stove": [
        ("hot", .000, .285), ("heat", .000, .030), ("pipe", .000, .018),
        ("cook", .000, .212), ("warm", .000, .012), ("fire", .000, .000),
        ("oven", .224, .279), ("wood", .030, .000), ("kitchen", .056, .018),
        ("lid", .000, .000), ("coal", .000, .000), ("gas", .026, .000),
        ("iron", .000, .000), ("range", .149, .012), ("furnace", .041, .000),
    ],
    "sweet": [
        ("sour", .405, .372), ("candy", .336, .162), ("sugar", .433, .061),
        ("bitter", .435, .020), ("good", .000, .014), ("taste", .071, .014),
        ("tooth", .000, .027), ("nice", .095, .095), ("honey", .451, .000),
        ("soda", .000, .000), ("chocolate", .101, .041), ("heart", .000, .000),
        ("cake", .027, .000), ("tart", .223, .000), ("pie", .000, .000),
    ],
    "swift": [
        ("fast", .016, .606), ("slow", .000, .176), ("river", .000, .000),
        ("jonathan", .054, .000), ("current", .000, .000), ("rapid", .021, .012),
        ("stream", .000, .000), ("water", .000, .000), ("quick", .000, .048),
        ("gulliver", .000, .000), ("run", .000, .024), ("sure", .000, .000),
        ("deer", .000, .000), ("car", .000, .000), ("author", .000, .000),
    ],
    "thief": [
        ("steal", .089, .000), ("robber", .361, .224), ("crook", .459, .091),
        ("burglar", .257, .085), ("money", .000, .012), ("cop", .000, .000),
        ("bad", .000, .012), ("rob", .074, .000), ("jail", .013, .000),
        ("gun", .000, .000), ("villain", .000, .000), ("crime", .028, .012),
        ("bank", .000, .000), ("bandit", .167, .000), ("criminal", .051, .012),
    ],
    "trash": [
        ("garbage", .456, .526), ("waste", .067, .026), ("can", .014, .212),
        ("refuse", .017, .000), ("sewage", .053, .000), ("bag", .000, .026),
        ("junk", .126, .013), ("rubbish", .397, .013), ("sweep", .000, .000),
        ("scraps", .048, .000), ("pile", .049, .000), ("dump", .218, .013),
        ("landfill", .186, .000), ("debris", .266, .000), ("litter", .209, .000),
    ],
    "trouble": [
        ("bad", .000, .123), ("shooter", .000, .048), ("worry", .000, .000),
        ("danger", .048, .027), ("sorrow", .000, .000), ("fear", .000, .000),
        ("school", .000, .000), ("problem", .027, .089), ("police", .000, .041),
        ("fight", .000, .027), ("sad", .000, .000), ("difficulty", .031, .000),
        ("help", .049, .041), ("maker", .000, .000), ("jail", .000, .000),
    ],
    "whiskey": [
        ("drink", .000, .081), ("drunk", .000, .121), ("beer", .000, .051),
        ("liquor", .000, .081), ("gin", .021, .000), ("bottles", .000, .020),
        ("alcohol", .000, .051), ("rye", .000, .000), ("glass", .000, .020),
        ("wine", .000, .030), ("rum", .035, .081), ("bourbon", .144, .051),
        ("evil", .000, .000), ("bar", .000, .000), ("scotch", .135, .040),
    ],
    "whistle": [
        ("stop", .000, .032), ("train", .045, .016), ("noise", .000, .071),
        ("sing", .000, .016), ("blow", .034, .175), ("tune", .000, .032),
        ("sound", .000, .000), ("dog", .000, .016), ("song", .000, .127),
        ("shrill", .000, .000), ("boy", .000, .000), ("lips", .000, .024),
        ("wolf", .000, .000), ("call", .000, .016), ("loud", .000, .048),
    ],
    "window": [
        ("door", .156, .147), ("glass", .144, .256), ("pane", .833, .179),
        ("shade", .021, .058), ("ledge", .152, .013), ("sill", .682, .128),
        ("house", .000, .000), ("open", .014, .019), ("curtain", .189, .038),
        ("frame", .014, .013), ("view", .048, .026), ("breeze", .000, .000),
        ("sash", .000, .000), ("screen", .027, .000), ("shutter", .480, .000),
    ],
    "wish": [
        ("want", .028, .071), ("dream", .014, .165), ("desire", .027, .039),
        ("hope", .068, .213), ("well", .000, .039), ("think", .045, .031),
        ("star", .000, .126), ("bone", .000, .079), ("ring", .000, .000),
        ("wash", .000, .016), ("thought", .000, .000), ("get", .000, .000),
        ("true", .000, .016), ("for", .000, .000), ("money", .000, .000),
    ],
}


def main() -> None:
    out_path = Path(__file__).parent / "stadler1999_55lists.json"
    payload = []
    for lure in sorted(PER_LIST_DATA.keys()):
        assoc_rows = ASSOCIATES[lure]
        if len(assoc_rows) != 15:
            raise ValueError(
                f"list '{lure}' has {len(assoc_rows)} associates, expected 15"
            )
        per_list = PER_LIST_DATA[lure]
        payload.append({
            "critical_lure": lure,
            "associates": [w for (w, _b, _f) in assoc_rows],
            "associate_bas": {w: b for (w, b, _f) in assoc_rows},
            "associate_fas": {w: f for (w, _b, f) in assoc_rows},
            "published_fa_rate": per_list["false_recognition"],
            "published_recall_rate": per_list["false_recall"],
            "published_bas": per_list["bas_mean"],
            "published_fas": per_list["fas_mean"],
            "connectivity": per_list["connectivity"],
            "veridical_recall": per_list["veridical_recall"],
            "length": per_list["length"],
            "log_freq": per_list["log_freq"],
            "concrete": per_list["concrete"],
            "raw_freq": per_list["raw_freq"],
            "ortho_dist": per_list["ortho_dist"],
            "colt_n": per_list["colt_n"],
            "notes": "Roediger, Watson, McDermott & Gallo (2001) Appendices A+B.",
        })
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {len(payload)} lists to {out_path}")


if __name__ == "__main__":
    main()
