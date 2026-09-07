"""Resolve speakers labelled party="Cabinet" to their actual parliamentary group.

Problem
-------
In the source corpus, members of the federal government carry ``Party="Cabinet"``
instead of their own party. 19,269 chunks (8.8% of the corpus) are therefore
invisible to a party filter: a query for ``{"party": "CDU/CSU"}`` silently
excludes Merkel, Altmaier, Scheuer, Klöckner and the rest of the front bench.

Resolution strategy
-------------------
1. ``corpus``  - 34 of 69 Cabinet speakers also appear as MdB elsewhere in the
   corpus with a real party. Those are derived from the data itself, with no
   ambiguity (verified: no speaker maps to more than one party).
2. ``manual``  - the remaining 35 are resolved from public record. Federal
   ministers and parliamentary state secretaries of the 19th legislative period.
3. ``state``   - seven speakers are *Land* ministers appearing via the Bundesrat,
   not the federal cabinet. They are resolved to their real party but tagged
   ``level="state"`` so party queries about Bundestag debates can exclude them.

For the planned expansion to all legislative periods, replace step 2 with the
Bundestag Stammdaten (MdB master data XML), which carries party affiliation for
every member and needs no hand-maintained table.
"""
from __future__ import annotations

# name -> (parliamentary group as used in the corpus, actual party, level)
MANUAL: dict[str, tuple[str, str, str]] = {
    # --- CDU/CSU, federal government (19. WP) ---
    "Andreas Scheuer":            ("CDU/CSU", "CSU", "federal"),
    "Anja Karliczek":             ("CDU/CSU", "CDU", "federal"),
    "Annegret Kramp-Karrenbauer": ("CDU/CSU", "CDU", "federal"),
    "Dorothee Bär":               ("CDU/CSU", "CSU", "federal"),
    "Enak Ferlemann":             ("CDU/CSU", "CDU", "federal"),
    "Gerd Müller":                ("CDU/CSU", "CSU", "federal"),
    "Hans-Joachim Fuchtel":       ("CDU/CSU", "CDU", "federal"),
    "Helge Braun":                ("CDU/CSU", "CDU", "federal"),
    "Horst Seehofer":             ("CDU/CSU", "CSU", "federal"),
    "Julia Klöckner":             ("CDU/CSU", "CDU", "federal"),
    "Maria Böhmer":               ("CDU/CSU", "CDU", "federal"),
    "Michael Meister":            ("CDU/CSU", "CDU", "federal"),
    "Monika Grütters":            ("CDU/CSU", "CDU", "federal"),
    "Ole Schröder":               ("CDU/CSU", "CDU", "federal"),
    "Peter Altmaier":             ("CDU/CSU", "CDU", "federal"),
    "Peter Tauber":               ("CDU/CSU", "CDU", "federal"),
    "Thomas Silberhorn":          ("CDU/CSU", "CSU", "federal"),
    "Ursula Leyen":               ("CDU/CSU", "CDU", "federal"),  # von der Leyen
    # --- SPD, federal government (19. WP) ---
    "Anette Kramme":              ("SPD", "SPD", "federal"),
    "Brigitte Zypries":           ("SPD", "SPD", "federal"),
    "Caren Marks":                ("SPD", "SPD", "federal"),
    "Christine Lambrecht":        ("SPD", "SPD", "federal"),
    "Florian Pronold":            ("SPD", "SPD", "federal"),
    "Franziska Giffey":           ("SPD", "SPD", "federal"),
    "Heiko Maas":                 ("SPD", "SPD", "federal"),
    "Olaf Scholz":                ("SPD", "SPD", "federal"),
    "Rita Schwarzelühr-Sutter":   ("SPD", "SPD", "federal"),
    "Svenja Schulze":             ("SPD", "SPD", "federal"),
    # --- Land ministers speaking via the Bundesrat, not federal cabinet ---
    "Andreas Pinkwart":           ("FDP",     "FDP",   "state"),
    "Benjamin-Immanuel Hoff":     ("LINKE",   "LINKE", "state"),
    "Boris Pistorius":            ("SPD",     "SPD",   "state"),
    "Georg Maier":                ("SPD",     "SPD",   "state"),
    "Joachim Stamp":              ("FDP",     "FDP",   "state"),
    "Karl-Josef Laumann":         ("CDU/CSU", "CDU",   "state"),
    "Till Backhaus":              ("SPD",     "SPD",   "state"),
}

CABINET_LABEL = "Cabinet"


def derive_from_corpus(df) -> dict[str, str]:
    """Map Cabinet speakers to a party using their own non-Cabinet speeches.

    Returns only unambiguous mappings (speakers with exactly one observed party).
    """
    cabinet_names = set(df.loc[df.Party == CABINET_LABEL, "speech_identification_ent"].dropna())
    other = df[(df.Party != CABINET_LABEL) & df.speech_identification_ent.isin(cabinet_names)]
    derived: dict[str, str] = {}
    for name, parties in other.groupby("speech_identification_ent").Party:
        observed = set(parties)
        if len(observed) == 1:
            derived[name] = observed.pop()
    return derived


def build_mapping(df) -> dict[str, dict]:
    """Full Cabinet->party mapping with provenance for every entry."""
    cabinet_names = sorted(set(df.loc[df.Party == CABINET_LABEL, "speech_identification_ent"].dropna()))
    derived = derive_from_corpus(df)

    mapping: dict[str, dict] = {}
    for name in cabinet_names:
        if name in derived:
            mapping[name] = {
                "party": derived[name],
                "party_detail": derived[name],
                "level": "federal",
                "source": "corpus",
            }
        elif name in MANUAL:
            group, detail, level = MANUAL[name]
            mapping[name] = {
                "party": group,
                "party_detail": detail,
                "level": level,
                "source": "manual",
            }
        else:
            mapping[name] = {
                "party": None,
                "party_detail": None,
                "level": None,
                "source": "unresolved",
            }
    return mapping
