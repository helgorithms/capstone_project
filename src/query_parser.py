"""Query parsing: natural-language question -> (semantic string, metadata filters).

Extracted verbatim from app.py so it can be imported and tested without
Streamlit. app.py should import from here rather than keeping its own copy.
"""
from __future__ import annotations
import re

FILTER_MAP = {
    # parties
    "spd":       ("party", "SPD"),
    "cdu":       ("party", "CDU"),
    "csu":       ("party", "CSU"),
    "cdu/csu":   ("party", "CDU/CSU"),
    "grüne":     ("party", "GRÜNE"),
    "grünen":    ("party", "GRÜNE"),
    "fdp":       ("party", "FDP"),
    "linke":     ("party", "LINKE"),
    "afd":       ("party", "AfD"),
    "fraktionslos": ("party", "fraktionslos"),
    "parteilos": ("party", "parteilos"),
    # government status
    "kabinett":         ("party", "Cabinet"),
    "regierungspartei": ("government_status", 1),
    "bundesregierung":  ("government_status", 1),
    "regierung":        ("government_status", 1),
    "opposition":       ("government_status", 0),
    # roles
    "kanzler":         ("role", "Bundeskanzler"),
    "bundeskanzler":   ("role", "Bundeskanzler"),
    "kanzlerin":       ("role", "Bundeskanzler"),
    "bundeskanzlerin": ("role", "Bundeskanzler"),
    "minister":        ("role", "Bundesminister"),
    "bundesminister":  ("role", "Bundesminister"),
    "staatssekretär":  ("role", "Staatssekretär"),
    "staatsminister":  ("role", "Staatsminister"),
    "abgeordnete":     ("role", "MdB"),
    "mdb":             ("role", "MdB"),
    "mitglied des bundestags": ("role", "MdB"),
    # legislative periods
    "19. wahlperiode": ("legislative_period", 19),
    "19 wahlperiode":  ("legislative_period", 19),
    "wp19":            ("legislative_period", 19),
    # time frame
    "2021": ("year", "2021"),
    "2020": ("year", "2020"),
    "2019": ("year", "2019"),
    "2018": ("year", "2018"),
    "2017": ("year", "2017"),
}


# ── Pipeline functions ───────────────────────────────────────────
def parse_query_filters(user_input: str, known_speakers: set) -> tuple[str, dict]:
    """Parse user query into semantic search string and metadata filters."""
    filters = {}
    semantic = user_input.lower()

    # FILTER_MAP lookup — word boundaries, longest first
    for term, (key, value) in sorted(FILTER_MAP.items(), key=lambda x: len(x[0]), reverse=True):
        pattern = r'(?<!\w)' + re.escape(term) + r'(?!\w)'
        if re.search(pattern, semantic):
            filters[key] = value
            semantic = re.sub(pattern, '', semantic)

    # Party takes precedence over conflicting government_status
    if "party" in filters and "government_status" in filters:
        del filters["government_status"]

    # Session
    session_match = re.search(r'(\d+)\.\s*sitzung|sitzung\s*(\d+)', semantic)
    if session_match:
        session_num = session_match.group(1) or session_match.group(2)
        filters["session"] = int(session_num)
        semantic = re.sub(r'(\d+)\.\s*sitzung|sitzung\s*(\d+)', '', semantic)

    # Date
    date_match = re.search(r'\d{4}-\d{2}-\d{2}|\d{2}\.\d{2}\.\d{4}', semantic)
    if date_match:
        filters["date"] = date_match.group()
        semantic = semantic.replace(date_match.group(), "")

    # Speaker — match against known speakers
    semantic_for_speaker = re.sub(r'(\w)s\b', r'\1', semantic)
    speaker_found = None
    for name in sorted(known_speakers, key=len, reverse=True):
        if name.lower() in semantic_for_speaker:
            speaker_found = name
            break
    if speaker_found:
        filters["speaker_name"] = speaker_found
        semantic = re.sub(re.escape(speaker_found.lower()) + r's?\b', '', semantic)

    semantic = re.sub(r'\s+', ' ', semantic).strip()
    return semantic, filters

