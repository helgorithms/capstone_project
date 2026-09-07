"""Repair party metadata in the FAISS docstore — without re-embedding.

``party`` is metadata, not text. The vectors in index.faiss encode the speech
content and are unaffected by a metadata correction, so only index.pkl (the
docstore) needs rewriting. This turns an hours-long re-embedding job into a
sub-minute operation, and guarantees the vectors are bit-identical before and
after — the retrieval geometry does not change, only what is reachable by filter.

Each patched document keeps a record of what was changed:
    party               -> resolved parliamentary group (was "Cabinet")
    party_detail        -> actual party (CDU or CSU, where the group merges them)
    speaker_level       -> "federal" | "state" (Bundesrat speakers)
    party_original      -> "Cabinet"
    party_resolved_by   -> "corpus" | "manual"

Usage:
    python src/patch_cabinet_metadata.py [--dry-run]
"""
from __future__ import annotations
import argparse, pickle, shutil, sys
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import pandas as pd
from cabinet_party_map import build_mapping, CABINET_LABEL

INDEX_PKL = ROOT / "vector_databases/vector_db_debates_lp19/index.pkl"
CSV = ROOT / "data/debates_lp19.csv"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--index", type=Path, default=INDEX_PKL)
    args = ap.parse_args()

    mapping = build_mapping(pd.read_csv(CSV))
    unresolved = [k for k, v in mapping.items() if v["party"] is None]
    if unresolved:
        print(f"refusing to patch: {len(unresolved)} unresolved speakers: {unresolved}")
        return 1

    obj = pickle.load(open(args.index, "rb"))
    docstore, id_map = obj[0], obj[1]
    docs = docstore._dict

    patched, missing = 0, Counter()
    for doc in docs.values():
        m = doc.metadata
        if m.get("party") != CABINET_LABEL:
            continue
        name = str(m.get("speaker_name", "")).strip()
        info = mapping.get(name)
        if not info:
            missing[name] += 1
            continue
        m["party_original"] = CABINET_LABEL
        m["party"] = info["party"]
        m["party_detail"] = info["party_detail"]
        m["speaker_level"] = info["level"]
        m["party_resolved_by"] = info["source"]
        patched += 1

    print(f"chunks patched: {patched:,}")
    if missing:
        print(f"unmapped speakers encountered: {dict(missing)}")
    print("new party distribution:")
    for k, v in Counter(d.metadata.get("party") for d in docs.values()).most_common():
        print(f"  {k}: {v:,}")

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return 0

    backup = args.index.with_suffix(".pkl.bak")
    if not backup.exists():
        shutil.copy2(args.index, backup)
        print(f"backup written: {backup.name}")
    with open(args.index, "wb") as f:
        pickle.dump((docstore, id_map, *obj[2:]), f)
    print(f"wrote {args.index.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
