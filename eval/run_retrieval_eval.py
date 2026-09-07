"""Tier-1 evaluation: retrieval reachability.

Answers one question per case: given the metadata filters that the query parser
derives from a question, how many chunks in the index are reachable, and are the
expected speakers among them?

This deliberately needs no embeddings and no LLM. Filter correctness is a
property of the metadata alone, so it runs in seconds, costs nothing, and
isolates retrieval bugs from generation bugs.

Usage:
    python eval/run_retrieval_eval.py [--index PATH] [--json OUT]
"""
from __future__ import annotations
import argparse, pickle, sys, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import yaml
from query_parser import parse_query_filters

DEFAULT_INDEX = ROOT / "vector_databases/vector_db_debates_lp19/index.pkl"
MIN_COVERAGE = 0.9  # a named speaker must be >=90% reachable under the filter


def load_docs(index_pkl: Path):
    docstore, _ = pickle.load(open(index_pkl, "rb"))[:2]
    return list(docstore._dict.values())


def matches(meta: dict, filters: dict) -> bool:
    return all(meta.get(k) == v for k, v in filters.items())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    docs = load_docs(args.index)
    metas = [d.metadata for d in docs]
    # same rule as app.py load_known_speakers(): drop blanks and 1-char names
    known_speakers = {str(m.get("speaker_name","")).strip() for m in metas}
    known_speakers = {n for n in known_speakers if len(n) > 1}
    cases = yaml.safe_load(open(ROOT / "eval/queries.yaml"))["cases"]

    results, passed, failed, skipped = [], 0, 0, 0
    print(f"index: {args.index}")
    print(f"chunks: {len(docs):,} | distinct speakers: {len(known_speakers):,}\n")

    for c in cases:
        if c.get("tier") != 1:
            skipped += 1
            continue
        _, filters = parse_query_filters(c["query"], known_speakers)
        hit_metas = [m for m in metas if matches(m, filters)]
        hits = len(hit_metas)
        reachable = {m.get("speaker_name") for m in hit_metas}

        problems = []
        if hits < c.get("expect_hits", 1):
            problems.append(f"{hits} hits < {c.get('expect_hits', 1)}")
        # Presence is not enough: a speaker whose chunks are mostly mislabelled
        # is still effectively unreachable. Require most of their corpus.
        for s in c.get("expect_speakers", []):
            total = sum(1 for m in metas if m.get("speaker_name") == s)
            got = sum(1 for m in hit_metas if m.get("speaker_name") == s)
            cov = got / total if total else 0.0
            if cov < MIN_COVERAGE:
                problems.append(
                    f"{s}: {got}/{total} chunks reachable ({cov:.1%})")

        ok = not problems
        passed, failed = passed + ok, failed + (not ok)
        print(f"  [{'PASS' if ok else 'FAIL'}] {c['id']:14} {hits:>7,} hits  {filters}")
        for p in problems:
            print(f"         └─ {p}")
        results.append({"id": c["id"], "ok": ok, "hits": hits,
                        "filters": filters, "problems": problems})

    total = passed + failed
    print(f"\n{passed}/{total} passed" + (f"  ({failed} failed)" if failed else "")
          + f"  [{skipped} tier-2 cases skipped]")
    if args.json:
        args.json.write_text(json.dumps(
            {"index": str(args.index), "passed": passed, "total": total,
             "results": results}, ensure_ascii=False, indent=2))
        print(f"wrote {args.json}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
