"""In-memory repair of the Cabinet/party metadata collision.

The published index carries ``party="Cabinet"`` for every member of the federal
government, which makes 19,269 chunks (8.8% of the corpus) unreachable by any
party filter. Rather than shipping a corrected index — which would mean pushing
141 MB through Git LFS on every change — the correction is applied to the
docstore in memory each time the index is loaded.

This is cheap (a dict update per affected document, well under a second for
219k chunks), keeps the published artefact immutable, and means the repair
survives a future migration to a different vector store.

Both the application and the evaluation harness call ``repair_docstore`` so the
two can never diverge.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Iterable

MAPPING_PATH = Path(__file__).resolve().parent / "cabinet_party_map.json"
CABINET_LABEL = "Cabinet"


def load_mapping(path: Path | None = None) -> dict[str, dict]:
    """Load the precomputed speaker -> party mapping (see cabinet_party_map.py)."""
    return json.loads((path or MAPPING_PATH).read_text(encoding="utf-8"))


def repair_metadata(metadatas: Iterable[dict], mapping: dict[str, dict] | None = None) -> int:
    """Resolve party="Cabinet" in place. Returns the number of records changed.

    Idempotent: records already repaired are skipped, so calling this on an
    already-corrected index is a no-op.
    """
    mapping = mapping if mapping is not None else load_mapping()
    changed = 0
    for meta in metadatas:
        if meta.get("party") != CABINET_LABEL:
            continue
        info = mapping.get(str(meta.get("speaker_name", "")).strip())
        if not info or not info.get("party"):
            continue
        meta["party_original"] = CABINET_LABEL
        meta["party"] = info["party"]
        meta["party_detail"] = info["party_detail"]
        meta["speaker_level"] = info["level"]
        meta["party_resolved_by"] = info["source"]
        changed += 1
    return changed


def repair_docstore(vectorstore, mapping: dict[str, dict] | None = None) -> int:
    """Repair a loaded LangChain FAISS vectorstore in place."""
    docs = vectorstore.docstore._dict.values()
    return repair_metadata((d.metadata for d in docs), mapping)
