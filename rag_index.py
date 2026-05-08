"""
rag_index.py

Build a ChromaDB vector index from part4_clauses_v2.json and part4_tables.json.

Collections:
  nbcc_sentences  — 633 sentence-level chunks
  nbcc_tables     — 35 table chunks

Embedding: all-MiniLM-L6-v2 via ChromaDB DefaultEmbeddingFunction (ONNX, no PyTorch)
Persist:   ./chroma_db/

Usage:
  python rag_index.py            # build from scratch
  python rag_index.py --reset    # wipe and rebuild
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

try:
    import chromadb
    from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
except ImportError:
    sys.exit("Run: python -m pip install chromadb")

ROOT = Path(__file__).resolve().parent
CLAUSES_PATH = ROOT / "part4_clauses_v2.json"
TABLES_PATH  = ROOT / "part4_tables.json"
CHROMA_DIR   = ROOT / "chroma_db"

COL_SENTENCES = "nbcc_sentences"
COL_TABLES    = "nbcc_tables"


# ── embedding text builders ─────────────────────────────────────────────────

def sentence_doc(art: dict, s: dict) -> str:
    """Rich text used for embedding a sentence chunk."""
    parts = [
        f"[{s['sentence_id']}]",
        f"Section: {art.get('section','')} > {art.get('subsection','')}",
        f"Article: {art.get('article_title','')}",
        "",
        s.get("original_text", ""),
    ]
    if s.get("requirements"):
        parts.append("Requirements: " + " | ".join(s["requirements"]))
    if s.get("conditions"):
        parts.append("Conditions: " + " | ".join(s["conditions"]))
    if s.get("exceptions"):
        parts.append("Exceptions: " + " | ".join(s["exceptions"]))
    if s.get("definitions"):
        syms = ", ".join(d["symbol"] for d in s["definitions"])
        parts.append(f"Defines: {syms}")
    return "\n".join(parts)


def table_doc(t: dict) -> str:
    """Rich text used for embedding a table chunk."""
    return "\n".join([
        f"[Table {t.get('table_id','')}]",
        f"Title: {t.get('table_title','')}",
        "",
        t.get("content", ""),
    ])


# ── metadata helpers ────────────────────────────────────────────────────────

def _j(v) -> str:
    """Serialize a value to compact JSON string for ChromaDB metadata."""
    return json.dumps(v, ensure_ascii=False)


def sentence_meta(art: dict, s: dict) -> dict:
    refs = s.get("references", {})
    return {
        "source_type":     "sentence",
        "sentence_id":     s.get("sentence_id", ""),
        "article_id":      art.get("article_id", ""),
        "article_title":   art.get("article_title", ""),
        "section":         art.get("section", ""),
        "subsection":      art.get("subsection", ""),
        "original_text":   s.get("original_text", ""),
        "conditions":      _j(s.get("conditions", [])),
        "requirements":    _j(s.get("requirements", [])),
        "exceptions":      _j(s.get("exceptions", [])),
        "definitions":     _j(s.get("definitions", [])),
        "ref_sentences":   _j(refs.get("sentences", [])),
        "ref_tables":      _j(refs.get("tables", [])),
        "ref_articles":    _j(refs.get("articles", [])),
        "ref_standards":   _j(refs.get("standards", [])),
        "needs_review":    str(s.get("needs_review", False)),
    }


def table_meta(t: dict) -> dict:
    return {
        "source_type": "table",
        "table_key":   t.get("table_key", ""),
        "table_id":    t.get("table_id", ""),
        "table_title": t.get("table_title", ""),
        "article_id":  t.get("article_id", ""),
        "content":     t.get("content", ""),
    }


# ── index builder ───────────────────────────────────────────────────────────

def build_index(reset: bool = False) -> None:
    print(f"Loading data...")
    clauses = json.loads(CLAUSES_PATH.read_text(encoding="utf-8"))
    tables  = json.loads(TABLES_PATH.read_text(encoding="utf-8"))

    print(f"  {len(clauses)} articles  |  {sum(len(a['sentences']) for a in clauses)} sentences  |  {len(tables)} tables")

    print(f"Initialising ChromaDB at {CHROMA_DIR} ...")
    ef = DefaultEmbeddingFunction()
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    if reset:
        for name in (COL_SENTENCES, COL_TABLES):
            try:
                client.delete_collection(name)
                print(f"  Dropped: {name}")
            except Exception:
                pass

    col_s = client.get_or_create_collection(COL_SENTENCES, embedding_function=ef,
                                            metadata={"hnsw:space": "cosine"})
    col_t = client.get_or_create_collection(COL_TABLES,    embedding_function=ef,
                                            metadata={"hnsw:space": "cosine"})

    # ── index sentences ──────────────────────────────────────────────────────
    existing_s = set(col_s.get(include=[])["ids"])
    docs_s, metas_s, ids_s = [], [], []
    seen_s: set[str] = set()

    for art in clauses:
        for s in art.get("sentences", []):
            sid_raw = s.get("sentence_id", "")
            # make globally unique: article_id + "|" + sentence_id
            uid = f"{art.get('article_id','')}|{sid_raw}"
            if not sid_raw or uid in existing_s or uid in seen_s:
                continue
            seen_s.add(uid)
            docs_s.append(sentence_doc(art, s))
            meta = sentence_meta(art, s)
            meta["uid"] = uid          # store the composite key for reference
            metas_s.append(meta)
            ids_s.append(uid)

    if ids_s:
        BATCH = 100
        for i in range(0, len(ids_s), BATCH):
            col_s.add(documents=docs_s[i:i+BATCH],
                      metadatas=metas_s[i:i+BATCH],
                      ids=ids_s[i:i+BATCH])
            print(f"  Sentences: {min(i+BATCH, len(ids_s))}/{len(ids_s)} indexed")
    else:
        print("  Sentences: all already indexed, skipping.")

    # ── index tables ─────────────────────────────────────────────────────────
    existing_t = set(col_t.get(include=[])["ids"])
    docs_t, metas_t, ids_t = [], [], []

    for t in tables:
        tid = t.get("table_key", "")
        if not tid or tid in existing_t:
            continue
        docs_t.append(table_doc(t))
        metas_t.append(table_meta(t))
        ids_t.append(tid)

    if ids_t:
        col_t.add(documents=docs_t, metadatas=metas_t, ids=ids_t)
        print(f"  Tables:    {len(ids_t)} indexed")
    else:
        print("  Tables: all already indexed, skipping.")

    print(f"\nDone.")
    print(f"  {col_s.count()} sentences in '{COL_SENTENCES}'")
    print(f"  {col_t.count()} tables    in '{COL_TABLES}'")
    print(f"  DB path: {CHROMA_DIR}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset", action="store_true", help="Drop and rebuild collections")
    args = ap.parse_args()
    build_index(reset=args.reset)


if __name__ == "__main__":
    main()
