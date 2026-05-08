"""
rag_query.py

RAG query engine: retrieve from ChromaDB, answer with Claude.
Importable by the Streamlit app.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

try:
    import chromadb
    from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
except ImportError:
    sys.exit("Run: python -m pip install chromadb")

try:
    from anthropic import Anthropic
except ImportError:
    sys.exit("Run: python -m pip install anthropic")

ROOT       = Path(__file__).resolve().parent
CHROMA_DIR = ROOT / "chroma_db"
COL_SENTENCES = "nbcc_sentences"
COL_TABLES    = "nbcc_tables"
CLAUDE_MODEL  = "claude-sonnet-4-6"
MAX_TOKENS    = 4096

TOP_K_SENTENCES = 6
TOP_K_TABLES    = 3

SYSTEM_PROMPT = """You are an expert on the National Building Code of Canada (NBC) Part 4 – Structural Design.
Answer questions accurately using ONLY the provided context excerpts.
Rules:
- Cite the source (sentence_id or table_id) for every claim you make, using inline brackets, e.g. [4.1.6.2.(1)] or [Table 4.1.6.2.-A].
- If the context does not contain enough information, say so clearly instead of guessing.
- Keep answers concise and structured. Use bullet points or numbered lists where appropriate.
- Do not translate or paraphrase; quote the original English text when precision matters.
- If a sentence has needs_review=true, mention that it may require verification.
"""


def _api_key() -> str:
    # 1. environment variable (works locally and on Streamlit Community Cloud)
    key = (os.environ.get("ANTHROPIC_API_KEY") or "").strip()
    if key:
        return key
    # 2. Streamlit secrets (when running inside a Streamlit app)
    try:
        import streamlit as st
        key = (st.secrets.get("ANTHROPIC_API_KEY") or "").strip()
        if key:
            return key
    except Exception:
        pass
    raise RuntimeError(
        "ANTHROPIC_API_KEY not found. "
        "Set it as an environment variable or in .streamlit/secrets.toml."
    )


def _load_chroma():
    ef = DefaultEmbeddingFunction()
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    col_s = client.get_collection(COL_SENTENCES, embedding_function=ef)
    col_t = client.get_collection(COL_TABLES,    embedding_function=ef)
    return col_s, col_t


# ── singleton cache (avoids reloading the embedding model on each call) ──────
_chroma_cache: tuple | None = None
_claude_cache: Anthropic | None = None


def _get_clients():
    global _chroma_cache, _claude_cache
    if _chroma_cache is None:
        _chroma_cache = _load_chroma()
    if _claude_cache is None:
        _claude_cache = Anthropic(api_key=_api_key())
    return _chroma_cache, _claude_cache


# ── retrieval ────────────────────────────────────────────────────────────────

def retrieve(query: str) -> tuple[list[dict], list[dict]]:
    """Return (sentence_hits, table_hits) sorted by relevance."""
    (col_s, col_t), _ = _get_clients()

    rs = col_s.query(
        query_texts=[query],
        n_results=TOP_K_SENTENCES,
        include=["metadatas", "distances", "documents"],
    )
    rt = col_t.query(
        query_texts=[query],
        n_results=TOP_K_TABLES,
        include=["metadatas", "distances"],
    )

    sentence_hits = [
        {**meta, "distance": dist, "document": doc}
        for meta, dist, doc in zip(
            rs["metadatas"][0], rs["distances"][0], rs["documents"][0]
        )
    ]
    table_hits = [
        {**meta, "distance": dist}
        for meta, dist in zip(rt["metadatas"][0], rt["distances"][0])
    ]
    return sentence_hits, table_hits


# ── context builder ──────────────────────────────────────────────────────────

def build_context(sentence_hits: list[dict], table_hits: list[dict]) -> str:
    blocks: list[str] = []

    for h in sentence_hits:
        refs = {
            "sentences": json.loads(h.get("ref_sentences", "[]")),
            "tables":    json.loads(h.get("ref_tables",    "[]")),
            "articles":  json.loads(h.get("ref_articles",  "[]")),
            "standards": json.loads(h.get("ref_standards", "[]")),
        }
        all_refs = refs["sentences"] + refs["tables"] + refs["articles"] + refs["standards"]
        refs_str = ", ".join(all_refs) if all_refs else "none"

        conds = json.loads(h.get("conditions",   "[]"))
        reqs  = json.loads(h.get("requirements", "[]"))
        excps = json.loads(h.get("exceptions",   "[]"))
        defs  = json.loads(h.get("definitions",  "[]"))
        review = h.get("needs_review", "False") == "True"

        lines = [
            f"--- SENTENCE [{h['sentence_id']}] ---",
            f"Section: {h.get('section','')} > {h.get('subsection','')}",
            f"Article: {h.get('article_title','')}  ({h.get('article_id','')})",
            f"Original text: {h.get('original_text','')}",
        ]
        if reqs:
            lines.append("Requirements: " + " | ".join(reqs))
        if conds:
            lines.append("Conditions: " + " | ".join(conds))
        if excps:
            lines.append("Exceptions: " + " | ".join(excps))
        if defs:
            def_strs = [f"{d['symbol']}: {d['meaning']}" for d in defs if isinstance(d, dict)]
            lines.append("Definitions: " + " | ".join(def_strs))
        lines.append(f"References: {refs_str}")
        if review:
            lines.append("⚠ needs_review=true (verify this sentence)")
        blocks.append("\n".join(lines))

    for h in table_hits:
        blocks.append(
            f"--- TABLE [{h.get('table_id','')}] ---\n"
            f"Title: {h.get('table_title','')}\n"
            f"Article: {h.get('article_id','')}\n"
            f"Content:\n{h.get('content','')}"
        )

    return "\n\n".join(blocks)


# ── answer generation ────────────────────────────────────────────────────────

def answer(
    question: str,
    history: list[dict] | None = None,
) -> tuple[str, list[dict], list[dict]]:
    """
    Returns (answer_text, sentence_hits, table_hits).
    history: list of {"role": "user"/"assistant", "content": "..."} for multi-turn.
    """
    sentence_hits, table_hits = retrieve(question)
    context = build_context(sentence_hits, table_hits)

    user_content = (
        f"Context from NBC Part 4:\n\n{context}\n\n"
        f"Question: {question}"
    )

    messages: list[dict] = []
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_content})

    _, client = _get_clients()
    resp = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=messages,
    )
    text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
    return text, sentence_hits, table_hits


# ── CLI convenience ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "What are the importance categories for buildings?"
    print(f"Query: {q}\n")
    ans, s_hits, t_hits = answer(q)
    print("=== Answer ===")
    print(ans)
    print(f"\n=== Sources ({len(s_hits)} sentences, {len(t_hits)} tables) ===")
    for h in s_hits:
        print(f"  [{h['sentence_id']}]  dist={h['distance']:.3f}  {h['subsection']}")
    for h in t_hits:
        print(f"  [Table {h['table_id']}]  dist={h['distance']:.3f}")
