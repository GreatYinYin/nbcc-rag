"""
extract_logic_v2.py

Read part4_clauses.json, extract sentence-level logic per Article using Claude,
write:
- part4_clauses_v2.json (article-level structured output)
- part4_tables.json (table blocks only)

Rules enforced by prompt:
- English original excerpts only (no translation)
- Table references only as IDs in sentence references
- needs_review=true when uncertain

Supports --resume and --limit.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

try:
    from anthropic import Anthropic
except ImportError:
    sys.exit("Please install dependency first: python -m pip install anthropic")

ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "part4_clauses.json"
DEFAULT_ARTICLE_OUTPUT = ROOT / "part4_clauses_v2.json"
DEFAULT_TABLE_OUTPUT = ROOT / "part4_tables.json"

API_KEY = ""
MODEL = "claude-sonnet-4-6"
MAX_BODY_CHARS = 120_000
MAX_TOKENS = 8192

SYSTEM_PROMPT = """You are an NBC Part 4 extraction assistant.
You must output strict JSON only.
Hard rules:
1) Keep extracted content in original English from provided clause text. No translation, no paraphrase.
2) Extract at sentence level.
3) If uncertain/misaligned/ambiguous, set needs_review=true for that sentence.
4) In references, classify each reference into its correct bucket:
   - sentences: items like "Sentence 4.1.x.x.(x)"
   - tables: items like "Table 4.1.x.x" or "A-Table ..."
   - articles: items like "Article 4.1.x.x", "Subsection 4.1.x.", "Section 4.1."
   - standards: items like "CSA S16", "ASTM ...", "NBC ...", "Note A-..."
5) Do not include table body content in references.
6) Preserve original_text exactly for each extracted sentence chunk from the clause body.
"""

USER_TEMPLATE = """Extract one ARTICLE object from the clause below.
Output ONE JSON object only with exactly these top-level fields:
- article_id: string
- article_title: string
- section: string  (e.g. "4.1. Structural Loads and Procedures"; use hint below or infer from clause)
- subsection: string (e.g. "Specified Loads and Effects"; use hint below or infer from clause)
- sentences: array

Each sentence item must be:
{
  "sentence_id": "4.1.x.x.(n) or best-guess; if unknown use article_id + '.(?)'",
  "definitions": [{"symbol": "...", "meaning": "..."}],
  "conditions": ["English original excerpt"],
  "requirements": ["English original excerpt"],
  "exceptions": ["English original excerpt"],
  "references": {
    "sentences": ["Sentence 4.1.x.x.(x)"],
    "tables": ["Table 4.1.x.x"],
    "articles": ["Article ...", "Subsection ...", "Section ..."],
    "standards": ["CSA S16", "ASTM ...", "Note A-..."]
  },
  "needs_review": false,
  "original_text": "exact English sentence/chunk from clause"
}

Constraints:
- Keep all arrays/objects present; use [] or {} when empty.
- definitions.meaning must be original English text.
- original_text must be copied from the clause text (no rewriting).
- Prefer explicit sentence numbering when available; if not explicit, infer cautiously and set needs_review=true.
- Exclude table body rows from sentences; table content is handled separately.

article_id hint: @ARTICLE_ID@
section hint: @SECTION@
subsection hint: @SUBSECTION@

Clause body (tables may be replaced by table markers):
---
@CLAUSE_BODY@
---
"""

ARTICLE_ID_RE = re.compile(r"\b(\d+\.\d+\.\d+\.\d+)\b")
TABLE_ID_RE = re.compile(r"\b(?:A-)?Table\s+([A-Za-z0-9\.\-]+)")
H_TAG_RE = re.compile(r"^\[H\d+\]\s*")
H3_RE = re.compile(r"\[H3\]\s*(.+)")
H8_RE = re.compile(r"\[H8\]\s*(.+)")


def _api_key_from_test_py() -> str | None:
    p = ROOT / "test.py"
    if not p.is_file():
        return None
    text = p.read_text(encoding="utf-8")
    m = re.search(r'^API_KEY\s*=\s*"([^\"]+)"', text, re.MULTILINE)
    if not m:
        return None
    key = m.group(1).strip()
    return key or None


def get_api_key() -> str:
    key = (os.environ.get("ANTHROPIC_API_KEY") or API_KEY or "").strip()
    if key and key != "YOUR_ANTHROPIC_API_KEY":
        return key
    key = _api_key_from_test_py()
    if key and key != "YOUR_ANTHROPIC_API_KEY":
        return key
    sys.exit("Missing API key. Set ANTHROPIC_API_KEY or API_KEY (or provide it in test.py).")


def strip_h_tag(line: str) -> str:
    return H_TAG_RE.sub("", line).strip()


def clean_h_text(text: str) -> str:
    """Remove tabs and extra whitespace from H-tag content."""
    return re.sub(r"\s+", " ", text).strip()


def guess_article_id(clause_id: str, body: str) -> str:
    cid = clause_id.strip()
    if cid:
        return cid
    if ARTICLE_ID_RE.fullmatch(cid):
        return clause_id
    m = ARTICLE_ID_RE.search(body)
    return m.group(1) if m else cid


def build_section_context(clauses: list[dict[str, Any]]) -> dict[str, tuple[str, str]]:
    """Pre-scan all clauses to build article_id -> (section, subsection) mapping.

    Tracks H3 (section) and H8 (subsection) tags sequentially across all clauses.
    """
    current_section = ""
    current_subsection = ""
    context: dict[str, tuple[str, str]] = {}

    for row in clauses:
        clause_id = str(row.get("clause_id", ""))
        body = str(row.get("body", ""))

        for line in body.splitlines():
            m3 = H3_RE.match(line.strip())
            if m3:
                raw = m3.group(1)
                # strip "Section " prefix and trailing dot+whitespace if present
                cleaned = clean_h_text(raw)
                cleaned = re.sub(r"^Section\s+", "", cleaned)
                current_section = cleaned

            m8 = H8_RE.match(line.strip())
            if m8:
                current_subsection = clean_h_text(m8.group(1))

        if clause_id and clause_id != "(preamble)":
            article_id = guess_article_id(clause_id, body)
            context[article_id] = (current_section, current_subsection)

    return context


def extract_tables(article_id: str, body: str) -> tuple[str, list[dict[str, Any]]]:
    """Extract [TABLE]...[/TABLE] blocks and return body with table placeholders."""
    lines = body.splitlines()
    clean_lines: list[str] = []
    tables: list[dict[str, Any]] = []

    in_table = False
    current_table_lines: list[str] = []
    pending_table_id: str | None = None
    pending_table_title: str | None = None
    table_index = 0

    def finalize_table() -> None:
        nonlocal current_table_lines, pending_table_id, pending_table_title, table_index
        if not current_table_lines:
            return
        table_index += 1
        table_id = pending_table_id or f"UNKNOWN_{article_id}_{table_index}"
        table_key = f"{article_id}:{table_id}:{table_index}"
        tables.append(
            {
                "table_key": table_key,
                "article_id": article_id,
                "table_id": table_id,
                "table_title": pending_table_title or "",
                "content": "\n".join(current_table_lines).strip(),
            }
        )
        clean_lines.append(f"[TABLE_REF: Table {table_id}]")
        current_table_lines = []

    for i, raw in enumerate(lines):
        line = raw.rstrip("\n")

        if not in_table:
            title_candidate = strip_h_tag(line)
            m = TABLE_ID_RE.search(title_candidate)
            if m:
                pending_table_id = m.group(1)
                pending_table_title = title_candidate

        if line.strip() == "[TABLE]":
            in_table = True
            current_table_lines = []
            continue

        if line.strip() == "[/TABLE]":
            in_table = False
            finalize_table()
            continue

        if in_table:
            current_table_lines.append(line)
        else:
            clean_lines.append(line)

        if not in_table and pending_table_id and i > 0 and len(clean_lines) % 40 == 0:
            pending_table_id = pending_table_id

    if in_table:
        finalize_table()

    return "\n".join(clean_lines), tables


def truncate_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n\n[...TRUNCATED...]"


def extract_json_object(text: str) -> dict[str, Any]:
    t = text.strip()
    if t.startswith("```"):
        parts = t.splitlines()
        if parts and parts[0].startswith("```"):
            parts = parts[1:]
        if parts and parts[-1].strip() == "```":
            parts = parts[:-1]
        t = "\n".join(parts).strip()
    return json.loads(t)


def classify_references(refs: Any) -> dict[str, list[str]]:
    """Classify a flat reference list or already-typed dict into typed buckets."""
    empty: dict[str, list[str]] = {"sentences": [], "tables": [], "articles": [], "standards": []}

    if isinstance(refs, dict):
        # already typed by Claude — normalise each bucket
        result: dict[str, list[str]] = {}
        for key in ("sentences", "tables", "articles", "standards"):
            raw = refs.get(key, [])
            result[key] = [str(x).strip() for x in (raw if isinstance(raw, list) else [raw]) if str(x).strip()]
        return result

    if not isinstance(refs, list):
        return empty

    result = {"sentences": [], "tables": [], "articles": [], "standards": []}
    for ref in refs:
        ref = str(ref).strip()
        if not ref:
            continue
        lower = ref.lower()
        if lower.startswith("sentence") or re.match(r"^\d+\.\d+\.\d+\.\d+\.\(", lower):
            result["sentences"].append(ref)
        elif lower.startswith("table") or lower.startswith("a-table"):
            result["tables"].append(ref)
        elif lower.startswith("article") or lower.startswith("subsection") or lower.startswith("section"):
            result["articles"].append(ref)
        else:
            result["standards"].append(ref)
    return result


def normalize_sentence(article_id: str, item: Any) -> dict[str, Any]:
    if not isinstance(item, dict):
        return {
            "sentence_id": f"{article_id}.(?)",
            "definitions": [],
            "conditions": [],
            "requirements": [],
            "exceptions": [],
            "references": {"sentences": [], "tables": [], "articles": [], "standards": []},
            "needs_review": True,
            "original_text": "",
        }

    def as_list_str(v: Any) -> list[str]:
        if v is None:
            return []
        if isinstance(v, str):
            return [v] if v.strip() else []
        if isinstance(v, list):
            return [str(x).strip() for x in v if str(x).strip()]
        return [str(v).strip()] if str(v).strip() else []

    defs = item.get("definitions", [])
    norm_defs: list[dict[str, str]] = []
    if isinstance(defs, list):
        for d in defs:
            if isinstance(d, dict):
                sym = str(d.get("symbol", "")).strip()
                mean = str(d.get("meaning", "")).strip()
                if sym or mean:
                    norm_defs.append({"symbol": sym, "meaning": mean})

    sid = str(item.get("sentence_id", f"{article_id}.(?)")).strip() or f"{article_id}.(?)"
    original = str(item.get("original_text", "")).strip()
    needs_review = bool(item.get("needs_review", False))
    if not original:
        needs_review = True

    return {
        "sentence_id": sid,
        "definitions": norm_defs,
        "conditions": as_list_str(item.get("conditions")),
        "requirements": as_list_str(item.get("requirements")),
        "exceptions": as_list_str(item.get("exceptions")),
        "references": classify_references(item.get("references", [])),
        "needs_review": needs_review,
        "original_text": original,
    }


def normalize_article_output(
    raw: dict[str, Any],
    article_id: str,
    section: str = "",
    subsection: str = "",
) -> dict[str, Any]:
    title = str(raw.get("article_title", "")).strip()
    # prefer Claude's section/subsection if provided and non-empty
    resolved_section = str(raw.get("section", "")).strip() or section
    resolved_subsection = str(raw.get("subsection", "")).strip() or subsection
    sentences_raw = raw.get("sentences", [])
    if not isinstance(sentences_raw, list):
        sentences_raw = []

    return {
        "article_id": article_id,
        "article_title": title,
        "section": resolved_section,
        "subsection": resolved_subsection,
        "sentences": [normalize_sentence(article_id, x) for x in sentences_raw],
    }


def call_claude_for_article(
    client: Anthropic,
    article_id: str,
    clause_body: str,
    section: str = "",
    subsection: str = "",
) -> dict[str, Any]:
    user_msg = (
        USER_TEMPLATE
        .replace("@ARTICLE_ID@", article_id)
        .replace("@SECTION@", section or "(unknown)")
        .replace("@SUBSECTION@", subsection or "(unknown)")
        .replace("@CLAUSE_BODY@", truncate_text(clause_body, MAX_BODY_CHARS))
    )
    msg = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
    )
    text = "".join(block.text for block in msg.content if getattr(block, "type", None) == "text")
    parsed = extract_json_object(text)
    return normalize_article_output(parsed, article_id, section=section, subsection=subsection)


def _empty_refs() -> dict[str, list[str]]:
    return {"sentences": [], "tables": [], "articles": [], "standards": []}


def load_done(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(data, list):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for item in data:
        if isinstance(item, dict) and isinstance(item.get("article_id"), str):
            out[item["article_id"]] = item
    return out


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract sentence-level logic from part4_clauses.json")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_ARTICLE_OUTPUT)
    parser.add_argument("--tables-output", type=Path, default=DEFAULT_TABLE_OUTPUT)
    parser.add_argument("--limit", type=int, default=0, help="Process first N articles only (0 = all)")
    parser.add_argument("--sleep", type=float, default=0.35, help="Seconds between API calls")
    parser.add_argument("--resume", action="store_true", help="Resume from *.checkpoint.json")
    args = parser.parse_args()

    in_path = args.input.expanduser().resolve()
    out_path = args.output.expanduser().resolve()
    tables_path = args.tables_output.expanduser().resolve()

    ck_path = out_path.parent / f"{out_path.stem}.checkpoint.json"
    tables_ck_path = tables_path.parent / f"{tables_path.stem}.checkpoint.json"

    if not in_path.is_file():
        sys.exit(f"Input file not found: {in_path}")

    clauses = json.loads(in_path.read_text(encoding="utf-8"))
    if not isinstance(clauses, list):
        sys.exit("Input JSON must be an array")

    # pre-scan ALL clauses for section/subsection context before applying --limit
    section_ctx = build_section_context(clauses)
    print(f"Section context built for {len(section_ctx)} articles.")

    if args.limit > 0:
        clauses = clauses[: args.limit]

    done_articles = load_done(ck_path) if args.resume else {}
    done_tables = (
        json.loads(tables_ck_path.read_text(encoding="utf-8"))
        if (args.resume and tables_ck_path.is_file())
        else []
    )
    if not isinstance(done_tables, list):
        done_tables = []

    table_keys = {str(t.get("table_key", "")) for t in done_tables if isinstance(t, dict)}

    client = Anthropic(api_key=get_api_key())

    results: list[dict[str, Any]] = []
    tables_all: list[dict[str, Any]] = list(done_tables)

    for idx, row in enumerate(clauses, start=1):
        clause_id = str(row.get("clause_id", f"#{idx}"))
        body = str(row.get("body", ""))
        article_id = guess_article_id(clause_id, body)
        section, subsection = section_ctx.get(article_id, ("", ""))

        if article_id in done_articles:
            results.append(done_articles[article_id])
            print(f"[{idx}/{len(clauses)}] Skip (resume) {article_id}")
            continue

        body_wo_tables, article_tables = extract_tables(article_id, body)
        for t in article_tables:
            if t.get("table_key") not in table_keys:
                tables_all.append(t)
                table_keys.add(str(t.get("table_key")))

        print(f"[{idx}/{len(clauses)}] Extract {article_id}  section={section!r}  subsection={subsection!r} ...")
        try:
            article_obj = call_claude_for_article(
                client, article_id, body_wo_tables, section=section, subsection=subsection
            )
        except Exception as e:
            article_obj = {
                "article_id": article_id,
                "article_title": "",
                "section": section,
                "subsection": subsection,
                "sentences": [
                    {
                        "sentence_id": f"{article_id}.(?)",
                        "definitions": [],
                        "conditions": [],
                        "requirements": [],
                        "exceptions": [],
                        "references": _empty_refs(),
                        "needs_review": True,
                        "original_text": "",
                        "_error": f"{type(e).__name__}: {e}",
                    }
                ],
            }
            print(f"    Failed: {type(e).__name__}: {e}", file=sys.stderr)

        results.append(article_obj)

        write_json(ck_path, results)
        write_json(tables_ck_path, tables_all)
        write_json(out_path, results)
        write_json(tables_path, tables_all)

        if args.sleep > 0 and idx < len(clauses):
            time.sleep(args.sleep)

    write_json(out_path, results)
    write_json(tables_path, tables_all)
    write_json(ck_path, results)
    write_json(tables_ck_path, tables_all)

    print(f"Done: {len(results)} articles -> {out_path}")
    print(f"Done: {len(tables_all)} tables -> {tables_path}")
    print(f"Checkpoint: {ck_path}")


if __name__ == "__main__":
    main()
