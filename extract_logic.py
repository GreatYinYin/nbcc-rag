"""
读取 part4_clauses.json，逐条调用 Claude API，按 conditions / requirements /
exceptions / references 四类抽取**英文原文**片段（禁止翻译成中文）。
结果写入 part4_logic.json。

需要: pip install anthropic
密钥: 环境变量 ANTHROPIC_API_KEY（推荐），或在下方填写 API_KEY。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

try:
    from anthropic import Anthropic
except ImportError:
    sys.exit("请先安装: python -m pip install anthropic")

DEFAULT_CLAUSES = Path(r"C:\Users\yinli\nbcc_ontology\part4_clauses.json")
DEFAULT_OUTPUT = Path(r"C:\Users\yinli\nbcc_ontology\part4_logic.json")
API_KEY = ""
MODEL = "claude-sonnet-4-6"
MAX_BODY_CHARS = 120_000
MAX_TOKENS = 8192

SYSTEM_PROMPT = """You extract structured excerpts from NBC Part 4 (Canada) clauses.
Rules:
- Copy wording from the user-provided clause text ONLY (English).
- NEVER translate into Chinese or any other language.
- NEVER paraphrase: use verbatim English sentences/clauses/phrases copied from the text (minimal trimming OK only to remove leading [Hn] tags or stray table markers mid-sentence if needed—still English only).
- If nothing applies, output an empty JSON array [] for that field.
- Do not invent cross-references that are absent from the text."""

USER_TEMPLATE = """Analyze the clause below and output ONE JSON object only (no markdown fences).

Fields (each is an array of strings; English verbatim from the clause body; empty [] if none):
- "conditions": Where/When/In this…/Applicability triggers; scope-setting sentences copied in English from the clause.
- "requirements": Mandatory provisions (typically "shall", "must", "required …") quoted in English verbatim from the clause.
- "exceptions": Exceptions/reliefs (e.g., "Except as provided", "Unless", "need not", "may" with conditions) verbatim in English.
- "references": Explicit citations in the clause to other Articles, Sentences, Tables, Figures, Notes, CSA/ASTM standards — copy exactly as appearing (English).

Each array element should be one contiguous excerpt copied from the provided text where possible (not summarized). Omit duplicate excerpts.

Clause identifier (informational only): @CLAUSE_ID@

Clause body:
---
@BODY@
---
"""


def _api_key_from_test_py() -> str | None:
    p = Path(__file__).resolve().parent / "test.py"
    if not p.is_file():
        return None
    m = re.search(r'^API_KEY\s*=\s*"([^"]+)"', p.read_text(encoding="utf-8"), re.MULTILINE)
    return m.group(1).strip() if m else None


def get_api_key() -> str:
    key = (os.environ.get("ANTHROPIC_API_KEY") or API_KEY or "").strip()
    if key and key != "YOUR_ANTHROPIC_API_KEY":
        return key
    key = _api_key_from_test_py()
    if key and key != "YOUR_ANTHROPIC_API_KEY":
        return key
    sys.exit("请设置环境变量 ANTHROPIC_API_KEY，或在 extract_logic.py 中填写 API_KEY（或与 test.py 同目录提供 test.py API_KEY）")


def truncate_body(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n\n[…正文过长已截断，后续内容省略…]"


def extract_json_object(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        while lines and lines[-1].strip() == "```":
            lines.pop()
        text = "\n".join(lines).strip()
    return json.loads(text)


def normalize_logic_payload(raw: dict, clause_id: str) -> dict:
    keys = ("conditions", "requirements", "exceptions", "references")
    out: dict = {"clause_id": clause_id}
    for k in keys:
        v = raw.get(k, [])
        if v is None:
            out[k] = []
        elif isinstance(v, str):
            out[k] = [v] if v.strip() else []
        elif isinstance(v, list):
            out[k] = [str(x).strip() for x in v if str(x).strip()]
        else:
            out[k] = [json.dumps(v, ensure_ascii=False)]
    return out


def call_claude(client: Anthropic, clause_id: str, body: str) -> dict:
    user_msg = USER_TEMPLATE.replace("@CLAUSE_ID@", clause_id).replace(
        "@BODY@", truncate_body(body, MAX_BODY_CHARS)
    )
    msg = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
    )
    text = "".join(
        block.text for block in msg.content if getattr(block, "type", None) == "text"
    )
    parsed = extract_json_object(text)
    return normalize_logic_payload(parsed, clause_id)


def load_checkpoint(path: Path) -> dict[str, dict]:
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {item["clause_id"]: item for item in data if "clause_id" in item}


def main() -> None:
    parser = argparse.ArgumentParser(description="用 Claude 从 part4_clauses.json 抽取逻辑要素")
    parser.add_argument("--input", type=Path, default=DEFAULT_CLAUSES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=0, help="仅处理前 N 条（0 表示全部）")
    parser.add_argument("--sleep", type=float, default=0.35, help="每条请求间隔秒数，缓和限流")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="跳过 checkpoint（part4_logic.checkpoint.json）里已有的 clause_id",
    )
    args = parser.parse_args()

    in_path = args.input.expanduser().resolve()
    out_path = args.output.expanduser().resolve()
    ck_path = out_path.parent / f"{out_path.stem}.checkpoint.json"

    if not in_path.is_file():
        sys.exit(f"找不到输入: {in_path}")

    clauses = json.loads(in_path.read_text(encoding="utf-8"))
    if args.limit > 0:
        clauses = clauses[: args.limit]

    done_map = load_checkpoint(ck_path) if args.resume else {}
    client = Anthropic(api_key=get_api_key())

    results: list[dict] = []
    for i, row in enumerate(clauses):
        cid = row.get("clause_id", f"#{i}")
        body = row.get("body") or ""

        if cid in done_map:
            results.append(done_map[cid])
            print(f"[{i + 1}/{len(clauses)}] 跳过（已缓存） {cid}")
            continue

        print(f"[{i + 1}/{len(clauses)}] 处理 {cid} …")
        try:
            logic = call_claude(client, cid, body)
            logic.pop("_error", None)
        except Exception as e:
            logic = {
                "clause_id": cid,
                "conditions": [],
                "requirements": [],
                "exceptions": [],
                "references": [],
                "_error": f"{type(e).__name__}: {e}",
            }
            print(f"    失败: {logic['_error']}", file=sys.stderr)

        results.append(logic)
        ck_path.write_text(
            json.dumps(results, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        if args.sleep > 0 and i + 1 < len(clauses):
            time.sleep(args.sleep)

    out_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"完成: {len(results)} 条 -> {out_path}")
    if ck_path.is_file():
        print(f"Checkpoint: {ck_path}")


if __name__ == "__main__":
    main()
