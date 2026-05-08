"""
读取 part4.txt，按条款号切分为独立条目，写入 part4_clauses.json。

条款起点：
- Division B：[H6] 行内的 4.x.x.x（与 extract_part4.py 导出一致）
- 附录注释：行首或 [H6] 中的 A-4... 样式编号

同一 clause_id 连续出现的片段会合并为一条。
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

DEFAULT_INPUT = Path(r"C:\Users\yinli\nbcc_ontology\part4.txt")
DEFAULT_OUTPUT = Path(r"C:\Users\yinli\nbcc_ontology\part4_clauses.json")

# [H6] 4.1.2.1.	Division B
RE_H6_ARTICLE_FIRST = re.compile(
    r"^\[H6\]\s+(4\.\d+\.\d+\.\d+)\.\s+Division B",
)
# [H6] Division B	4.1.3.1.
RE_H6_DIVB_ARTICLE = re.compile(
    r"^\[H6\]\s+Division B\s+(4\.\d+\.\d+\.\d+)\.",
)
# [H9] 4.1.6.16.	Roofs with Solar Panels（少数 Article 仅用 H9 标记）
RE_H9_ARTICLE = re.compile(
    r"^\[H9\]\s+(4\.\d+\.\d+\.\d+)\.",
)
# [H6] A-4.1.2.2.(1)	Division B
RE_H6_APPENDIX = re.compile(
    r"^\[H6\]\s+(A-4(?:\.\d+)+(?:\.\(\d+\))?)\.?\s+Division B",
)
# 行首附录：A-4.1.2.1.(1)	...
RE_LINE_APPENDIX = re.compile(
    r"^(A-4(?:\.\d+)+(?:\.\(\d+\))?)\.?\s+",
)


def normalize_id(raw: str) -> str:
    return raw.rstrip(".").strip()


def parse_clause_start(line: str) -> str | None:
    m = RE_H6_APPENDIX.match(line)
    if m:
        return normalize_id(m.group(1))
    m = RE_H6_ARTICLE_FIRST.match(line)
    if m:
        return normalize_id(m.group(1))
    m = RE_H6_DIVB_ARTICLE.match(line)
    if m:
        return normalize_id(m.group(1))
    m = RE_H9_ARTICLE.match(line)
    if m:
        return normalize_id(m.group(1))
    m = RE_LINE_APPENDIX.match(line)
    if m:
        return normalize_id(m.group(1))
    return None


def strip_header(lines: list[str]) -> list[str]:
    """去掉 # 注释头，从 [H1] Part 4 起保留正文。"""
    out: list[str] = []
    started = False
    for ln in lines:
        if not started:
            if ln.startswith("[H1]") and "Part 4" in ln:
                started = True
            else:
                continue
        out.append(ln.rstrip("\n"))
    return out


def flush_buffer(
    clauses: list[dict],
    clause_id: str,
    buf: list[str],
) -> None:
    text = "\n".join(buf).strip()
    if not text:
        return
    if clauses and clauses[-1]["clause_id"] == clause_id:
        clauses[-1]["body"] = clauses[-1]["body"] + "\n\n" + text
    else:
        clauses.append({"clause_id": clause_id, "body": text})


def split_into_clauses(lines: list[str]) -> list[dict]:
    body = strip_header(lines)
    if not body:
        return []

    clauses: list[dict] = []
    current_id = "(preamble)"
    buf: list[str] = []

    for line in body:
        new_id = parse_clause_start(line)
        if new_id is not None:
            flush_buffer(clauses, current_id, buf)
            current_id = new_id
            buf = [line]
        else:
            buf.append(line)

    flush_buffer(clauses, current_id, buf)
    return clauses


def main() -> None:
    in_path = DEFAULT_INPUT.resolve()
    out_path = DEFAULT_OUTPUT.resolve()
    if len(sys.argv) >= 2:
        in_path = Path(sys.argv[1]).expanduser().resolve()
    if len(sys.argv) >= 3:
        out_path = Path(sys.argv[2]).expanduser().resolve()

    if not in_path.is_file():
        sys.exit(f"找不到输入文件: {in_path}")

    lines = in_path.read_text(encoding="utf-8").splitlines()
    clauses = split_into_clauses(lines)

    out_path.write_text(
        json.dumps(clauses, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"已写入 {len(clauses)} 条记录 -> {out_path}")


if __name__ == "__main__":
    main()
