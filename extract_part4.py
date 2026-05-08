"""
从 NBC Word 文档中提取 Part 4 正文（按文档顺序含表格），写入 part4.txt。
列出所有标题中含「Part 4」的条目；正文从 Division B 的「Part 4」起（若存在两个 [H1] Part 4，
则使用第二个，以跳过 Division A 目录），直到下一个「Part <数字>」（数字≠4）标题为止。
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

try:
    import docx
    from docx.table import Table
    from docx.text.paragraph import Paragraph
    from docx.oxml.ns import qn
except ImportError:
    sys.exit("请先安装: python -m pip install python-docx")

DEFAULT_DOC_PATH = Path(r"C:\Users\yinli\nbcc_ontology\NBC 2020 Volume 1-2.docx")
OUTPUT_PATH = Path(r"C:\Users\yinli\nbcc_ontology\part4.txt")

HEADING_STYLE_RE = re.compile(r"(?i)^heading\s*(\d+)\s*$")
PART_4_START_RE = re.compile(r"(?is)^\s*part\s+4\b")
NEXT_MAJOR_PART_RE = re.compile(r"(?is)^\s*part\s+(\d+)\b")


def paragraph_heading_level(paragraph: Paragraph) -> int | None:
    name = (paragraph.style.name or "").strip()
    m = HEADING_STYLE_RE.match(name)
    if m:
        return int(m.group(1))
    return None


def iter_body_blocks(doc):
    for child in doc.element.body.iterchildren():
        if child.tag == qn("w:p"):
            yield Paragraph(child, doc)
        elif child.tag == qn("w:tbl"):
            yield Table(child, doc)


def paragraph_lines(paragraph: Paragraph) -> list[str]:
    text = (paragraph.text or "").strip()
    if not text:
        return []
    lev = paragraph_heading_level(paragraph)
    if lev is not None:
        return [f"[H{lev}] {text}"]
    return [text]


def table_lines(table: Table) -> list[str]:
    lines: list[str] = []
    for row in table.rows:
        cells = [(c.text or "").replace("\n", " ").strip() for c in row.cells]
        lines.append("\t".join(cells))
    return lines


def block_lines(block: Paragraph | Table) -> list[str]:
    if isinstance(block, Paragraph):
        return paragraph_lines(block)
    return ["[TABLE]", *table_lines(block), "[/TABLE]"]


def headings_containing_part_4(doc) -> list[str]:
    found: list[str] = []
    for block in iter_body_blocks(doc):
        if not isinstance(block, Paragraph):
            continue
        lev = paragraph_heading_level(block)
        if lev is None:
            continue
        text = (block.text or "").strip()
        if not text:
            continue
        if "part 4" in text.lower():
            found.append(f"[H{lev}] {text}")
    return found


def find_part4_start_index(blocks: list[Paragraph | Table]) -> int | None:
    """定位 Part 4 正文起点：NBC 合并卷里第一个 [H1] Part 4 常为 Division A 目录，第二个起为 Division B。"""
    h1_part4: list[int] = []
    for i, block in enumerate(blocks):
        if not isinstance(block, Paragraph):
            continue
        lev = paragraph_heading_level(block)
        if lev != 1:
            continue
        text = (block.text or "").strip()
        if PART_4_START_RE.match(text):
            h1_part4.append(i)

    if h1_part4:
        return h1_part4[1] if len(h1_part4) >= 2 else h1_part4[0]

    for i, block in enumerate(blocks):
        if not isinstance(block, Paragraph):
            continue
        lev = paragraph_heading_level(block)
        if lev is None:
            continue
        text = (block.text or "").strip()
        if "part 4" in text.lower():
            return i
    return None


def is_next_major_part_heading(paragraph: Paragraph) -> bool:
    lev = paragraph_heading_level(paragraph)
    if lev is None:
        return False
    text = (paragraph.text or "").strip()
    m = NEXT_MAJOR_PART_RE.match(text)
    if not m:
        return False
    return m.group(1) != "4"


def main() -> None:
    doc_path = DEFAULT_DOC_PATH.resolve()
    out_path = OUTPUT_PATH.resolve()

    if len(sys.argv) >= 2:
        doc_path = Path(sys.argv[1]).expanduser().resolve()
    if len(sys.argv) >= 3:
        out_path = Path(sys.argv[2]).expanduser().resolve()

    if not doc_path.is_file():
        sys.exit(f"找不到文件: {doc_path}")

    doc = docx.Document(str(doc_path))
    blocks = list(iter_body_blocks(doc))

    heading_index = headings_containing_part_4(doc)

    start = find_part4_start_index(blocks)
    if start is None:
        sys.exit("未找到包含「Part 4」的标题段落。")

    end = len(blocks)
    for j in range(start + 1, len(blocks)):
        b = blocks[j]
        if isinstance(b, Paragraph) and is_next_major_part_heading(b):
            end = j
            break

    body_chunks: list[str] = []
    for block in blocks[start:end]:
        body_chunks.extend(block_lines(block))

    header_lines = [
        "# NBC — Part 4 提取",
        f"# 来源: {doc_path}",
        "",
        "## 所有标题中含「Part 4」的条目（索引 / 层级）",
    ]
    if heading_index:
        for item in heading_index:
            header_lines.append(f"- {item}")
    else:
        header_lines.append("- （无）")
    header_lines.extend(
        [
            "",
            f"## 正文范围: 文档块 {start + 1}–{end}（含起始标题，止于下一 Part）",
            "",
        ]
    )

    text_out = "\n".join(header_lines + body_chunks) + "\n"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text_out, encoding="utf-8")

    print(f"已写入: {out_path}")
    print(f"「Part 4」相关标题共 {len(heading_index)} 条；正文块数 {end - start}")


if __name__ == "__main__":
    main()
