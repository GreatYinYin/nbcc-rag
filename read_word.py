"""
用 python-docx 读取 Word，列出标题层级与正文（默认只看前 100 条）。
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

try:
    import docx
except ImportError:
    sys.exit("请先安装: python -m pip install python-docx")

DEFAULT_DOC_PATH = Path(r"C:\Users\yinli\nbcc_ontology\NBC 2020 Volume 1-2.docx")

HEADING_STYLE_RE = re.compile(r"(?i)^heading\s*(\d+)\s*$")


def paragraph_heading_level(paragraph: docx.text.paragraph.Paragraph) -> int | None:
    name = (paragraph.style.name or "").strip()
    m = HEADING_STYLE_RE.match(name)
    if m:
        return int(m.group(1))
    return None


def main() -> None:
    doc_path = DEFAULT_DOC_PATH.resolve()
    limit = 100

    if len(sys.argv) >= 2:
        doc_path = Path(sys.argv[1]).expanduser().resolve()
    if len(sys.argv) >= 3:
        limit = max(1, int(sys.argv[2]))

    if not doc_path.is_file():
        sys.exit(f"找不到文件: {doc_path}")

    document = docx.Document(str(doc_path))
    count = 0

    print(f"文件: {doc_path.name}")
    print(f"最多列出前 {limit} 个标题\n")

    for para in document.paragraphs:
        level = paragraph_heading_level(para)
        if level is None:
            continue
        text = (para.text or "").strip()
        if not text:
            continue
        count += 1
        indent = "  " * (level - 1)
        print(f"[H{level}] {indent}{text}")
        if count >= limit:
            print(f"\n…已达到上限 {limit} 条，其余标题省略。")
            break

    if count == 0:
        print(
            "未识别到内置 Heading 1–9 样式。"
            "若该文档用自定义样式做标题，需要在 paragraph_heading_level() 里补充规则。"
        )


if __name__ == "__main__":
    main()
