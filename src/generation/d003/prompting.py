"""6. Prompting: define the system+human prompt and formatting helpers"""

from typing import List, Tuple
import os
import html
import re

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate


def format_docs_for_context(documents: List[Document], max_chars: int = 1200) -> str:
    """
    Build a compact context string from retrieved documents.
    Each entry includes a source hint and a truncated content snippet.
    """
    lines: List[str] = []
    for idx, doc in enumerate(documents, start=1):
        source = (doc.metadata or {}).get("source", "unknown")
        content = doc.page_content or ""
        snippet = content[:max_chars].strip()
        lines.append(f"[{idx}] source: {source}\n{snippet}")
    return "\n\n".join(lines)


def build_chat_prompt() -> ChatPromptTemplate:
    """Define the system+human prompt that includes context and question placeholders."""
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "주어진 컨텍스트를 이용해 사용자 질문에 간결하고 정확하게 답하세요.\n"
                "모르면 모른다고 답하세요.\n"
                "컨텍스트:\n{context}",
            ),
            ("human", "질문: {question}"),
        ]
    )


# ---- Formatting helpers for final answers ----


def extract_link_info(doc: Document) -> Tuple[str, str | None, str]:
    meta = doc.metadata or {}
    src = meta.get("source", "unknown")
    title = meta.get("title") or os.path.basename(str(src)) or "unknown"
    url = meta.get("url") or meta.get("source_url")
    return title, url, src


def insert_line_breaks_korean(text: str) -> str:
    text = re.sub(r"(다|요|니다)\.(\s+)", r"\1.\n\n", text)
    text = re.sub(r"\s+(또한|그리고|한편|추가로|더불어|다만|참고로),", r"\n\n\1,", text)
    return text


def emphasize_price_terms_md(text: str) -> str:
    patterns = [
        r"(?:(?:매?월|연)\s*)?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?\s*원",
        r"\d+(?:\.\d+)?\s*%",
    ]

    def repl(m: re.Match) -> str:
        return f"**{m.group(0)}**"

    for p in patterns:
        text = re.sub(p, repl, text)
    return text


def emphasize_price_terms_html_escaped(escaped_text: str) -> str:
    patterns = [
        r"(?:(?:매?월|연)\s*)?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?\s*원",
        r"\d+(?:\.\d+)?\s*%",
    ]

    def repl(m: re.Match) -> str:
        return f"<strong>{m.group(0)}</strong>"

    for p in patterns:
        escaped_text = re.sub(p, repl, escaped_text)
    return escaped_text


def format_answer_md(answer: str) -> str:
    prepared = insert_line_breaks_korean(answer.strip())
    return emphasize_price_terms_md(prepared)


def format_answer_html(answer: str) -> str:
    prepared = insert_line_breaks_korean(answer.strip())
    escaped = html.escape(prepared)
    emphasized = emphasize_price_terms_html_escaped(escaped)
    return "<div>" + emphasized.replace("\n", "<br/>") + "</div>"
