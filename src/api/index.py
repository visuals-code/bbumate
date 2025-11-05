"""API index: bridge to expose a simple query handler using d003.

Main server (main.py or dedicated FastAPI modules) can import `query`
to answer questions without depending on domain-specific paths.
"""

from typing import Dict, Any

from src.chains.index import answer_question
from src.generation.index import format_answer_md, format_answer_html, extract_link_info


def query(question: str, k: int = 3) -> Dict[str, Any]:
    answer, docs = answer_question(question, k=k)
    sources = []
    for d in docs:
        title, url, src = extract_link_info(d)
        sources.append({"title": title, "url": url, "source": src})

    return {
        "answer": answer,
        "answer_md": format_answer_md(answer),
        "answer_html": format_answer_html(answer),
        "sources": sources,
    }


__all__ = ["query"]
