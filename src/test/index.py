"""Test index: simple helper to run a d003 query in-process.

Switch to other domains by editing imports here only.
"""

from typing import Dict, Any

from src.chains.index import answer_question
from src.generation.index import format_answer_md, format_answer_html, extract_link_info


def run(question: str, k: int = 3, output_format: str = "html") -> Dict[str, Any]:
    answer, docs = answer_question(question, k=k)
    sources = []
    for d in docs:
        title, url, src = extract_link_info(d)
        sources.append({"title": title, "url": url, "source": src})

    formatted = (
        format_answer_html(answer)
        if output_format == "html"
        else format_answer_md(answer)
    )
    return {
        "answer": answer,
        "formatted": formatted,
        "answer_md": format_answer_md(answer),
        "answer_html": format_answer_html(answer),
        "sources": sources,
    }


__all__ = ["run"]
