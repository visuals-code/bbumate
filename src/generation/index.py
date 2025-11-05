"""Generation index: bridge to generation utilities.

Currently re-exports d003 generation helpers.
Switching domain only requires editing imports here.
"""

from src.generation.d003.prompting import (
    build_chat_prompt,
    format_docs_for_context,
    extract_link_info,
    format_answer_md,
    format_answer_html,
)

__all__ = [
    "build_chat_prompt",
    "format_docs_for_context",
    "extract_link_info",
    "format_answer_md",
    "format_answer_html",
]
