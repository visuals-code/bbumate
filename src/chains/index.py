"""Chain index: bridge module to select which domain chain to use.

Currently wired to d003. To switch domain (e.g., d001/d002),
edit imports in this file only.
"""

from src.chains.d003.chain import (
    build_chain,
    answer_question,
    build_llm,
)  # d003 default

__all__ = [
    "build_chain",
    "answer_question",
    "build_llm",
]
