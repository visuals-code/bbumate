"""6. Prompting: define the system+human prompt"""

from typing import List

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
