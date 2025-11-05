"""Retrieval index: bridge to retrieval utilities.

Currently exposes d003 retriever functions.
"""

from src.retrieval.d003.retriever import get_retriever, retrieve_relevant_documents

__all__ = [
    "get_retriever",
    "retrieve_relevant_documents",
]
