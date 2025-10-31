"""persist to chroma database"""

import os
from typing import List, Optional

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma


# RAG 시스템에서 청크된 문서들을 검색 가능한 형태로 영구 저장
def persist_to_chroma(
    documents: List[Document],
    persist_directory: Optional[str] = None,
    embedding_function=None,
) -> None:
    load_dotenv()
    persist_dir = persist_directory or os.getenv("CHROMA_DB_DIR")
    if not embedding_function:
        raise ValueError("embedding_function is required (e.g., UpstageEmbeddings)")

    vectorstore = Chroma.from_documents(
        documents=documents, embedding=embedding_function, persist_directory=persist_dir
    )
    vectorstore.persist()
