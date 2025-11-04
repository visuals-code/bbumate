"""5. Retrieve relevant documents"""

import os
from typing import List, Optional

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma

from src.ingestion.d003.embedder import get_upstage_embeddings


def get_retriever(k: int = 3):
    """
    Open a persisted Chroma index with Upstage embeddings and return a retriever.
    """
    load_dotenv()
    persist_dir = os.getenv("CHROMA_DB_DIR", "./chroma_storage")

    embeddings = get_upstage_embeddings()
    vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": k})


def retrieve_relevant_documents(query: str, k: int = 3) -> List[Document]:
    """
    Retrieve top-k relevant chunks for the query from the persisted Chroma index.
    """
    retriever = get_retriever(k=k)
    docs: List[Document] = retriever.invoke(query)
    return docs
