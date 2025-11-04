import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_upstage import UpstageEmbeddings
from langchain_core.vectorstores import VectorStoreRetriever
from typing import List


def load_retriever(
    db_path: str = "./chroma_storage",
    collection_name: str = "pdf_promotion_chunks",
    k: int = 3,
    search_type: str = "similarity",
) -> VectorStoreRetriever:

    load_dotenv()
    api_key = os.getenv("UPSTAGE_API_KEY")
    model = os.getenv("UPSTAGE_EMBEDDING_MODEL", "solar-embedding-1-large")

    embeddings = UpstageEmbeddings(api_key=api_key, model=model)

    vectorstore = Chroma(
        persist_directory=db_path,
        embedding_function=embeddings,
        collection_name=collection_name,
    )

    retriever = vectorstore.as_retriever(
        search_type=search_type, search_kwargs={"k": k}
    )

    # vectorstore 참조를 retriever에 추가
    retriever.vectorstore = vectorstore

    return retriever


# 검색된 문서들을 하나의 문자열로 결합 (RAG 체인에서 사용)
def format_docs(docs: List) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


# 쿼리로 문서 검색 및 결과 출력
def search_documents(query: str, k: int = 3, search_type: str = "similarity"):

    # Retriever 로드
    retriever = load_retriever(k=k, search_type=search_type)

    # 검색 실행
    documents = retriever.invoke(query)

    return documents
