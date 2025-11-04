from langchain_chroma import Chroma
from langchain_upstage import UpstageEmbeddings
import os
from langchain_core.documents import Document


class VectorStoreManager:

    @staticmethod

    # 문서 벡터 저장소에 저장
    def save_documents(
        documents: list[Document],
        collection_name=None,
        db_path="./chroma_storage",
        api_key=None,
        embedding_model=None,
        batch_size=3,
    ):

        api_key = api_key or os.getenv("UPSTAGE_API_KEY")
        db_path = db_path or os.getenv("CHROMA_DB_DIR", "./chroma_storage")
        embedding_model = embedding_model or os.getenv(
            "UPSTAGE_EMBEDDING_MODEL", "solar-embedding-1-large"
        )
        collection_name = collection_name or os.getenv(
            "COLLECTION_NAME", "pdf_promotion_chunks"
        )

        embeddings = UpstageEmbeddings(api_key=api_key, model=embedding_model)

        # from_documents를 사용하여 벡터 저장소 생성 및 저장
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            collection_name=collection_name,
            persist_directory=db_path,
        )

        print(f"ChromaDB 저장 완료: {db_path}")

        collection_data = vectorstore.get()
        print(f" 저장 검증: {len(collection_data['ids'])}개 문서 확인됨")

        # 샘플 출력
        if collection_data["ids"]:
            print(f"   첫 문서 ID: {collection_data['ids'][0]}")
            print(f"   첫 문서 메타데이터: {collection_data['metadatas'][0]}")

        return vectorstore
