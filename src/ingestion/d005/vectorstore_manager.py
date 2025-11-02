# pip install langchain-chroma

from langchain_chroma import Chroma
from langchain_upstage import UpstageEmbeddings
import os
from langchain_core.documents import Document


class VectorStoreManager:
    # 초기화
    def __init__(self, db_path="./chroma_storage", api_key=None):
        self.db_path = db_path
        self.api_key = api_key or os.getenv("UPSTAGE_API_KEY")

        # Upstage Embeddings 초기화
        self.embeddings = UpstageEmbeddings(
            api_key=self.api_key, model="solar-embedding-1-large-passage"
        )

    # 문서 벡터 저장소에 저장
    def save_documents(
        self,
        documents: list[Document],
        collection_name="pdf_promotion_chunks",
        batch_size=3,
    ):

        # from_documents를 사용하여 벡터 저장소 생성 및 저장
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            collection_name=collection_name,
            persist_directory=self.db_path,
        )
        print(f"✅ ChromaDB 저장 완료: {self.db_path}")

        # 저장된 데이터 확인
        collection_data = vectorstore.get()
        print(f"저장된 문서 수: {len(collection_data['ids'])}")

        return vectorstore
