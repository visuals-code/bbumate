from langchain_chroma import Chroma
from langchain_upstage import UpstageEmbeddings
import os
from langchain_core.documents import Document
from dotenv import load_dotenv


class VectorStoreManager:
    @staticmethod
    def save_documents(
        documents: list[Document],
        collection_name=None,
        db_path=None,
        api_key=None,
        embedding_model=None,
        batch_size=3,
    ):
        """문서를 벡터 저장소에 저장"""

        # 환경 변수 로드
        load_dotenv()

        # 환경변수에서 값 가져오기
        api_key = api_key or os.getenv("UPSTAGE_API_KEY")
        db_path = db_path or os.getenv("CHROMA_DB_DIR", "./chroma_storage")
        embedding_model = embedding_model or os.getenv(
            "UPSTAGE_EMBEDDING_MODEL", "solar-embedding-1-large"
        )
        collection_name = collection_name or os.getenv(
            "COLLECTION_NAME", "pdf_subscription_chunks"
        )

        # Upstage Embeddings 초기화
        embeddings = UpstageEmbeddings(api_key=api_key, model=embedding_model)

        # from_documents를 사용하여 벡터 저장소 생성 및 저장
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            collection_name=collection_name,
            persist_directory=db_path,
        )
        print(f"ChromaDB 저장 완료: {db_path}")

        # 저장된 데이터 확인
        collection_data = vectorstore.get()
        print(f"저장된 문서 수: {len(collection_data['ids'])}")

        return vectorstore
