import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# 프로젝트 루트 경로 추가
PROJECT_ROOT = Path(__name__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.d004.batch_processor import process_pdf_directory
from src.ingestion.d004.vectorstore_manager import VectorStoreManager


def main():
    """전체 파이프라인 실행"""
    load_dotenv()

    # 설정
    PDF_DIR = os.getenv("PDF_DIR", "./data")
    CHROMA_DB_DIR = os.getenv("CHROMA_DB_DIR", "./chroma_storage")
    COLLECTION_NAME = os.getenv("COLLECTION_NAME", "pdf_subscription_chunks")

    print("=" * 60)
    print("D005 Ingestion Pipeline 시작")
    print("=" * 60)

    # 1단계: PDF 디렉토리 처리
    print(f"\n[1/2] PDF 파일 처리 중... (경로: {PDF_DIR})")
    documents = process_pdf_directory(PDF_DIR)

    if not documents:
        print("[경고] 처리된 문서가 없습니다.")
        return

    print(f"✓ 총 {len(documents)}개의 청크 생성 완료")

    # 2단계: ChromaDB에 저장
    print(f"\n[2/2] ChromaDB에 저장 중... (경로: {CHROMA_DB_DIR})")
    VectorStoreManager.save_documents(
        documents=documents, collection_name=COLLECTION_NAME, db_path=CHROMA_DB_DIR
    )

    print("\n" + "=" * 60)
    print("✓ 파이프라인 완료!")
    print(f"  - 저장 경로: {CHROMA_DB_DIR}")
    print(f"  - 컬렉션명: {COLLECTION_NAME}")
    print(f"  - 문서 수: {len(documents)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
