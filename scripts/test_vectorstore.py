"""벡터 저장소 테스트"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv


project_root = Path.cwd()
sys.path.insert(0, str(project_root))

from src.ingestion import VectorStoreManager


def test_vectorstore_search():

    env_path = project_root / ".env"
    if not env_path.exists():
        print(f"⚠️  .env 파일을 찾을 수 없습니다: {env_path}")
    else:
        load_dotenv(dotenv_path=env_path)
        print(f"✅ .env 파일 로드: {env_path}")

    # 설정
    PDF_DIRECTORY = project_root / "data" / "subscription"
    DB_PATH = project_root / "chroma_storage"
    COLLECTION_NAME = "pdf_subscription_chunks"

    print("=" * 60)
    print("벡터 저장소 검색 테스트")
    print("=" * 60)
    print(f"\n[경로 확인]")
    print(f"  - 프로젝트 루트: {project_root}")
    print(f"  - PDF 디렉토리: {PDF_DIRECTORY}")
    print(f"  - DB 경로: {DB_PATH}")
    print(f"  - 컬렉션: {COLLECTION_NAME}")

    # PDF 디렉토리 존재 확인
    if not PDF_DIRECTORY.exists():
        print(f"❌ PDF 디렉토리가 존재하지 않습니다: {PDF_DIRECTORY}")
        return False
    else:
        pdf_files = list(PDF_DIRECTORY.glob("*.pdf"))
        print(f"✅ PDF 디렉토리 존재 (PDF 파일 {len(pdf_files)}개)")

    # ChromaDB 디렉토리 존재 확인
    if not DB_PATH.exists():
        print(f"❌ ChromaDB 디렉토리가 존재하지 않습니다: {DB_PATH}")
        print(f"💡 먼저 ingest_pdfs.py를 실행하여 PDF를 벡터 저장소에 저장하세요.")
        return False
    else:
        print(f"✅ ChromaDB 디렉토리 존재")

    # VectorStoreManager 초기화
    print(f"\n[초기화] 벡터 저장소 로드")

    try:
        manager = VectorStoreManager(db_path=str(DB_PATH))
        vectorstore = manager.load_vectorstore(collection_name=COLLECTION_NAME)

        # 저장된 문서 수 확인
        collection_data = vectorstore.get()
        doc_count = len(collection_data["ids"])

        if doc_count == 0:
            print(f"⚠️  벡터 저장소가 비어있습니다!")
            print(f"💡 먼저 ingest_pdfs.py를 실행하여 PDF를 처리하세요.")
            return False

        print(f"✅ 로드 완료 (문서 수: {doc_count})")

    except Exception as e:
        print(f"❌ 벡터 저장소 로드 실패: {e}")
        import traceback

        traceback.print_exc()
        return False

    # 테스트 쿼리 목록
    test_queries = [
        "특별공급 신청 자격",
        "생애최초 특별공급 조건",
        "다자녀 가구 특별공급",
        "신혼부부 특별공급",
        "노부모 부양 특별공급",
        "청약 당첨자 발표",
    ]

    print(f"\n[검색 테스트] {len(test_queries)}개 쿼리 실행")
    print("-" * 60)

    for i, query in enumerate(test_queries, 1):
        print(f"\n[쿼리 {i}] '{query}'")
        try:
            results = vectorstore.similarity_search(query, k=3)
            print(f"  결과: {len(results)}개 문서")

            if not results:
                print(f"  ⚠️  검색 결과가 없습니다.")
                continue

            for j, doc in enumerate(results, 1):
                heading = doc.metadata.get("heading", "N/A")
                source = Path(doc.metadata.get("source_file", "N/A")).name
                preview = doc.page_content[:80].replace("\n", " ")

                print(f"    [{j}] 제목: {heading[:40]}...")
                print(f"        출처: {source}")
                print(f"        내용: {preview}...")

        except Exception as e:
            print(f"  ❌ 검색 실패: {e}")
            import traceback

            traceback.print_exc()

    print("\n" + "=" * 60)
    print("✅ 검색 테스트 완료!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = test_vectorstore_search()
    sys.exit(0 if success else 1)
