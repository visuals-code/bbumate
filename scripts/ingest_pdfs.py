import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# 프로젝트 루트를 Python path에 추가
project_root = Path.cwd()
sys.path.insert(0, str(project_root))

from src.ingestion.d004.batch_processor import process_pdf_directory
from src.ingestion.d004.vectorstore_manager import VectorStoreManager


def main():
    # 환경 변수 로드
    env_path = project_root / ".env"

    print(f"DEBUG: .env Path: {env_path}")

    if not env_path.exists():
        print(f"⚠️  .env 파일을 찾을 수 없습니다: {env_path}")
        print(f"💡 .env 파일에 UPSTAGE_API_KEY를 설정하세요.")
        return False

    load_dotenv(dotenv_path=env_path)

    if not os.getenv("UPSTAGE_API_KEY"):
        print("❌ UPSTAGE_API_KEY가 설정되지 않았습니다.")
        return False
    else:
        print("✅ UPSTAGE_API_KEY가 환경 변수에서 성공적으로 로드되었습니다.")

    print(f"✅ .env 파일 로드: {env_path}\n")

    # 설정
    PDF_DIRECTORY = project_root / "data" / "d004"
    DB_PATH = project_root / "chroma_storage"
    COLLECTION_NAME = "pdf_subscription_chunks"
    BATCH_SIZE = 3

    print("=" * 60)
    print("PDF 파일 벡터화 및 저장")
    print("=" * 60)
    print(f"\n[설정]")
    print(f"  - PDF 디렉토리: {PDF_DIRECTORY}")
    print(f"  - DB 경로: {DB_PATH}")
    print(f"  - 컬렉션: {COLLECTION_NAME}")
    print(f"  - 배치 크기: {BATCH_SIZE}")

    # PDF 디렉토리 확인
    if not PDF_DIRECTORY.exists():
        print(f"\n❌ PDF 디렉토리가 존재하지 않습니다: {PDF_DIRECTORY}")
        return False

    pdf_files = list(PDF_DIRECTORY.rglob("*.pdf"))

    if not pdf_files:
        print(f"\n❌ PDF 파일이 없습니다: {PDF_DIRECTORY} 및 하위 폴더.")
        return False

    print(f"\n✅ PDF 파일 {len(pdf_files)}개 발견")

    # 1단계: PDF 처리
    print("\n" + "=" * 60)
    print("1단계: PDF 파일 처리 (의미론적 청킹)")
    print("=" * 60)

    try:
        # process_pdf_directory 함수는 str 경로를 요구하므로 str()로 형변환
        documents = process_pdf_directory(str(PDF_DIRECTORY))

        if not documents:
            print("❌ 처리된 문서가 없습니다.")
            return False

        print(f"\n✅ 총 {len(documents)}개의 문서 청크 생성 완료")

    except Exception as e:
        print(f"❌ PDF 처리 중 오류 발생: {e}")
        import traceback

        traceback.print_exc()
        return False

    # 2단계: 벡터 저장소에 저장
    print("\n" + "=" * 60)
    print("2단계: ChromaDB에 저장")
    print("=" * 60)

    try:
        manager = VectorStoreManager(db_path=str(DB_PATH))

        vectorstore = manager.save_documents(
            documents=documents, collection_name=COLLECTION_NAME, batch_size=BATCH_SIZE
        )

        # 저장 확인
        collection_data = vectorstore.get()
        print(f"\n✅ 최종 저장된 문서 수: {len(collection_data['ids'])}")

    except Exception as e:
        print(f"❌ 벡터 저장소 저장 중 오류 발생: {e}")
        import traceback

        traceback.print_exc()
        return False

    # 3단계: 간단한 검색 테스트
    print("\n" + "=" * 60)
    print("3단계: 검색 테스트")
    print("=" * 60)

    try:
        test_query = "특별공급 신청 자격"
        print(f"\n테스트 쿼리: '{test_query}'")

        results = vectorstore.similarity_search(test_query, k=3)
        print(f"검색 결과: {len(results)}개\n")

        for i, doc in enumerate(results, 1):
            heading = doc.metadata.get("heading", "N/A")
            source = Path(doc.metadata.get("source_file", "N/A")).name
            preview = doc.page_content[:60].replace("\n", " ")

            print(f"[{i}] {heading[:40]}...")
            print(f"    출처: {source}")
            print(f"    내용: {preview}...\n")

    except Exception as e:
        print(f"⚠️  검색 테스트 실패: {e}")

    print("=" * 60)
    print("✅ 모든 작업 완료!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
