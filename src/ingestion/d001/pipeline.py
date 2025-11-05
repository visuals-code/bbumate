"""PDF 문서를 로드하고 Upstage 임베딩을 사용하여 Chroma DB에 저장하는 데이터 수집 파이프라인."""

import time
from pathlib import Path

from dotenv import load_dotenv

from .pdf_loader import load_pdfs
from .text_splitter import split_documents
from .vector_store import store_in_chroma, verify_storage

# 전역 설정
PDF_DIRECTORY = Path("data/d001/housing")  # PDF 파일들이 있는 디렉토리 경로


def main_pipeline() -> None:
    """PDF 로드부터 벡터 DB 저장까지 전체 파이프라인을 실행합니다.

    Returns:
        None. 파이프라인 실행 결과를 콘솔에 출력합니다.
    """
    # 전체 파이프라인 시작 시간 측정
    pipeline_start_time = time.time()

    # 1. `.env` 파일 로드
    load_dotenv()
    print("`.env` 파일의 환경 변수를 로드했습니다.")

    # 2. PDF 디렉토리 존재 확인
    if not PDF_DIRECTORY.exists():
        print(f"PDF 파일을 '{PDF_DIRECTORY}' 안에 넣은 후 다시 실행해 주세요.")
        return

    # 3. PDF 로드
    documents = load_pdfs(PDF_DIRECTORY)

    if not documents:
        print("로드할 문서(PDF 파일)가 디렉토리 내에 없거나 로드에 실패했습니다.")
        return

    # 4. 텍스트 분할
    chunks = split_documents(documents)

    # 5. 벡터 DB 저장
    vector_db = store_in_chroma(chunks)

    if vector_db:
        # 전체 파이프라인 종료 시간 측정
        pipeline_end_time = time.time()
        total_elapsed_time = pipeline_end_time - pipeline_start_time

        print("\n" + "="*60)
        print("--- 파이프라인 실행 완료 ---")
        print(f"[전체 파이프라인] 총 소요 시간: {total_elapsed_time:.2f}초 ({total_elapsed_time/60:.2f}분)")
        print("="*60)

        # 6. 저장 확인 함수 호출
        verify_storage()


if __name__ == "__main__":
    main_pipeline()
