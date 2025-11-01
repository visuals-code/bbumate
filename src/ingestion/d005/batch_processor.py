from pathlib import Path
import sys
import os


PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__name__), "..", "..", "..")
)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# 이제 절대 경로 임포트를 사용합니다. (cwd 설정에 의존하지 않음)
from src.ingestion.d005.pdf_processor import process_pdf_to_semantic_chunks
from langchain_core.documents import Document


def process_pdf_directory(pdf_directory: str) -> list[Document]:

    all_documents = []

    # 1. 경로를 Path 객체로 변환하여 사용
    base_path = Path(pdf_directory)

    # 2. rglob을 사용하여 디렉토리 내의 모든 *.pdf 파일을 재귀적으로 찾기
    pdf_files = list(base_path.rglob("*.pdf"))

    print(f"총 {len(pdf_files)}개의 PDF 파일이 발견되었습니다.")

    for file_path in pdf_files:
        try:
            chunks = process_pdf_to_semantic_chunks(str(file_path))

            # 처리된 청크들을 최종 리스트에 추가
            all_documents.extend(chunks)

            print(f"    - 청크 {len(chunks)}개 추출 완료.")

        except Exception as e:
            print(f"    - !!! 오류 발생 ({file_path.name}): {e}")
            print(f"    - 오류 발생 파일 경로: {file_path}")

    return all_documents
