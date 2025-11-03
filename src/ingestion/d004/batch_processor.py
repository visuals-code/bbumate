from pathlib import Path
import sys
import os


PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__name__), "..", "..", "..")
)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.ingestion.d004.pdf_processor import process_pdf_to_semantic_chunks
from langchain_core.documents import Document


def process_pdf_directory(pdf_directory: str) -> list[Document]:

    all_documents = []

    base_path = Path(pdf_directory)

    pdf_files = list(base_path.rglob("*.pdf"))

    print(f"총 {len(pdf_files)}개의 PDF 파일이 발견되었습니다.")

    for file_path in pdf_files:
        try:
            chunks = process_pdf_to_semantic_chunks(str(file_path))

            all_documents.extend(chunks)

            print(f"    - 청크 {len(chunks)}개 추출 완료.")

        except Exception as e:
            print(f"    - !!! 오류 발생 ({file_path.name}): {e}")
            print(f"    - 오류 발생 파일 경로: {file_path}")

    return all_documents
