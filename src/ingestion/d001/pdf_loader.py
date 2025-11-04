"""PDF 문서 로더 모듈."""

from pathlib import Path
from typing import List

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_core.documents import Document


def load_pdfs(directory_path: Path) -> List[Document]:
    """지정된 디렉토리에서 모든 PDF 파일을 로드합니다.

    Args:
        directory_path: PDF 파일이 있는 디렉토리 경로.

    Returns:
        로드된 문서 리스트. 로드 실패 시 빈 리스트 반환.
    """
    print(f"\n디렉토리: '{directory_path}'에서 PDF 로드를 시작합니다...")

    try:
        # DirectoryLoader는 문자열 경로를 받으므로 str()로 변환
        loader = DirectoryLoader(
            path=str(directory_path),
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,
            show_progress=True,
        )

        documents = loader.load()
        print(f"✅ 총 {len(documents)}개의 문서 페이지/청크를 로드했습니다.")
        return documents
    except FileNotFoundError:
        print(
            f"❌ 파일 로드 중 FileNotFoundError 발생: "
            f"PDF 파일을 {directory_path}에 넣어주세요."
        )
        return []
