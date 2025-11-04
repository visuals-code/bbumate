"""문서 텍스트 분할 모듈."""

from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(documents: List[Document]) -> List[Document]:
    """로드된 문서를 검색에 적합한 크기의 청크로 분할합니다.

    Args:
        documents: 분할할 문서 리스트.

    Returns:
        분할된 청크 리스트.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = text_splitter.split_documents(documents)
    print(f"✅ 총 {len(chunks)}개의 청크로 분할되었습니다.")
    return chunks
