"""2. Chunking text"""

from typing import List
from bs4 import BeautifulSoup
from langchain_core.documents import Document


def html_to_text(html: str) -> str:
    """
    Convert HTML to plain text using BeautifulSoup.
    HTML을 일반 텍스트로 변환하는 전처리 함수:
    BeautifulSoup로 HTML 태그를 제거하고 텍스트만 추출해 간단한 문자열로 반환
    """
    soup = BeautifulSoup(html or "", "html.parser")
    text = soup.get_text(separator="\n")
    lines = [line.strip() for line in text.splitlines()]
    compact = "\n".join(line for line in lines if line)
    return compact


def split_documents(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    strip_html: bool = False,
) -> List[Document]:
    """
    Split Documents into smaller overlapping chunks.
    Documents를 작은 조각(chunks)으로 분할하는 함수:
     - strip_html=True이면, HTML 태그를 제거하고 순수 텍스트만 추출
     - strip_html=False이면, 원본 텍스트를 그대로 사용
    """
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    chunked_docs: List[Document] = []

    for doc_index, doc in enumerate(documents):
        raw_text = doc.page_content or ""
        text = html_to_text(raw_text) if strip_html else raw_text
        start = 0
        chunk_index = 0

        # start 위치에서 chunk_size만큼 잘라내서 chunk_text로 저장
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_text = text[start:end]

            metadata = dict(doc.metadata or {})
            metadata.update(
                {
                    "chunk_index": chunk_index,
                    "chunk_start": start,
                    "chunk_end": end,
                    "original_doc_index": doc_index,
                }
            )

            chunked_docs.append(Document(page_content=chunk_text, metadata=metadata))

            if end == len(text):
                break
            start = end - chunk_overlap  # overlap만큼 뒤로 이동
            chunk_index += 1

    return chunked_docs
