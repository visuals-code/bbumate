"""문서 포맷팅 유틸리티.

검색된 문서를 프롬프트에 전달할 형식으로 변환합니다.
"""

import re
from typing import Dict, List, Optional, Tuple

from langchain_core.documents import Document


def format_docs(docs: List[Document]) -> str:
    """검색된 문서를 프롬프트에 전달할 수 있는 형식으로 변환.

    문서의 내용과 출처 정보를 포함하여 포맷팅합니다.

    Args:
        docs: 검색된 Document 객체 리스트

    Returns:
        str: 포맷팅된 문서 내용 (출처 포함)

    Examples:
        >>> docs = [Document(page_content="내용", metadata={"source": "test.pdf"})]
        >>> format_docs(docs)
        '출처: test.pdf\\n내용: 내용'
    """
    if not docs:
        return "검색된 문서가 없습니다."

    formatted_parts = []
    for doc in docs:
        source = doc.metadata.get('source', '알 수 없음')
        content = doc.page_content
        formatted_parts.append(f"출처: {source}\n내용: {content}")

    return "\n\n".join(formatted_parts)


def format_docs_with_scores(docs_with_scores: List[Tuple[Document, float]]) -> str:
    """유사도 점수를 포함하여 문서 포맷팅.

    Args:
        docs_with_scores: (Document, similarity_score) 튜플 리스트

    Returns:
        str: 점수가 포함된 포맷팅된 문서
    """
    if not docs_with_scores:
        return "검색된 문서가 없습니다."

    formatted_parts = []
    for doc, score in docs_with_scores:
        source = doc.metadata.get('source', '알 수 없음')
        content = doc.page_content
        formatted_parts.append(
            f"출처: {source} (유사도: {score:.4f})\n내용: {content}"
        )

    return "\n\n".join(formatted_parts)


def format_answer_to_markdown(answer: str) -> str:
    """답변을 마크다운 형식으로 변환합니다.

    - 금액과 비율을 굵게 처리 (예: **1,000만 원**, **50%**)
    - 문장 단위로 줄바꿈 적용

    Args:
        answer: 원본 답변 텍스트

    Returns:
        str: 마크다운 형식의 답변

    Examples:
        >>> format_answer_to_markdown("최대 3900만 원을 지원합니다. 통상임금의 100%입니다.")
        "최대 **3900만 원**을 지원합니다.\\n\\n통상임금의 **100%**입니다."
    """
    # 금액 패턴: 숫자 + 만 원, 억 원, 천 원 등
    money_pattern = r'(\d{1,3}(?:,?\d{3})*(?:만|억|천|백)?\s*원)'
    # 비율 패턴: 숫자 + %
    percentage_pattern = r'(\d+(?:\.\d+)?%)'

    # 금액을 굵게 처리
    formatted = re.sub(money_pattern, r'**\1**', answer)
    # 비율을 굵게 처리
    formatted = re.sub(percentage_pattern, r'**\1**', formatted)

    # 문장 단위 줄바꿈 (마침표, 물음표, 느낌표 뒤에 줄바꿈)
    formatted = re.sub(r'([.!?])\s+', r'\1\n\n', formatted)

    return formatted.strip()


def format_answer_to_html(answer: str) -> str:
    """답변을 HTML 형식으로 변환합니다.

    - 금액과 비율을 <strong> 태그로 감쌈
    - 줄바꿈을 <br/> 태그로 변환
    - 전체를 <div> 태그로 감쌈

    Args:
        answer: 원본 답변 텍스트

    Returns:
        str: HTML 형식의 답변

    Examples:
        >>> format_answer_to_html("최대 3900만 원을 지원합니다. 통상임금의 100%입니다.")
        "<div>최대 <strong>3900만 원</strong>을 지원합니다.<br/><br/>통상임금의 <strong>100%</strong>입니다.</div>"
    """
    # 금액 패턴
    money_pattern = r'(\d{1,3}(?:,?\d{3})*(?:만|억|천|백)?\s*원)'
    # 비율 패턴
    percentage_pattern = r'(\d+(?:\.\d+)?%)'

    # 금액을 <strong>으로 감싸기
    formatted = re.sub(money_pattern, r'<strong>\1</strong>', answer)
    # 비율을 <strong>으로 감싸기
    formatted = re.sub(percentage_pattern, r'<strong>\1</strong>', formatted)

    # 문장 단위 줄바꿈을 <br/>로 변환
    formatted = re.sub(r'([.!?])\s+', r'\1<br/><br/>', formatted)

    # div로 감싸기
    return f"<div>{formatted.strip()}</div>"


def extract_sources_from_docs(docs: List[Document]) -> List[Dict[str, Optional[str]]]:
    """검색된 문서에서 source 정보를 추출합니다.

    Args:
        docs: 검색된 Document 객체 리스트

    Returns:
        List[Dict[str, Optional[str]]]: source 정보 리스트
            각 항목은 {"title": str, "url": str|None, "source": str} 형식

    Examples:
        >>> docs = [Document(
        ...     page_content="내용",
        ...     metadata={
        ...         "source": "data/d003/report.pdf",
        ...         "title": "보고서 제목",
        ...         "url": "https://example.com"
        ...     }
        ... )]
        >>> extract_sources_from_docs(docs)
        [{"title": "보고서 제목", "url": "https://example.com", "source": "data/d003/report.pdf"}]
    """
    sources = []

    for doc in docs:
        source_path = doc.metadata.get('source', '알 수 없음')

        # title과 url 추출 (metadata에 있으면 사용, 없으면 None)
        title = doc.metadata.get('title')
        url = doc.metadata.get('url')

        # title이 없으면 source 파일명을 title로 사용
        if not title:
            # 파일 경로에서 파일명만 추출
            import os
            title = os.path.basename(source_path)

        sources.append({
            "title": title,
            "url": url,
            "source": source_path
        })

    return sources
