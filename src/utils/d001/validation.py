"""입력 검증 및 sanitization 유틸리티.

사용자 입력을 검증하고 프롬프트 인젝션 공격을 방지합니다.
"""

import re

from src.utils.d001.exceptions import RAGException
from src.utils.d001.logger import get_logger

logger = get_logger(__name__)


def sanitize_query(question: str) -> str:
    """사용자 질문을 sanitize하여 프롬프트 인젝션 공격 방지.

    Args:
        question: 사용자 입력 질문

    Returns:
        str: Sanitize된 질문

    Raises:
        RAGException: 악의적인 입력 패턴 감지 시
    """
    if not question:
        raise RAGException("질문이 비어있습니다.")

    # 제어 문자 제거
    question = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", question)

    # 반복된 특수문자 제한 (3개 이상 → 2개로)
    question = re.sub(r"([!?.,]){3,}", r"\1\1", question)

    # 프롬프트 인젝션 패턴 탐지
    suspicious_patterns = [
        r"ignore\s+(previous|above|prior|all)\s+(instructions?|prompts?|rules?)",
        r"system\s*:\s*",
        r"<\|.*?\|>",  # ChatML injection
        r"###\s*(instruction|system|assistant)",
        r"you\s+are\s+now",
        r"forget\s+(everything|all|previous)",
        r"new\s+(instructions?|role|task)",
    ]

    for pattern in suspicious_patterns:
        if re.search(pattern, question, re.IGNORECASE):
            logger.warning(f"Suspicious input pattern detected: {pattern}")
            raise RAGException(
                "입력에 부적절한 패턴이 감지되었습니다. 일반적인 질문 형식으로 다시 시도해주세요."
            )

    # 과도하게 긴 입력 경고
    if len(question) > 1000:
        logger.warning(f"Very long input detected: {len(question)} characters")

    return question.strip()


def validate_query_length(
    question: str, min_length: int = 1, max_length: int = 1000
) -> bool:
    """질문 길이 검증.

    Args:
        question: 검증할 질문
        min_length: 최소 길이
        max_length: 최대 길이

    Returns:
        bool: 유효하면 True

    Raises:
        RAGException: 길이가 범위를 벗어나는 경우
    """
    length = len(question.strip())

    if length < min_length:
        raise RAGException(
            f"질문이 너무 짧습니다. 최소 {min_length}자 이상 입력해주세요."
        )

    if length > max_length:
        raise RAGException(
            f"질문이 너무 깁니다. 최대 {max_length}자 이하로 입력해주세요."
        )

    return True
