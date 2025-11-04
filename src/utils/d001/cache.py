"""캐싱 유틸리티.

검색 결과를 캐싱하여 성능을 향상시킵니다.
"""

import hashlib
from typing import Any, Dict, Optional

from cachetools import TTLCache

from .logger import get_logger

logger = get_logger(__name__)

# 검색 결과 캐시 (최대 100개 항목, 5분 TTL)
retrieval_cache: TTLCache = TTLCache(maxsize=100, ttl=300)

# LLM 응답 캐시 (최대 50개 항목, 15분 TTL)
llm_response_cache: TTLCache = TTLCache(maxsize=50, ttl=900)


def generate_cache_key(query: str, k: Optional[int] = None) -> str:
    """쿼리와 k 값으로 캐시 키 생성.

    Args:
        query: 검색 쿼리
        k: 검색할 문서 수 (optional)

    Returns:
        str: SHA256 해시 기반 캐시 키
    """
    cache_input = f"{query}:{k}" if k is not None else query
    return hashlib.sha256(cache_input.encode()).hexdigest()


def get_cached_retrieval(query: str, k: Optional[int] = None) -> Optional[Any]:
    """캐시된 검색 결과 조회.

    Args:
        query: 검색 쿼리
        k: 검색할 문서 수

    Returns:
        Optional[Any]: 검색 결과 또는 None (캐시 미스 시)
    """
    key = generate_cache_key(query, k)
    result = retrieval_cache.get(key)

    if result is not None:
        logger.debug(f"Cache hit for retrieval: {query[:30]}...")

    return result


def set_cached_retrieval(query: str, k: Optional[int], result: Any) -> None:
    """검색 결과를 캐시에 저장.

    Args:
        query: 검색 쿼리
        k: 검색할 문서 수
        result: 검색 결과
    """
    key = generate_cache_key(query, k)
    retrieval_cache[key] = result
    logger.debug(f"Cached retrieval result: {query[:30]}...")


def get_cached_llm_response(query: str) -> Optional[str]:
    """캐시된 LLM 응답 조회.

    Args:
        query: 사용자 질문

    Returns:
        Optional[str]: LLM 응답 또는 None (캐시 미스 시)
    """
    key = generate_cache_key(query)
    result = llm_response_cache.get(key)

    if result is not None:
        logger.info(f"Cache hit for LLM response: {query[:30]}...")

    return result


def set_cached_llm_response(query: str, response: str) -> None:
    """LLM 응답을 캐시에 저장.

    Args:
        query: 사용자 질문
        response: LLM 생성 응답
    """
    key = generate_cache_key(query)
    llm_response_cache[key] = response
    logger.debug(f"Cached LLM response: {query[:30]}...")


def clear_all_caches() -> None:
    """모든 캐시 초기화."""
    retrieval_cache.clear()
    llm_response_cache.clear()
    logger.info("All caches cleared")


def get_cache_stats() -> Dict[str, int]:
    """캐시 통계 반환.

    Returns:
        Dict[str, int]: 캐시 크기 및 통계 정보
    """
    return {
        "retrieval_cache_size": len(retrieval_cache),
        "retrieval_cache_maxsize": retrieval_cache.maxsize,
        "llm_cache_size": len(llm_response_cache),
        "llm_cache_maxsize": llm_response_cache.maxsize,
    }
