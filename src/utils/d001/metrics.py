"""모니터링 및 메트릭 수집 유틸리티.

Prometheus 형식의 메트릭을 수집합니다.
"""

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

from .logger import get_logger

logger = get_logger(__name__)

# Prometheus 레지스트리
registry = CollectorRegistry()

# 메트릭 정의
query_counter = Counter(
    "rag_queries_total", "Total number of RAG queries", ["status"], registry=registry
)

query_duration = Histogram(
    "rag_query_duration_seconds",
    "RAG query processing duration in seconds",
    registry=registry,
)

retrieval_counter = Counter(
    "rag_retrieval_total",
    "Total number of document retrievals",
    ["cache_hit"],
    registry=registry,
)

retrieval_k = Gauge(
    "rag_retrieval_k", "Number of documents retrieved per query", registry=registry
)

llm_call_counter = Counter(
    "rag_llm_calls_total",
    "Total number of LLM API calls",
    ["status"],
    registry=registry,
)

llm_call_duration = Histogram(
    "rag_llm_call_duration_seconds",
    "LLM API call duration in seconds",
    registry=registry,
)

cache_hit_counter = Counter(
    "rag_cache_hits_total",
    "Total number of cache hits",
    ["cache_type"],
    registry=registry,
)

error_counter = Counter(
    "rag_errors_total", "Total number of errors", ["error_type"], registry=registry
)


def record_query_success() -> None:
    """성공한 쿼리 기록."""
    query_counter.labels(status="success").inc()


def record_query_error() -> None:
    """실패한 쿼리 기록."""
    query_counter.labels(status="error").inc()


def record_retrieval(cache_hit: bool = False) -> None:
    """문서 검색 기록.

    Args:
        cache_hit: 캐시 히트 여부
    """
    retrieval_counter.labels(cache_hit=str(cache_hit)).inc()


def record_llm_call(success: bool = True) -> None:
    """LLM 호출 기록.

    Args:
        success: 호출 성공 여부
    """
    status = "success" if success else "error"
    llm_call_counter.labels(status=status).inc()


def record_cache_hit(cache_type: str) -> None:
    """캐시 히트 기록.

    Args:
        cache_type: 캐시 유형 (retrieval, llm)
    """
    cache_hit_counter.labels(cache_type=cache_type).inc()


def record_error(error_type: str) -> None:
    """에러 기록.

    Args:
        error_type: 에러 유형 (validation, retrieval, generation, etc.)
    """
    error_counter.labels(error_type=error_type).inc()
    logger.warning(f"Error recorded: {error_type}")
