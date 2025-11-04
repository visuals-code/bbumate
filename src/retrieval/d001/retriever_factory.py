"""Retriever 생성 및 관리 모듈.

Chroma DB를 로드하고 문서 검색을 위한 Retriever를 제공합니다.
"""

from typing import Any, List

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict, Field

from src.config import settings
from src.exceptions import DatabaseError
from src.utils.d001.cache import get_cached_retrieval, set_cached_retrieval
from src.utils.d001.embeddings import get_embeddings
from src.utils.d001.metrics import record_retrieval
from src.utils.d001.logger import get_logger

logger = get_logger(__name__)


class CachedRetriever(BaseRetriever):
    """캐싱 기능이 있는 Retriever 래퍼.

    검색 결과를 캐싱하여 동일한 쿼리에 대한 반복 검색을 방지합니다.
    """

    base_retriever: Any = Field(description="기본 검색기 객체")
    k: int = Field(description="검색할 문서 개수")

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _get_relevant_documents(self, query: str) -> List[Document]:
        """쿼리에 대한 문서 검색 (캐싱 지원).

        Args:
            query: 검색 쿼리.

        Returns:
            검색된 문서 리스트.
        """
        # 캐시 확인
        cached_result = get_cached_retrieval(query, self.k)
        if cached_result is not None:
            record_retrieval(cache_hit=True)
            logger.debug("Cache hit for query: %s...", query[:30])
            return cached_result

        # 캐시 미스: 실제 검색 수행
        logger.debug("Cache miss, performing retrieval for: %s...", query[:30])
        retrieved_docs = self.base_retriever.invoke(query)

        # 결과 캐싱
        set_cached_retrieval(query, self.k, retrieved_docs)
        record_retrieval(cache_hit=False)

        return retrieved_docs

    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        """비동기 문서 검색 (캐싱 지원).

        Args:
            query: 검색 쿼리.

        Returns:
            검색된 문서 리스트.
        """
        # 캐시 확인
        cached_result = get_cached_retrieval(query, self.k)
        if cached_result is not None:
            record_retrieval(cache_hit=True)
            logger.debug("Cache hit for query: %s...", query[:30])
            return cached_result

        # 캐시 미스: 실제 검색 수행
        logger.debug("Cache miss, performing async retrieval for: %s...", query[:30])
        retrieved_docs = await self.base_retriever.ainvoke(query)

        # 결과 캐싱
        set_cached_retrieval(query, self.k, retrieved_docs)
        record_retrieval(cache_hit=False)

        return retrieved_docs


def get_chroma_retriever(k: int = None) -> BaseRetriever:
    """저장된 Chroma DB를 로드하고, k개의 문서를 검색하는 Retriever를 반환합니다.

    Args:
        k: 검색할 문서(청크)의 개수. None인 경우 설정의 기본값 사용.

    Returns:
        초기화된 Retriever 객체.

    Raises:
        DatabaseError: Chroma DB 로드 또는 Retriever 생성 실패.
    """
    if k is None:
        k = settings.DEFAULT_RETRIEVAL_K

    logger.info("Chroma DB 로드 및 Retriever 생성 (k=%d)", k)

    chroma_persist_directory = settings.get_chroma_persist_directory()

    if not chroma_persist_directory.exists():
        raise DatabaseError(
            f"Chroma DB 디렉토리를 찾을 수 없습니다: '{chroma_persist_directory}'. "
            "ingestion 파이프라인을 먼저 실행하세요."
        )

    try:
        # 임베딩 모델 초기화
        embeddings = get_embeddings()

        # 저장된 Chroma DB 로드
        persisted_db = Chroma(
            persist_directory=str(chroma_persist_directory),
            embedding_function=embeddings,
            collection_name=settings.CHROMA_COLLECTION_NAME,
        )

        # Retriever 생성 및 설정
        base_retriever = persisted_db.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k},
        )

        # 캐싱 기능이 있는 Retriever로 래핑
        cached_retriever = CachedRetriever(base_retriever=base_retriever, k=k)

        logger.info("Chroma DB 로드 및 Retriever 생성 완료 (k=%d)", k)
        return cached_retriever

    except Exception as e:  # pylint: disable=broad-except
        # 모든 예외를 DatabaseError로 래핑하여 통일된 에러 처리
        logger.error("Chroma DB 로드 또는 Retriever 생성 중 오류 발생: %s", e)
        raise DatabaseError(f"Retriever 생성 실패: {e}") from e
