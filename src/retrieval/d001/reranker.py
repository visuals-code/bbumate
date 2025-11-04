"""문서 재정렬 모듈 (Document Reranker).

Grade 후 관련성 점수에 따라 문서를 재정렬합니다.
"""

from typing import List, Tuple

from langchain_core.documents import Document

from src.utils.d001.logger import get_logger

logger = get_logger(__name__)


class DocumentReranker:
    """문서 재정렬기.

    Grade에서 평가된 관련성 점수를 기반으로 문서를 재정렬합니다.
    """

    def __init__(self, top_k: int = 3) -> None:
        """DocumentReranker를 초기화합니다.

        Args:
            top_k: 최종적으로 선택할 상위 문서 개수.
        """
        self.top_k = top_k

    def rerank(self, graded_docs: List[Tuple[Document, float]]) -> List[Document]:
        """문서를 재정렬합니다.

        Args:
            graded_docs: (문서, 관련성 점수) 튜플 리스트.
                        점수는 0.0 ~ 1.0 사이 값.

        Returns:
            재정렬된 상위 k개 문서 리스트.
        """
        if not graded_docs:
            logger.warning("No documents to rerank")
            return []

        # 관련성 점수 기준으로 내림차순 정렬
        sorted_docs = sorted(
            graded_docs, key=lambda x: x[1], reverse=True  # confidence score
        )

        # 상위 k개만 선택
        top_docs = sorted_docs[: self.top_k]

        logger.info(
            "Reranked %d documents, selected top %d documents",
            len(graded_docs),
            len(top_docs),
        )

        # 로깅: 선택된 문서들의 점수
        for i, (doc, score) in enumerate(top_docs, 1):
            logger.debug("Rank %d: confidence=%.3f", i, score)

        # Document 객체만 반환
        return [doc for doc, score in top_docs]

    def rerank_with_scores(
        self, graded_docs: List[Tuple[Document, float]]
    ) -> List[Tuple[Document, float]]:
        """문서를 재정렬합니다 (점수 포함).

        Args:
            graded_docs: (문서, 관련성 점수) 튜플 리스트.

        Returns:
            재정렬된 (문서, 점수) 튜플 리스트.
        """
        if not graded_docs:
            logger.warning("No documents to rerank")
            return []

        # 관련성 점수 기준으로 내림차순 정렬
        sorted_docs = sorted(graded_docs, key=lambda x: x[1], reverse=True)

        # 상위 k개만 선택
        top_docs = sorted_docs[: self.top_k]

        logger.info(
            "Reranked %d documents with scores, selected top %d",
            len(graded_docs),
            len(top_docs),
        )

        return top_docs


def create_reranker(top_k: int = 3) -> DocumentReranker:
    """Document Reranker 생성 헬퍼 함수.

    Args:
        top_k: 최종 선택할 문서 개수.

    Returns:
        문서 재정렬기 인스턴스.
    """
    return DocumentReranker(top_k=top_k)
