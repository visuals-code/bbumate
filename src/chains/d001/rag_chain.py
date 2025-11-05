"""RAG 체인 구성 및 관리 모듈.

Retriever와 LLM을 연결하여 완전한 RAG 파이프라인을 구축합니다.
"""

from typing import Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnablePassthrough

from src.utils.d001.config import settings
from src.utils.d001.exceptions import RAGException
from src.generation.d001.generator import get_llm_model, get_rag_prompt_template
from src.retrieval.d001.retriever_factory import get_chroma_retriever
from src.utils.d001.formatters import format_docs
from src.utils.d001.logger import get_logger

logger = get_logger(__name__)


def setup_rag_chain(k: Optional[int] = None) -> Runnable:
    """RAG 파이프라인 전체를 구성하는 LangChain Runnable 체인을 설정합니다.

    Args:
        k: 검색할 문서(청크)의 개수. None인 경우 설정의 기본값 사용.

    Returns:
        Runnable: 사용자 질문을 입력받아 최종 답변을 반환하는 RAG 체인.

    Raises:
        RAGException: RAG 체인 구성 중 오류 발생.
    """
    if k is None:
        k = settings.DEFAULT_RETRIEVAL_K

    logger.info("RAG 체인 설정 시작 (검색 문서 개수 k=%d)", k)

    try:
        # 1. Retriever 설정 (검색 모듈)
        retriever = get_chroma_retriever(k=k)

        # 2. LLM 및 Prompt 설정 (생성 모듈)
        llm = get_llm_model()
        prompt = get_rag_prompt_template()

        # 3. LCEL(LangChain Expression Language) 체인 구성
        rag_chain = (
            {
                "context": retriever | format_docs,
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
            | StrOutputParser()
        )

        logger.info("RAG 체인 구성 완료")
        return rag_chain

    except Exception as e:  # pylint: disable=broad-except
        # 모든 예외를 RAGException으로 래핑하여 일관된 에러 처리
        logger.error("RAG 체인 구성 중 오류 발생: %s", e)
        raise RAGException(f"RAG 체인 설정 실패: {e}") from e
