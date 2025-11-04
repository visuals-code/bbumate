"""쿼리 재작성 모듈 (Query Rewriter).

검색 결과가 부족할 때 사용자 질문을 더 나은 검색어로 재작성합니다.
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_upstage import ChatUpstage

from src.config import settings
from src.utils.d001.logger import get_logger

logger = get_logger(__name__)


class QueryRewriter:
    """쿼리 재작성기.

    사용자의 질문을 웹 검색에 적합한 형태로 재작성합니다.
    """

    def __init__(self) -> None:
        """QueryRewriter를 초기화합니다."""
        self.llm = ChatUpstage(
            model=settings.UPSTAGE_CHAT_MODEL,
            temperature=0.0,
        )
        self.rewrite_prompt = self._create_rewrite_prompt()

    def _create_rewrite_prompt(self) -> ChatPromptTemplate:
        """쿼리 재작성 프롬프트를 생성합니다.

        Returns:
            쿼리 재작성용 프롬프트 템플릿.
        """
        system_template = """당신은 검색 쿼리 최적화 전문가입니다.

사용자의 질문을 웹 검색에 적합한 형태로 재작성하세요.

재작성 규칙:
1. 핵심 키워드만 추출
2. 불필요한 조사, 어미 제거
3. 검색 엔진이 이해하기 쉬운 형태로 변환
4. 동의어나 관련 용어 추가 가능
5. 한국어로 유지

원본 질문: {question}

재작성된 검색 쿼리만 출력하세요 (설명 없이):"""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_template),
            ("human", "검색 쿼리를 재작성해주세요."),
        ])

        return prompt

    def rewrite(self, question: str) -> str:
        """질문을 검색에 적합한 형태로 재작성합니다.

        Args:
            question: 원본 질문.

        Returns:
            재작성된 검색 쿼리. 실패 시 원본 질문 반환.
        """
        logger.info("Rewriting query: %s", question)

        try:
            chain = self.rewrite_prompt | self.llm | StrOutputParser()
            rewritten_query = chain.invoke({"question": question})

            logger.info("Rewritten query: %s", rewritten_query)
            return rewritten_query.strip()

        except Exception as e:  # pylint: disable=broad-except
            # 폴백: LLM 실패 시에도 원본 질문 반환으로 검색 계속 진행
            logger.error("Query rewriting failed: %s", e)
            # 실패 시 원본 질문 반환
            return question
            
    async def arewrite(self, question: str) -> str:
        """질문을 비동기적으로 검색에 적합한 형태로 재작성합니다.

        Args:
            question: 원본 질문.

        Returns:
            재작성된 검색 쿼리. 실패 시 원본 질문 반환.
        """
        logger.info("Rewriting query: %s", question)

        try:
            chain = self.rewrite_prompt | self.llm | StrOutputParser()
            rewritten_query = await chain.ainvoke({"question": question})

            logger.info("Rewritten query: %s", rewritten_query)
            return rewritten_query.strip()

        except Exception as e:  # pylint: disable=broad-except
            # 폴백: LLM 실패 시에도 원본 질문 반환으로 검색 계속 진행
            logger.error("Query rewriting failed: %s", e)
            # 실패 시 원본 질문 반환
            return question


def create_rewriter() -> QueryRewriter:
    """Query Rewriter 생성 헬퍼 함수.

    Returns:
        쿼리 재작성기 인스턴스.
    """
    return QueryRewriter()
