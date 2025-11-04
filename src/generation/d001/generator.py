"""LLM 기반 답변 생성 모듈.

검색된 문서를 기반으로 LLM을 사용하여 최종 답변을 생성합니다.
"""

from typing import List

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_upstage import ChatUpstage

from src.config import settings
from src.exceptions import ConfigurationError, GenerationError
from src.utils.d001.formatters import format_docs
from src.utils.d001.logger import get_logger

logger = get_logger(__name__)


def get_llm_model() -> ChatUpstage:
    """Upstage LLM 객체를 생성하여 반환합니다.

    환경 변수에서 API 키와 모델 이름을 자동으로 로드합니다.

    Returns:
        초기화된 LLM 모델.

    Raises:
        ConfigurationError: API 키가 설정되지 않은 경우.
    """
    if not settings.UPSTAGE_API_KEY:
        raise ConfigurationError(
            "UPSTAGE_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요."
        )

    logger.info("Upstage LLM 모델 '%s' 초기화", settings.UPSTAGE_CHAT_MODEL)

    # ChatUpstage는 UPSTAGE_API_KEY 환경 변수를 자동으로 사용
    return ChatUpstage(model=settings.UPSTAGE_CHAT_MODEL, temperature=settings.LLM_TEMPERATURE)


def get_rag_prompt_template() -> ChatPromptTemplate:
    """RAG 작업에 사용될 ChatPromptTemplate을 정의하여 반환합니다.

    Returns:
        RAG용 프롬프트 템플릿.
    """
    # 시스템 메시지: LLM에게 역할과 지침을 부여
    system_template = (
        "당신은 유용한 질의응답 비서입니다. 제공된 다음 문맥(Context)만을 사용하여 "
        "사용자의 질문(Question)에 한국어로 자세히 답변하세요. "
        "문맥에 관련 정보가 없거나 부족하면, **'제공된 문맥에 관련 정보가 없습니다.'**라고 명시하고 "
        "추측하거나 꾸며내지 마세요."
        "\n\n--- 문맥 ---"
        "\n{context}"
    )

    # 사용자 메시지: 실제 질문을 전달
    human_template = "{question}"

    # 템플릿 생성
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_template),
            ("human", human_template),
        ]
    )

    return prompt


def generate_answer(question: str, context_documents: List[Document]) -> str:
    """검색된 문서를 기반으로 LLM을 호출하여 최종 답변을 생성합니다.

    Args:
        question: 사용자의 질문.
        context_documents: 검색 모듈에서 가져온 관련 문서 청크 목록.

    Returns:
        LLM이 생성한 최종 답변 텍스트.

    Raises:
        GenerationError: LLM 호출 또는 답변 생성 실패.
    """
    logger.info("LLM을 사용하여 최종 답변 생성 시작")

    try:
        # 1. LLM 객체 초기화
        llm = get_llm_model()

        # 2. 프롬프트 템플릿 정의
        prompt = get_rag_prompt_template()

        # 3. Context를 하나의 문자열로 결합 (공통 포맷터 사용)
        context_text = format_docs(context_documents)

        # 4. LangChain Expression Language (LCEL) 체인 구성
        rag_chain = prompt | llm | StrOutputParser()

        # 5. 체인 실행 및 답변 생성
        answer = rag_chain.invoke({"context": context_text, "question": question})
        logger.info("답변 생성 완료")
        return answer

    except ConfigurationError:
        # 설정 오류는 그대로 전파
        raise
    except Exception as e:  # pylint: disable=broad-except
        # LLM 호출 실패를 GenerationError로 래핑하여 일관된 에러 처리
        logger.error("LLM 호출 중 오류 발생: %s", e)
        raise GenerationError(f"답변 생성 중 오류가 발생했습니다: {e}") from e
