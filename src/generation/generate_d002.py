import os
import logging
from typing import List

from dotenv import load_dotenv
from langchain_upstage import ChatUpstage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)
load_dotenv()


# --- 1. Upstage LLM 연결 ---
def load_llm(temperature: float = 0.3) -> ChatUpstage:
    """Upstage Chat 모델 초기화"""
    api_key = os.getenv("UPSTAGE_API_KEY")
    if not api_key:
        raise ValueError("UPSTAGE_API_KEY 환경변수가 필요합니다")

    # 올바른 모델명 사용
    model_name = os.getenv("UPSTAGE_CHAT_MODEL", "solar-mini")

    try:
        llm = ChatUpstage(api_key=api_key, model=model_name, temperature=temperature)
        logger.info(f"✅ LLM 초기화: {model_name} (temp={temperature})")
        return llm
    except Exception as e:
        raise ValueError(f"LLM 초기화 실패: {e}")


# --- 2. 프롬프트 템플릿 (인덴트 제거) ---
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """당신은 신혼부부 지원정책 도메인 전문가입니다.
주어진 컨텍스트만을 근거로 정확하게 답변하세요.

규칙:
- 컨텍스트에 없는 내용은 답변하지 마세요
- 모를 경우 "제공된 문서에서 해당 정보를 찾을 수 없습니다"라고 답하세요
- 답변은 간결하고 명확하게 작성하세요
- 반드시 답변 끝에 참고한 출처를 나열하세요 (예: [출처: 파일명.pdf])

컨텍스트:
{context}""",
        ),
        ("human", "{question}"),
    ]
)

output_parser = StrOutputParser()


# --- 3. 문서 포맷팅 ---
def _format_docs(docs: List[Document]) -> str:
    """검색된 문서를 LLM이 읽기 좋은 형식으로 변환"""
    if not docs:
        return "제공된 문서 없음"

    formatted = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        content = (doc.page_content or "").strip()

        # 너무 긴 내용은 잘라냄
        if len(content) > 2000:
            content = content[:2000] + "..."

        formatted.append(f"[문서 {i}] 출처: {source}\n{content}")

    return "\n\n---\n\n".join(formatted)


# --- 4. 응답 생성 ---
def generate_response(
    user_query: str, retrieved_docs: List[Document], temperature: float = 0.3
) -> str:
    """검색된 문서 기반 답변 생성"""

    # 문서 없으면 즉시 반환
    if not retrieved_docs:
        logger.warning("검색된 문서가 없습니다")
        return "죄송합니다. 관련 정보를 찾을 수 없습니다. 다른 질문을 시도해보세요."

    try:
        llm = load_llm(temperature=temperature)
        chain = prompt | llm | output_parser

        context = _format_docs(retrieved_docs)
        logger.info(f"컨텍스트 길이: {len(context)}자")

        response = chain.invoke({"context": context, "question": user_query})

        return response.strip()

    except Exception as e:
        logger.error(f"응답 생성 실패: {e}")
        return f"죄송합니다. 답변 생성 중 오류가 발생했습니다: {str(e)}"


# --- 5. 통합 RAG 함수 (옵션) ---
def rag_pipeline(query: str, vectordb, k: int = 5, temperature: float = 0.3) -> dict:
    """검색 + 생성을 한번에 처리"""
    # 검색
    logger.info(f"🔍 검색: '{query}'")
    docs = vectordb.similarity_search(query, k=k)

    if not docs:
        return {
            "query": query,
            "answer": "관련 문서를 찾을 수 없습니다.",
            "sources": [],
        }

    # 생성
    logger.info(f"💬 답변 생성 중...")
    answer = generate_response(query, docs, temperature=temperature)

    # 출처 정리
    sources = [doc.metadata.get("source", "unknown") for doc in docs]

    return {
        "query": query,
        "answer": answer,
        "sources": list(set(sources)),  # 중복 제거
        "num_docs": len(docs),
    }


if __name__ == "__main__":
    # 테스트용 가짜 문서
    from langchain.schema import Document

    fake_docs = [
        Document(
            page_content="버팀목 전세자금대출 금리는 연 1.8~2.4%입니다. "
            "신혼부부의 경우 0.2%p 우대금리가 적용됩니다.",
            metadata={"source": "주택도시기금_2024_공고.pdf"},
        ),
        Document(
            page_content="디딤돌대출 신혼부부 특례는 연 2.15~3.0% 금리로 "
            "최대 3.6억원까지 지원합니다.",
            metadata={"source": "국토교통부_주택금융안내.pdf"},
        ),
    ]

    query = "신혼부부 전세자금대출 금리가 어떻게 되나요?"

    print("=" * 70)
    print(f"질문: {query}")
    print("=" * 70)

    answer = generate_response(query, fake_docs)
    print(f"\n답변:\n{answer}\n")

    print("=" * 70)
