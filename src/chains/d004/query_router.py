from langchain_upstage import ChatUpstage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os


class QueryRouter:

    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = api_key or os.getenv("UPSTAGE_API_KEY")
        self.model = model or os.getenv("UPSTAGE_CHAT_MODEL", "solar-1-mini-chat")
        self.llm = ChatUpstage(api_key=self.api_key, model=self.model)

        self.route_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "당신은 질문 분석 및 라우팅 전문가입니다.\n"
                    "사용자 질문을 분석하여 다음 중 하나로 분류하세요:\n\n"
                    "1. VECTORDB: 일반적인 신혼부부 혜택/정책 정보 (백화점, 카드, 가전, 통신사 등)\n"
                    "   - 기존에 문서화된 정보로 답변 가능한 질문\n"
                    "   - 예: '신혼부부 백화점 혜택', '현대카드 할인', 'LG 가전 혜택'\n\n"
                    "2. WEBSEARCH: 실시간/최신 정보가 필요하거나 DB에 없을 가능성이 높은 질문\n"
                    "   - 특정 날짜/시간 언급 (오늘, 최근, 2025년 등)\n"
                    "   - 매우 구체적이거나 희귀한 지역/주제\n"
                    "   - 최신 정책 변경/발표 관련\n"
                    "   - 예: '오늘 발표된 정책', '독도 거주 혜택', '2025년 11월 금리'\n\n"
                    "3. UNCLEAR: 질문이 너무 모호함\n"
                    "   - 예: '혜택', '뭐 있어?', '알려줘'\n\n"
                    "응답 형식:\n"
                    "- 'VECTORDB': 벡터DB 검색으로 답변 가능\n"
                    "- 'WEBSEARCH': 웹 검색 필요\n"
                    "- 'CLARIFICATION: [재질문]': 질문이 모호하여 명확화 필요\n\n"
                    "예시:\n"
                    "질문: '신혼부부 백화점 혜택' -> VECTORDB\n"
                    "질문: '롯데백화점 웨딩 마일리지' -> VECTORDB\n"
                    "질문: '오늘 발표된 신혼부부 정책' -> WEBSEARCH\n"
                    "질문: '독도 거주 신혼부부 혜택' -> WEBSEARCH\n"
                    "질문: '혜택' -> CLARIFICATION: 어떤 종류의 혜택을 찾고 계신가요?\n"
                    "질문: '신혼부부 우주여행 지원' -> WEBSEARCH\n"
                    "반드시 VECTORDB, WEBSEARCH, CLARIFICATION 중 하나로 시작하세요.",
                ),
                ("human", "질문: {question}"),
            ]
        )

        self.route_chain = self.route_prompt | self.llm | StrOutputParser()

    def route(self, question: str) -> dict:
        """
        질문을 분석하여 라우팅 결정

        Returns:
            {
                "status": "VECTORDB" | "WEBSEARCH" | "UNCLEAR",
                "clarification": str or None,
                "original_question": str
            }
        """
        result = self.route_chain.invoke({"question": question})
        result = result.strip()

        if result.startswith("VECTORDB"):
            return {
                "status": "CLEAR",  # chain.py와 호환성 유지
                "clarification": None,
                "original_question": question,
            }
        elif result.startswith("WEBSEARCH"):
            return {
                "status": "WEB_SEARCH",
                "clarification": None,
                "original_question": question,
            }
        elif result.startswith("CLARIFICATION:"):
            clarification = result.replace("CLARIFICATION:", "").strip()
            return {
                "status": "UNCLEAR",
                "clarification": clarification,
                "original_question": question,
            }
        else:
            # 기본값: 벡터DB 검색 시도
            return {
                "status": "CLEAR",
                "clarification": None,
                "original_question": question,
            }
