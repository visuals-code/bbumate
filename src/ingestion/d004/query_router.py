from langchain_upstage import ChatUpstage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os


class QueryRouter:
    """질문이 명확한지 판단하고, 필요시 재질문을 생성"""

    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = api_key or os.getenv("UPSTAGE_API_KEY")
        self.model = model or os.getenv("UPSTAGE_CHAT_MODEL", "solar-1-mini-chat")
        self.llm = ChatUpstage(api_key=self.api_key, model=self.model)

        self.route_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "당신은 질문 분석 전문가입니다. 사용자의 질문이 명확한지 판단하세요.\n"
                    "명확한 질문: 구체적인 정보를 요구하거나 특정 주제에 대해 묻는 질문\n"
                    "모호한 질문: 너무 광범위하거나, 맥락이 부족하거나, 무엇을 원하는지 불명확한 질문\n\n"
                    "응답 형식:\n"
                    "- 'CLEAR': 질문이 명확함\n"
                    "- 'UNCLEAR': 질문이 모호함\n"
                    "- 'CLARIFICATION: [재질문 내용]': 질문이 모호하며 추가 정보가 필요함\n\n"
                    "예시:\n"
                    "질문: '신혼부부 백화점 혜택 알려줘' -> CLEAR\n"
                    "질문: '혜택' -> CLARIFICATION: 어떤 종류의 혜택에 대해 알고 싶으신가요? (신혼부부, 청년, 학생 등)\n"
                    "질문: '뭐 있어?' -> CLARIFICATION: 어떤 정보를 찾고 계신가요? 구체적으로 말씀해주세요.",
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
                "status": "CLEAR" | "UNCLEAR",
                "clarification": str or None,
                "original_question": str
            }
        """
        result = self.route_chain.invoke({"question": question})

        if result.startswith("CLEAR"):
            return {
                "status": "CLEAR",
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
        else:  # UNCLEAR
            return {
                "status": "UNCLEAR",
                "clarification": "질문을 좀 더 구체적으로 말씀해 주시겠어요?",
                "original_question": question,
            }
