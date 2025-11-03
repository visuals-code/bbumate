from langchain_upstage import ChatUpstage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os


class QueryRewriter:
    """검색 결과가 좋지 않을 때 질문을 재작성하는 클래스"""

    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = api_key or os.getenv("UPSTAGE_API_KEY")
        self.model = model or os.getenv("UPSTAGE_CHAT_MODEL", "solar-1-mini-chat")
        self.llm = ChatUpstage(api_key=self.api_key, model=self.model)

        self.rewrite_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "당신은 검색 질의 최적화 전문가입니다.\n"
                    "사용자의 질문을 더 나은 검색 결과를 얻을 수 있도록 재작성하세요.\n\n"
                    "재작성 가이드:\n"
                    "1. 핵심 키워드를 명확하게 유지\n"
                    "2. 구체적인 정보 요청으로 변환\n"
                    "3. 동의어나 관련 용어 추가\n"
                    "4. 불필요한 단어 제거\n"
                    "5. 신혼부부 지원정책 및 혜택 관련 맥락 추가\n\n"
                    "예시:\n"
                    "원본: '백화점' -> 재작성: '신혼부부 백화점 할인 혜택 정책'\n"
                    "원본: '혜택 알려줘' -> 재작성: '신혼부부가 받을 수 있는 혜택 및 지원 프로그램'\n\n"
                    "재작성된 질문만 출력하세요.",
                ),
                ("human", "원본 질문: {question}\n\n재작성된 질문:"),
            ]
        )

        self.rewrite_chain = self.rewrite_prompt | self.llm | StrOutputParser()

    def rewrite(self, question: str) -> str:
        """
        질문을 재작성하여 더 나은 검색 결과를 얻을 수 있도록 함

        Args:
            question: 원본 질문

        Returns:
            재작성된 질문
        """
        rewritten = self.rewrite_chain.invoke({"question": question})
        return rewritten.strip()

    def rewrite_with_history(self, question: str, failed_attempts: int = 0) -> str:
        """
        이전 시도 횟수를 고려하여 질문 재작성

        Args:
            question: 원본 질문
            failed_attempts: 실패한 시도 횟수

        Returns:
            재작성된 질문
        """
        if failed_attempts == 0:
            return self.rewrite(question)

        # 실패 횟수가 많을수록 더 다양한 접근 시도
        enhanced_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    f"이전 {failed_attempts}번의 검색이 실패했습니다.\n"
                    "질문을 완전히 다른 관점에서 재작성하세요.\n"
                    "더 광범위하거나 다른 키워드를 사용하세요.\n\n"
                    "재작성된 질문만 출력하세요.",
                ),
                ("human", "원본 질문: {question}\n\n재작성된 질문:"),
            ]
        )

        enhanced_chain = enhanced_prompt | self.llm | StrOutputParser()
        rewritten = enhanced_chain.invoke({"question": question})
        return rewritten.strip()
