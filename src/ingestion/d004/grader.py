from langchain_upstage import ChatUpstage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import List
from langchain_core.documents import Document
import os


class DocumentGrader:
    """검색된 문서의 관련성을 평가하는 클래스"""

    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = api_key or os.getenv("UPSTAGE_API_KEY")
        self.model = model or os.getenv("UPSTAGE_CHAT_MODEL", "solar-1-mini-chat")
        self.llm = ChatUpstage(api_key=self.api_key, model=self.model)

        self.grading_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "당신은 문서 관련성 평가 전문가입니다.\n"
                    "사용자의 질문과 검색된 문서가 관련이 있는지 판단하세요.\n\n"
                    "평가 기준:\n"
                    "- 문서가 질문에 답하는 데 필요한 정보를 포함하고 있는가?\n"
                    "- 문서의 내용이 질문의 주제와 일치하는가?\n\n"
                    "응답 형식:\n"
                    "- 'yes': 문서가 질문과 관련이 있음\n"
                    "- 'no': 문서가 질문과 관련이 없음\n\n"
                    "반드시 'yes' 또는 'no'로만 답변하세요.",
                ),
                (
                    "human",
                    "질문: {question}\n\n문서 내용:\n{document}\n\n이 문서는 질문과 관련이 있습니까?",
                ),
            ]
        )

        self.grading_chain = self.grading_prompt | self.llm | StrOutputParser()

    def grade_documents(self, question: str, documents: List[Document]) -> dict:
        """
        검색된 문서들을 평가하여 관련 있는 문서만 필터링

        Returns:
            {
                "relevant_documents": List[Document],
                "irrelevant_count": int,
                "total_count": int,
                "needs_web_search": bool
            }
        """
        relevant_docs = []
        irrelevant_count = 0

        for doc in documents:
            # 각 문서의 관련성 평가
            result = self.grading_chain.invoke(
                {"question": question, "document": doc.page_content}
            )

            # 결과 파싱 (yes/no)
            is_relevant = result.strip().lower().startswith("yes")

            if is_relevant:
                relevant_docs.append(doc)
            else:
                irrelevant_count += 1

        # 관련 문서가 없으면 웹 검색 필요
        needs_web_search = len(relevant_docs) == 0

        return {
            "relevant_documents": relevant_docs,
            "irrelevant_count": irrelevant_count,
            "total_count": len(documents),
            "needs_web_search": needs_web_search,
        }

    def grade_single_document(self, question: str, document: Document) -> bool:
        """단일 문서의 관련성 평가"""
        result = self.grading_chain.invoke(
            {"question": question, "document": document.page_content}
        )
        return result.strip().lower().startswith("yes")
