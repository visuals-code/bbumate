import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from typing import Dict, List
from langchain_upstage import ChatUpstage, UpstageEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

sys.path.insert(0, "/Users/a/KDT_BE13_Toy_Project4/src/ingestion/d004")

from query_router import QueryRouter
from retrieval import load_retriever, format_docs
from grader import DocumentGrader
from query_rewriter import QueryRewriter
from web_search_fallback import WebSearchFallback


class AdvancedRAGChain:
    """
    완전한 RAG 파이프라인:
    1. Query Router - 질문 명확성 판단
    2. Retrieval - 문서 검색
    3. Grader - 문서 관련성 평가
    4. Query Rewriter - 질문 재작성
    5. Web Search - 웹 검색 폴백
    6. LLM Answer Generation - 답변 생성
    """

    def __init__(
        self,
        api_key: str = None,
        model: str = None,
        db_path: str = None,
        max_rewrite_attempts: int = 2,
    ):
        load_dotenv()

        self.api_key = api_key or os.getenv("UPSTAGE_API_KEY")
        self.model = model or os.getenv("UPSTAGE_CHAT_MODEL", "solar-1-mini-chat")
        self.db_path = db_path or os.getenv("CHROMA_DB_DIR", "./chroma_storage")
        self.max_rewrite_attempts = max_rewrite_attempts

        # 각 컴포넌트 초기화
        self.router = QueryRouter(api_key=self.api_key, model=self.model)
        self.grader = DocumentGrader(api_key=self.api_key, model=self.model)
        self.rewriter = QueryRewriter(api_key=self.api_key, model=self.model)
        self.web_search = WebSearchFallback(api_key=self.api_key, model=self.model)

        # Retriever 초기화
        self.retriever = load_retriever(db_path=self.db_path, k=3)

        # LLM 초기화
        self.llm = ChatUpstage(api_key=self.api_key, model=self.model)

        # 답변 생성 프롬프트
        self.answer_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "주어진 컨텍스트를 사용하여 사용자 질문에 간결하고 정확하게 답변하세요.\n"
                    "신혼부부 정책이나 혜택과 관련되지 않으면 답변하지 마세요.\n"
                    "모르면 모른다고 답하세요.\n\n"
                    "컨텍스트:\n{context}",
                ),
                ("human", "질문: {question}"),
            ]
        )

        self.answer_chain = self.answer_prompt | self.llm | StrOutputParser()

    def invoke(self, question: str) -> Dict:
        """
        전체 RAG 파이프라인 실행

        Returns:
            {
                "answer": str,
                "original_question": str,
                "final_question": str,
                "routing_status": str,
                "documents_retrieved": int,
                "relevant_documents": int,
                "source": str,
                "rewrite_count": int
            }
        """
        print(f"\n{'='*60}")
        print(f"[RAG 파이프라인 시작] 질문: {question}")
        print(f"{'='*60}")

        # Step 1: Query Router - 질문 명확성 판단
        print("\n[1/6] Query Router: 질문 분석 중...")
        routing_result = self.router.route(question)

        if routing_result["status"] == "UNCLEAR":
            print(f"  → 질문이 모호합니다. 재질문 필요.")
            return {
                "answer": routing_result["clarification"],
                "original_question": question,
                "final_question": question,
                "routing_status": "UNCLEAR",
                "documents_retrieved": 0,
                "relevant_documents": 0,
                "source": "clarification",
                "rewrite_count": 0,
            }

        print(f"  → 질문이 명확합니다. 검색을 진행합니다.")

        # Step 2: Retrieval with potential rewriting
        current_question = question
        rewrite_count = 0
        relevant_docs = []
        documents = []

        for attempt in range(self.max_rewrite_attempts + 1):
            print(f"\n[2/6] Retrieval: 문서 검색 중... (시도 {attempt + 1})")
            print(f"  → 검색 질문: {current_question}")

            # 문서 검색
            documents = self.retriever.invoke(current_question)
            print(f"  → {len(documents)}개 문서 검색됨")

            if not documents:
                print("  → 검색 결과 없음")
                if attempt < self.max_rewrite_attempts:
                    print(f"\n[4/6] Query Rewriter: 질문 재작성 중...")
                    current_question = self.rewriter.rewrite_with_history(
                        question, attempt
                    )
                    rewrite_count += 1
                    print(f"  → 재작성된 질문: {current_question}")
                    continue
                else:
                    break

            # Step 3: Grader - 문서 관련성 평가
            print(f"\n[3/6] Grader: 문서 관련성 평가 중...")
            grading_result = self.grader.grade_documents(current_question, documents)

            print(
                f"  → 관련 문서: {len(grading_result['relevant_documents'])}/{grading_result['total_count']}"
            )

            relevant_docs = grading_result["relevant_documents"]

            # 관련 문서가 있으면 답변 생성
            if relevant_docs:
                print(f"  → 관련 문서 발견! 답변 생성으로 진행")
                break

            # 관련 문서가 없고 재시도 가능하면 질문 재작성
            if attempt < self.max_rewrite_attempts:
                print(f"\n[4/6] Query Rewriter: 질문 재작성 중...")
                current_question = self.rewriter.rewrite_with_history(question, attempt)
                rewrite_count += 1
                print(f"  → 재작성된 질문: {current_question}")
            else:
                print(f"  → 최대 재시도 횟수 도달")

        # Step 5: Web Search Fallback (관련 문서가 없을 경우)
        if not relevant_docs:
            print(f"\n[5/6] Web Search: 웹 검색 폴백 수행 중...")
            web_result = self.web_search.search_and_answer(question)

            return {
                "answer": web_result["answer"],
                "original_question": question,
                "final_question": current_question,
                "routing_status": "CLEAR",
                "documents_retrieved": len(documents) if documents else 0,
                "relevant_documents": 0,
                "source": web_result["source"],
                "rewrite_count": rewrite_count,
            }

        # Step 6: LLM Answer Generation
        print(f"\n[6/6] LLM Answer Generation: 답변 생성 중...")
        context = format_docs(relevant_docs)
        answer = self.answer_chain.invoke(
            {"context": context, "question": current_question}
        )

        print(f"\n{'='*60}")
        print(f"[RAG 파이프라인 완료]")
        print(f"{'='*60}")

        return {
            "answer": answer,
            "original_question": question,
            "final_question": current_question,
            "routing_status": "CLEAR",
            "documents_retrieved": len(documents) if documents else 0,
            "relevant_documents": len(relevant_docs),
            "source": "vectorstore",
            "rewrite_count": rewrite_count,
        }


# 사용 예시
if __name__ == "__main__":
    # RAG 체인 초기화
    rag = AdvancedRAGChain()

    # 테스트 질문들
    test_questions = [
        "신혼부부 백화점 혜택",
        "혜택",  # 모호한 질문
        "신혼부부 주택 구입 지원",
    ]

    for question in test_questions:
        result = rag.invoke(question)
        print(f"\n질문: {result['original_question']}")
        print(f"답변: {result['answer']}")
        print(f"출처: {result['source']}")
        print(f"재작성 횟수: {result['rewrite_count']}")
        print("-" * 60)
