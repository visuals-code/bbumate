# -*- coding: utf-8 -*-
import os
import sys
import re
import time
from pathlib import Path
from dotenv import load_dotenv
from typing import Dict, List
from langchain_upstage import ChatUpstage, UpstageEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

sys.path.insert(0, "/Users/a/KDT_BE13_Toy_Project4/src/chains/d004")
sys.path.insert(0, "/Users/a/KDT_BE13_Toy_Project4/src/generation/d004")
sys.path.insert(0, "/Users/a/KDT_BE13_Toy_Project4/src/ingestion/d004")
sys.path.insert(0, "/Users/a/KDT_BE13_Toy_Project4/src/retrieval/d004")
sys.path.insert(0, "/Users/a/KDT_BE13_Toy_Project4/src/test/d004")

from query_router import QueryRouter
from retrieval import load_retriever, format_docs
from grader import DocumentGrader
from query_rewriter import QueryRewriter
from web_search_fallback import WebSearchFallback


class AdvancedRAGChain:
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
        self.collection_name = os.getenv("COLLECTION_NAME", "pdf_promotion_chunks")
        self.max_rewrite_attempts = max_rewrite_attempts

        # 각 컴포넌트 초기화
        self.router = QueryRouter(api_key=self.api_key, model=self.model)
        self.grader = DocumentGrader(api_key=self.api_key, model=self.model)
        self.rewriter = QueryRewriter(api_key=self.api_key, model=self.model)
        self.web_search = WebSearchFallback(api_key=self.api_key, model=self.model)

        # Retriever 초기화
        self.retriever = load_retriever(
            db_path=self.db_path, collection_name=self.collection_name, k=3
        )

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

    def _extract_sources(self, documents: List[Document]) -> List[Dict]:
        sources = []
        seen_sources = set()

        for doc in documents:
            metadata = doc.metadata
            source_file = metadata.get("source_file", metadata.get("source", ""))

            # 중복 체크
            source_key = source_file
            if source_key in seen_sources:
                continue
            seen_sources.add(source_key)

            # title 추출
            title = metadata.get("heading", "")
            if not title:
                title = Path(source_file).stem.replace("_", " ").title()

            # url 추출 및 디버깅
            url = metadata.get("url", None)

            # 🔍 디버깅 출력 (나중에 제거 가능)
            print(f"[DEBUG] 출처: {Path(source_file).name}")
            print(f"  - URL: {url}")
            print(f"  - 전체 메타데이터: {metadata}")

            sources.append({"title": title, "url": url, "source": source_file})

        return sources

    def _extract_web_sources(self, web_result: Dict) -> List[Dict]:
        sources = []
        for result in web_result.get("results", []):
            sources.append(
                {"title": result["title"], "url": result["url"], "source": "web_search"}
            )
        return sources

    def _format_markdown(self, text: str) -> str:
        # 금액 패턴: 숫자
        # 비율 패턴: 숫자 + %
        pattern = r"(\d+(?:,\d+)*(?:\.\d+)?(?:만원|억원|천원|원|%))"
        text = re.sub(pattern, r"**\1**", text)

        # 문장 단위 줄바꿈
        text = re.sub(r"\.\s+", ".\\n\\n", text)

        return text.strip()

    def _format_html(self, text: str) -> str:

        # 금액/비율 패턴
        pattern = r"(\d+(?:,\d+)*(?:\.\d+)?(?:만원|억원|천원|원|%))"
        text = re.sub(pattern, r"<strong>\1</strong>", text)

        # 문장 단위 줄바꿈
        text = re.sub(r"\.\s+", ".<br/><br/>", text)

        # div로 감싸기
        return f"<div>{text}</div>"

    def invoke(
        self, question: str, region: str = None, housing_type: str = None
    ) -> Dict:

        print(f"\n{'='*60}")
        print(f"[RAG 파이프라인 시작] 질문: {question}")
        if region:
            print(f"  - 거주지역: {region}")
        if housing_type:
            print(f"  - 주거형태: {housing_type}")
        print(f"{'='*60}")

        # Step 1: Query Router - 질문 명확성 판단
        print("\n Query Router: 질문 분석 중...")
        routing_result = self.router.route(question)

        if routing_result["status"] == "WEB_SEARCH":
            print(f"  → 실시간 정보 필요. 웹 검색으로 우회합니다.")
            web_result = self.web_search.search_and_answer(question)
            answer_text = web_result["answer"]
            web_sources = self._extract_web_sources(web_result)

            return {
                "answer": answer_text,
                "answer_md": self._format_markdown(answer_text),
                "answer_html": self._format_html(answer_text),
                "sources": web_sources,
                "original_question": question,
                "final_question": question,
                "routing_status": "WEB_SEARCH",
                "documents_retrieved": 0,
                "relevant_documents": 0,
                "source": "web_search",
                "rewrite_count": 0,
            }

        print(f"  → 질문이 명확합니다. 검색을 진행합니다.")

        # Step 2: Retrieval with potential rewriting
        current_question = question
        rewrite_count = 0
        relevant_docs = []
        documents = []

        # metadata 필터 구성 (Chroma 형식)
        metadata_filter = None
        if region and housing_type:
            # 두 조건 모두 있으면 $and 사용
            metadata_filter = {
                "$and": [
                    {"region": {"$eq": region}},
                    {"housing_type": {"$eq": housing_type}},
                ]
            }
        elif region:
            # region만 있으면
            metadata_filter = {"region": {"$eq": region}}
        elif housing_type:
            # housing_type만 있으면
            metadata_filter = {"housing_type": {"$eq": housing_type}}

        for attempt in range(self.max_rewrite_attempts + 1):
            if metadata_filter:
                print(f"  → 필터 적용: {metadata_filter}")
                # vectorstore에 직접 접근
                documents = self.retriever.vectorstore.similarity_search(
                    current_question, k=3, filter=metadata_filter
                )
            else:
                documents = self.retriever.invoke(current_question)

            # 문서 검색 (필터 적용)
            start_time = time.time()
            if metadata_filter:
                # Chroma의 where 필터 사용
                documents = self.retriever.vectorstore.similarity_search(
                    current_question, k=3, filter=metadata_filter
                )
            else:
                documents = self.retriever.invoke(current_question)

            search_time = time.time() - start_time
            print(f"  → {len(documents)}개 문서 검색됨 (소요시간: {search_time:.2f}초)")

            if not documents:
                print("  → 검색 결과 없음")
                if attempt < self.max_rewrite_attempts:
                    print(f"\n Query Rewriter: 질문 재작성 중...")
                    current_question = self.rewriter.rewrite_with_history(
                        question, attempt
                    )
                    rewrite_count += 1
                    print(f"  → 재작성된 질문: {current_question}")
                    continue
                else:
                    break

            # Step 3: Grader - 문서 관련성 평가
            print(f"\n Grader: 문서 관련성 평가 중...")
            grading_result = self.grader.grade_documents(current_question, documents)

            print(
                f"  → 관련 문서: {len(grading_result['relevant_documents'])}/{grading_result['total_count']}"
            )

            relevant_docs = grading_result["relevant_documents"]

            if not relevant_docs and attempt == 0:
                print(f"  → 첫 검색에서 관련 문서 없음. 웹 검색으로 전환합니다.")
                print(f"\n Web Search: 웹 검색 수행 중...")
                web_result = self.web_search.search_and_answer(question)
                answer_text = web_result["answer"]
                web_sources = self._extract_web_sources(web_result)

                return {
                    "answer": answer_text,
                    "answer_md": self._format_markdown(answer_text),
                    "answer_html": self._format_html(answer_text),
                    "sources": web_sources,
                    "original_question": question,
                    "final_question": current_question,
                    "routing_status": "CLEAR",
                    "documents_retrieved": len(documents),
                    "relevant_documents": 0,
                    "source": "web_search",
                    "rewrite_count": 0,
                }

            # 관련 문서가 있으면 답변 생성
            if relevant_docs:
                break

            # 관련 문서가 없고 재시도 가능하면 질문 재작성
            if attempt < self.max_rewrite_attempts:
                print(f"\n Query Rewriter: 질문 재작성 중...")
                current_question = self.rewriter.rewrite_with_history(question, attempt)
                rewrite_count += 1
                print(f"  → 재작성된 질문: {current_question}")
            else:
                print(f"  → 최대 재시도 횟수 도달")

        # Step 5: Web Search Fallback (관련 문서가 없을 경우)
        if not relevant_docs:
            print(f"\n Web Search: 웹 검색 폴백 수행 중...")
            web_result = self.web_search.search_and_answer(question)

            answer_text = web_result["answer"]

            web_sources = self._extract_web_sources(web_result)

            return {
                "answer": answer_text,
                "answer_md": self._format_markdown(answer_text),
                "answer_html": self._format_html(answer_text),
                "sources": web_sources,
                "original_question": question,
                "final_question": current_question,
                "routing_status": "CLEAR",
                "documents_retrieved": len(documents) if documents else 0,
                "relevant_documents": 0,
                "source": web_result["source"],
                "rewrite_count": rewrite_count,
            }

        # Step 6: LLM Answer Generation
        print(f"\n LLM Answer Generation: 답변 생성 중...")
        context = format_docs(relevant_docs)

        start_time = time.time()
        answer = self.answer_chain.invoke(
            {"context": context, "question": current_question}
        )
        generation_time = time.time() - start_time
        print(f"  → 답변 생성 완료 (소요시간: {generation_time:.2f}초)")

        # 출처 정보 추출
        sources = self._extract_sources(relevant_docs)

        print(f"\n{'='*60}")
        print(f"[RAG 파이프라인 완료]")
        print(f"{'='*60}")

        return {
            "answer": answer,
            "answer_md": self._format_markdown(answer),
            "answer_html": self._format_html(answer),
            "sources": sources,
            "original_question": question,
            "final_question": current_question,
            "routing_status": "CLEAR",
            "documents_retrieved": len(documents) if documents else 0,
            "relevant_documents": len(relevant_docs),
            "source": "vectorstore",
            "rewrite_count": rewrite_count,
        }
