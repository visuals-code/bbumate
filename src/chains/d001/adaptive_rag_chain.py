"""적응형 RAG 체인 (Adaptive RAG Chain).

문서 관련성을 평가하고, 필요시 쿼리를 재작성하여 웹 검색을 수행하는
고급 RAG 파이프라인입니다.

워크플로우:
1. Clarification: 질문 모호성 검사 및 명확화 (필요 시)
2. Context Enrichment: 사용자 컨텍스트(지역, 주거형태) 통합
3. Retrieve: 벡터 DB에서 문서 검색
4. Grade: 검색된 문서의 관련성 평가
5. Rerank: 관련성 점수 기반 재정렬
6. Decision:
   - 관련성 높음 → DB 문서로 답변 생성
   - 관련성 낮음 → Query Rewrite → Web Search → 답변 생성
"""

import time
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser

from src.chains.d001.clarification_chain import create_clarification_chain
from src.config import settings
from src.exceptions import RAGException
from src.generation.d001.generator import get_llm_model, get_rag_prompt_template
from src.retrieval.d001.grader import create_grader
from src.retrieval.d001.reranker import create_reranker
from src.retrieval.d001.retriever_factory import get_chroma_retriever
from src.retrieval.d001.rewriter import create_rewriter
from src.retrieval.d001.web_search import create_web_search_tool
from src.utils.d001.formatters import format_docs
from src.utils.d001.logger import get_logger

logger = get_logger(__name__)


class AdaptiveRAGChain:
    """적응형 RAG 체인.

    문서 품질을 평가하고 필요시 웹 검색을 수행하는 고급 RAG 시스템입니다.
    """

    def __init__(
        self,
        k: Optional[int] = None,
        top_k: int = 3,
        relevance_threshold: float = 0.6,
        confidence_threshold: float = 0.7,
        ambiguity_threshold: float = 0.6,
        use_clarification: bool = True,
        use_web_search: bool = True,
        use_mock_web: bool = True,
    ) -> None:
        if k is None:
            k = settings.DEFAULT_RETRIEVAL_K

        self.k = k
        self.top_k = top_k
        self.relevance_threshold = relevance_threshold
        self.confidence_threshold = confidence_threshold
        self.ambiguity_threshold = ambiguity_threshold
        self.use_clarification = use_clarification
        self.use_web_search = use_web_search

        # Core RAG components
        self.retriever = get_chroma_retriever(k=k)
        self.grader = create_grader(relevance_threshold=relevance_threshold)
        self.reranker = create_reranker(top_k=top_k)
        self.rewriter = create_rewriter()
        self.web_search = create_web_search_tool(max_results=3, use_mock=use_mock_web)
        self.llm = get_llm_model()
        self.prompt = get_rag_prompt_template()

        # Clarification component
        if self.use_clarification:
            self.clarification_chain = create_clarification_chain(ambiguity_threshold=ambiguity_threshold)
        else:
            self.clarification_chain = None

        logger.info(
            "Adaptive RAG Chain initialized (k=%d, top_k=%d, relevance_threshold=%.2f, "
            "confidence_threshold=%.2f, use_clarification=%s, ambiguity_threshold=%.2f)",
            k, top_k, relevance_threshold, confidence_threshold, use_clarification, ambiguity_threshold
        )

    async def ainvoke(
        self,
        question: str,
        region: Optional[str] = None,
        residence_type: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """비동기적으로 RAG 파이프라인을 실행합니다.

        Args:
            question: 사용자 질문
            region: 사용자 거주 지역 (선택)
            residence_type: 사용자 주거 형태 리스트 (선택)

        Returns:
            Dict containing:
                - answer: 생성된 답변
                - source: 답변 출처 ('database' or 'web_search')
                - needs_clarification: 명확화 필요 여부
                - clarification_questions: 명확화 질문 리스트 (필요 시)
                - session_id: 명확화 세션 ID (필요 시)
        """
        pipeline_start = time.perf_counter()

        logger.info("=" * 80)
        logger.info("Adaptive RAG Pipeline Started")
        logger.info("=" * 80)
        logger.info("User question: %s", question)
        if region:
            logger.info("User region: %s", region)
        if residence_type:
            logger.info("User residence type: %s", residence_type)

        # Step 0: Clarification (if enabled)
        final_question = question
        needs_clarification = False
        clarification_questions = []
        session_id = None

        if self.use_clarification and self.clarification_chain:
            logger.info("\n" + "=" * 80)
            logger.info("Step 0: Clarification Check")
            logger.info("=" * 80)

            step_start = time.perf_counter()
            ambiguity_score = self.clarification_chain.analyze_question(question)
            step_duration = time.perf_counter() - step_start

            logger.info("  Ambiguity detected: %s", ambiguity_score.is_ambiguous)
            logger.info("  Ambiguity level: %.2f", ambiguity_score.ambiguity_level)
            logger.info("  Step 0 Duration: %.3f seconds", step_duration)

            if ambiguity_score.is_ambiguous:
                logger.info("  Question is ambiguous. Clarification needed.")
                logger.info("  Reasons: %s", ambiguity_score.reasons)
                logger.info("  Clarification questions: %s", ambiguity_score.clarification_questions)

                # Create clarification session
                session_id = self.clarification_chain.create_session(
                    question, ambiguity_score.clarification_questions
                )

                total_duration = time.perf_counter() - pipeline_start
                logger.info("\n" + "=" * 80)
                logger.info("Total Pipeline Duration: %.3f seconds", total_duration)
                logger.info("=" * 80)

                return {
                    "answer": None,
                    "source": None,
                    "needs_clarification": True,
                    "clarification_questions": ambiguity_score.clarification_questions,
                    "session_id": session_id,
                }
            else:
                logger.info("  Question is clear. Proceeding to retrieval.")

        # Step 1: Retrieve
        logger.info("\n" + "=" * 80)
        logger.info("Step 1: Retrieving documents from vector DB")
        logger.info("=" * 80)

        step1_start = time.perf_counter()
        retrieved_docs = await self.retriever.ainvoke(final_question)
        step1_duration = time.perf_counter() - step1_start

        logger.info("  Retrieved %d documents", len(retrieved_docs))
        logger.info("  Step 1 Duration: %.3f seconds", step1_duration)

        # Step 2: Grade
        logger.info("\n" + "=" * 80)
        logger.info("Step 2: Grading document relevance")
        logger.info("=" * 80)

        step2_start = time.perf_counter()
        relevant_docs_with_scores, avg_confidence = await self.grader.afilter_relevant_documents_with_scores(
            final_question, retrieved_docs
        )
        step2_duration = time.perf_counter() - step2_start

        logger.info("  Relevant documents: %d/%d", len(relevant_docs_with_scores), len(retrieved_docs))
        logger.info("  Average confidence: %.2f", avg_confidence)
        logger.info("  Step 2 Duration: %.3f seconds", step2_duration)

        # Step 3: Rerank
        logger.info("\n" + "=" * 80)
        logger.info("Step 3: Reranking documents by relevance score")
        logger.info("=" * 80)

        step3_start = time.perf_counter()
        reranked_docs = self.reranker.rerank(relevant_docs_with_scores)
        step3_duration = time.perf_counter() - step3_start

        logger.info("  Reranked to top %d documents", len(reranked_docs))
        logger.info("  Step 3 Duration: %.3f seconds", step3_duration)

        # Step 4: Decision
        use_db_docs = len(reranked_docs) > 0 and avg_confidence >= self.confidence_threshold

        logger.info("\n" + "=" * 80)
        logger.info("Step 4: Decision Making")
        logger.info("=" * 80)
        logger.info("  Reranked docs: %d", len(reranked_docs))
        logger.info("  Avg confidence: %.2f", avg_confidence)
        logger.info("  Confidence threshold: %.2f", self.confidence_threshold)
        logger.info("  Use DB docs: %s", use_db_docs)

        step4_start = time.perf_counter()

        if use_db_docs:
            logger.info("  DECISION: Using database documents")
            context_docs = reranked_docs
            source = "database"
        else:
            if not self.use_web_search:
                logger.warning("  DECISION: Insufficient docs but web search disabled")
                context_docs = retrieved_docs
                source = "database_fallback"
            else:
                logger.info("  DECISION: Insufficient relevant docs. Falling back to web search")

                logger.info("\n" + "=" * 80)
                logger.info("Step 4a: Query Rewriting")
                logger.info("=" * 80)

                step4a_start = time.perf_counter()
                rewritten_query = await self.rewriter.arewrite(final_question)
                step4a_duration = time.perf_counter() - step4a_start

                logger.info("  Original: %s", final_question)
                logger.info("  Rewritten: %s", rewritten_query)
                logger.info("  Step 4a Duration: %.3f seconds", step4a_duration)

                logger.info("\n" + "=" * 80)
                logger.info("Step 4b: Web Search")
                logger.info("=" * 80)

                step4b_start = time.perf_counter()
                web_docs = await self.web_search.asearch(rewritten_query)
                step4b_duration = time.perf_counter() - step4b_start

                if web_docs:
                    context_docs = web_docs
                    source = "web_search"
                    logger.info("  Found %d web search results", len(web_docs))
                else:
                    logger.warning("  Web search returned no results. Using DB docs as fallback")
                    context_docs = retrieved_docs
                    source = "database_fallback"

                logger.info("  Step 4b Duration: %.3f seconds", step4b_duration)

        step4_duration = time.perf_counter() - step4_start
        logger.info("  Step 4 Total Duration: %.3f seconds", step4_duration)

        # Step 5: Generate
        logger.info("\n" + "=" * 80)
        logger.info("Step 5: Generating Answer")
        logger.info("=" * 80)
        logger.info("  Source: %s", source.upper())
        logger.info("  Context docs count: %d", len(context_docs))

        step5_start = time.perf_counter()
        answer = await self._agenerate_answer(final_question, context_docs, source, region, residence_type)
        step5_duration = time.perf_counter() - step5_start

        logger.info("  Answer generated successfully")
        logger.info("  Step 5 Duration: %.3f seconds", step5_duration)

        # Total pipeline time
        total_duration = time.perf_counter() - pipeline_start

        logger.info("\n" + "=" * 80)
        logger.info("Adaptive RAG Pipeline Completed")
        logger.info("=" * 80)
        logger.info("TIMING SUMMARY:")
        if self.use_clarification and self.clarification_chain:
            logger.info("  Step 0 (Clarification): %.3f s", step_duration if 'step_duration' in locals() else 0.0)
        logger.info("  Step 1 (Retrieve):      %.3f s", step1_duration)
        logger.info("  Step 2 (Grade):         %.3f s", step2_duration)
        logger.info("  Step 3 (Rerank):        %.3f s", step3_duration)
        logger.info("  Step 4 (Decision):      %.3f s", step4_duration)
        if 'step4a_duration' in locals():
            logger.info("    - Step 4a (Rewrite): %.3f s", step4a_duration)
        if 'step4b_duration' in locals():
            logger.info("    - Step 4b (WebSearch): %.3f s", step4b_duration)
        logger.info("  Step 5 (Generate):      %.3f s", step5_duration)
        logger.info("  " + "-" * 50)
        logger.info("  TOTAL:                  %.3f s", total_duration)
        logger.info("=" * 80)

        return {
            "answer": answer,
            "source": source,
            "needs_clarification": False,
            "clarification_questions": [],
            "session_id": None,
        }

    async def _agenerate_answer(
        self,
        question: str,
        context_docs: List[Document],
        source: str,
        region: Optional[str] = None,
        residence_type: Optional[List[str]] = None
    ) -> str:
        """답변을 생성합니다 (사용자 컨텍스트 포함).

        Args:
            question: 사용자 질문
            context_docs: 컨텍스트 문서 리스트
            source: 문서 출처
            region: 사용자 거주 지역
            residence_type: 사용자 주거 형태 리스트

        Returns:
            생성된 답변 텍스트
        """
        context_text = format_docs(context_docs)

        # 웹 검색 결과인 경우 표시
        if source == "web_search":
            context_text = (
                "[웹 검색 결과]\n"
                f"{context_text}\n\n"
                "※ 위 정보는 웹에서 검색한 최신 정보입니다."
            )

        # 사용자 컨텍스트 통합
        enriched_question = question
        user_context_parts = []

        if region:
            user_context_parts.append(f"거주 지역: {region}")

        if residence_type:
            residence_str = ", ".join(residence_type)
            user_context_parts.append(f"주거 형태: {residence_str}")

        if user_context_parts:
            user_context = " | ".join(user_context_parts)
            enriched_question = (
                f"{question}\n\n"
                f"[사용자 사전 선택 정보: {user_context}]\n\n"
                f"**답변 우선순위 규칙:**\n"
                f"1. 사용자가 질문에서 명시적으로 다른 지역이나 주거형태를 언급한 경우, "
                f"질문 속 정보를 최우선으로 답변하세요.\n"
                f"2. 질문에 지역이나 주거형태 언급이 없는 경우, 사전 선택 정보를 기반으로 답변하세요.\n"
                f"3. 답변 시 사용한 지역/주거형태를 명확히 언급하세요."
            )
            logger.info("  User context: %s", user_context)

        chain = self.prompt | self.llm | StrOutputParser()
        answer = await chain.ainvoke({"context": context_text, "question": enriched_question})

        return answer

    def refine_question_with_clarification(self, session_id: str, clarification_answer: str) -> str:
        """명확화 응답으로 질문을 재구성합니다.

        Args:
            session_id: 명확화 세션 ID
            clarification_answer: 사용자의 명확화 응답

        Returns:
            재구성된 질문

        Raises:
            ValueError: 세션을 찾을 수 없거나 clarification_chain이 없는 경우
        """
        if not self.clarification_chain:
            raise ValueError("Clarification chain is not enabled")

        refined_question = self.clarification_chain.refine_question(session_id, clarification_answer)
        logger.info("Question refined: %s", refined_question)

        # 세션 정리
        self.clarification_chain.delete_session(session_id)

        return refined_question

    async def aget_workflow_info(self, question: str) -> Dict[str, Any]:
        retrieved_docs = await self.retriever.ainvoke(question)
        relevant_docs_with_scores, avg_confidence = await self.grader.afilter_relevant_documents_with_scores(
            question, retrieved_docs
        )
        reranked_docs = self.reranker.rerank(relevant_docs_with_scores)
        use_db = len(reranked_docs) > 0 and avg_confidence >= self.confidence_threshold

        return {
            "question": question,
            "retrieved_docs_count": len(retrieved_docs),
            "relevant_docs_count": len(relevant_docs_with_scores),
            "reranked_docs_count": len(reranked_docs),
            "avg_confidence": avg_confidence,
            "confidence_threshold": self.confidence_threshold,
            "decision": "use_database" if use_db else "use_web_search",
        }

def setup_adaptive_rag_chain(
    k: Optional[int] = None,
    top_k: int = 3,
    relevance_threshold: float = 0.6,
    confidence_threshold: float = 0.7,
    ambiguity_threshold: float = 0.6,
    use_clarification: bool = True,
    use_web_search: bool = True,
    use_mock_web: bool = True,
) -> AdaptiveRAGChain:
    """Adaptive RAG Chain을 설정합니다.

    Args:
        k: 검색할 문서 개수
        top_k: 재정렬 후 선택할 문서 개수
        relevance_threshold: 문서 관련성 임계값
        confidence_threshold: 신뢰도 임계값
        ambiguity_threshold: 모호성 임계값
        use_clarification: 명확화 기능 사용 여부
        use_web_search: 웹 검색 사용 여부
        use_mock_web: Mock 웹 검색 사용 여부

    Returns:
        AdaptiveRAGChain 인스턴스

    Raises:
        RAGException: 설정 실패 시
    """
    try:
        chain = AdaptiveRAGChain(
            k=k,
            top_k=top_k,
            relevance_threshold=relevance_threshold,
            confidence_threshold=confidence_threshold,
            ambiguity_threshold=ambiguity_threshold,
            use_clarification=use_clarification,
            use_web_search=use_web_search,
            use_mock_web=use_mock_web,
        )
        logger.info("Adaptive RAG Chain setup complete")
        return chain
    except Exception as e:
        logger.error("Adaptive RAG Chain setup failed: %s", e)
        raise RAGException(f"Adaptive RAG Chain 설정 실패: {e}") from e

