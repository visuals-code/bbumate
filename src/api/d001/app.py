"""FastAPI 기반 RAG 서버.

신혼부부 지원정책 상담을 위한 RAG API 엔드포인트를 제공합니다.
"""

import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, AsyncIterator

from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import generate_latest
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from src.chains.d001.rag_chain import setup_rag_chain
from src.config import settings
from src.retrieval.d001.retriever_factory import get_chroma_retriever
from src.retrieval.d001.grader import create_grader
from src.retrieval.d001.reranker import create_reranker
from src.retrieval.d001.rewriter import create_rewriter
from src.retrieval.d001.web_search import create_web_search_tool
from src.generation.d001.generator import get_llm_model, get_rag_prompt_template
from src.exceptions import (
    ConfigurationError,
    DatabaseError,
    GenerationError,
    RAGException,
    RetrievalError,
)
from src.utils.d001.formatters import (
    extract_sources_from_docs,
    format_answer_to_html,
    format_answer_to_markdown,
    format_docs,
)
from src.utils.d001.metrics import (
    query_duration,
    record_error,
    record_llm_call,
    record_query_error,
    record_query_success,
    registry,
)
from src.utils.d001.validation import sanitize_query, validate_query_length
from src.utils.d001.logger import get_logger, setup_logging
from langchain_core.output_parsers import StrOutputParser

# 로깅 초기화
setup_logging()
logger = get_logger(__name__)

# Rate limiter 초기화
limiter = Limiter(key_func=get_remote_address)

# 전역 RAG 체인 및 Retriever 변수
rag_chain = None
retriever = None
adaptive_rag_chain = None
# Legacy Adaptive RAG 컴포넌트 (deprecated)
grader = None
reranker = None
rewriter = None
web_search = None
llm = None
prompt = None


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """애플리케이션 생명주기를 관리합니다 (시작/종료 시 실행).

    Args:
        _app: FastAPI 애플리케이션 인스턴스.

    Yields:
        None: 애플리케이션 실행 중.
    """
    global rag_chain, retriever, adaptive_rag_chain, grader, reranker, rewriter, web_search, llm, prompt

    # 시작 시
    logger.info("=" * 80)
    logger.info("RAG Server Initialization Started")
    logger.info("=" * 80)

    # Basic RAG Chain 초기화
    try:
        rag_chain = setup_rag_chain()
        retriever = get_chroma_retriever(k=settings.DEFAULT_RETRIEVAL_K)
        logger.info("Basic RAG chain and Retriever initialized successfully")
    except (ConfigurationError, DatabaseError) as e:
        logger.error("RAG chain initialization failed: %s", e)
        logger.warning(
            "RAG chain not initialized. "
            "Chroma DB may not exist. Please run ingestion/ingest.py first."
        )
        rag_chain = None
        retriever = None
    except Exception as e:  # pylint: disable=broad-except
        logger.error("Unexpected error during RAG chain initialization: %s", e)
        rag_chain = None
        retriever = None

    # Adaptive RAG Chain 초기화 (권장)
    if settings.USE_ADAPTIVE_RAG and retriever is not None:
        try:
            from src.chains.d001.adaptive_rag_chain import setup_adaptive_rag_chain

            adaptive_rag_chain = setup_adaptive_rag_chain(
                k=settings.DEFAULT_RETRIEVAL_K,
                top_k=3,
                relevance_threshold=settings.RELEVANCE_THRESHOLD,
                confidence_threshold=settings.CONFIDENCE_THRESHOLD,
                ambiguity_threshold=0.6,
                use_clarification=True,
                use_web_search=settings.USE_WEB_SEARCH,
                use_mock_web=settings.USE_MOCK_WEB_SEARCH,
            )
            logger.info("Adaptive RAG Chain initialized successfully")
        except Exception as e:  # pylint: disable=broad-except
            logger.error("Adaptive RAG Chain initialization failed: %s", e)
            adaptive_rag_chain = None

            # Fallback to legacy components
            try:
                grader = create_grader(relevance_threshold=settings.RELEVANCE_THRESHOLD)
                reranker = create_reranker(top_k=3)
                rewriter = create_rewriter()
                web_search = create_web_search_tool(max_results=3, use_mock=settings.USE_MOCK_WEB_SEARCH)
                llm = get_llm_model()
                prompt = get_rag_prompt_template()
                logger.info("Legacy Adaptive RAG components initialized successfully (fallback)")
            except Exception as fallback_e:  # pylint: disable=broad-except
                logger.error("Legacy Adaptive RAG initialization also failed: %s", fallback_e)
                grader = None
                reranker = None
                rewriter = None
                web_search = None
                llm = None
                prompt = None

    logger.info("=" * 80)
    logger.info("RAG Server Initialization Complete")
    logger.info("  - Basic RAG: %s", "Available" if rag_chain else "Unavailable")
    logger.info("  - Adaptive RAG: %s", "Available" if adaptive_rag_chain else "Unavailable")
    logger.info("=" * 80)

    yield

    # 종료 시
    logger.info("RAG Server Shutdown")


# FastAPI 앱 인스턴스 생성
app = FastAPI(
    title="쀼메이트 - 신혼부부 지원정책 RAG API",
    description="신혼부부를 위한 주택·대출·복지 정책 상담 AI 챗봇",
    version="1.0.0",
    lifespan=lifespan,
)

# Rate limiter 설정
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS 미들웨어 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.get_cors_origins_list(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 요청/응답 로깅 미들웨어
@app.middleware("http")
async def log_requests(request: Request, call_next: Any) -> Response:
    """모든 HTTP 요청과 응답을 로깅합니다.

    Args:
        request: HTTP 요청 객체.
        call_next: 다음 미들웨어 또는 엔드포인트 호출 함수.

    Returns:
        Response: HTTP 응답 객체.
    """
    start_time = time.time()

    # 요청 로깅
    logger.info("Request: %s %s", request.method, request.url.path)

    # 응답 처리
    response = await call_next(request)

    # 응답 시간 계산 및 로깅
    duration = time.time() - start_time
    logger.info(
        "Response: %d - Duration: %.3fs - Path: %s",
        response.status_code, duration, request.url.path
    )

    return response


# 요청 모델
class QueryRequest(BaseModel):
    """질의응답 요청 모델."""

    question: str = Field(
        ...,
        description="사용자 질문",
        min_length=1,
        max_length=1000,
        examples=[
            "신혼부부 특별 공급의 주요 혜택은?",
            "전세자금대출 조건이 어떻게 되나요?",
            "행복주택 신청 자격 알려주세요",
        ],
    )
    region: str | None = Field(
        None,
        description="사용자 거주 지역 (예: '서울시 강남구', '부산시')",
        max_length=100,
        examples=["서울시 강남구", "부산시 해운대구", "대전시"],
    )
    residence_type: list[str] | None = Field(
        None,
        description="사용자 주거 형태 (예: ['전세', '월세'])",
        max_items=5,
        examples=[["전세"], ["월세"], ["전세", "월세"]],
    )
    clarification_session_id: str | None = Field(
        None,
        description="명확화 세션 ID (re-ask 후 재요청 시 사용)",
    )
    clarification_answer: str | None = Field(
        None,
        description="명확화 질문에 대한 사용자 응답",
        max_length=500,
    )


# Source 모델
class Source(BaseModel):
    """출처 정보 모델."""

    title: str = Field(..., description="문서 제목")
    url: str | None = Field(None, description="문서 URL (없으면 null)")
    source: str = Field(..., description="문서 파일 경로")


# 응답 모델
class QueryResponse(BaseModel):
    """질의응답 응답 모델."""

    answer: str | None = Field(None, description="AI가 생성한 답변 (순수 텍스트, 선택)")
    answer_md: str | None = Field(None, description="마크다운 형식의 답변 (금액/비율 굵게 처리, 문장 단위 줄바꿈)")
    answer_html: str | None = Field(None, description="HTML 형식의 답변 (금액/비율 <strong>, 줄바꿈 <br/>, 선택)")
    sources: list[Source] | None = Field(None, description="답변 생성에 사용된 출처 목록")
    needs_clarification: bool = Field(False, description="명확화 필요 여부")
    clarification_questions: list[str] | None = Field(None, description="명확화 질문 리스트")
    clarification_session_id: str | None = Field(None, description="명확화 세션 ID")


class ErrorResponse(BaseModel):
    """에러 응답 모델."""

    error_code: str = Field(..., description="에러 코드")
    message: str = Field(..., description="에러 메시지")
    detail: str = Field(None, description="상세 정보")
    timestamp: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat(),
        description="발생 시각 (UTC)"
    )


class HealthResponse(BaseModel):
    """헬스 체크 응답 모델."""

    message: str
    status: str
    rag_chain_ready: bool


# API 엔드포인트
@app.get("/", response_model=HealthResponse, tags=["Health"])
def root() -> HealthResponse:
    """헬스 체크 엔드포인트.

    서버 상태와 RAG 체인 준비 여부를 확인합니다.

    Returns:
        HealthResponse: 서버 상태 정보.
    """
    is_ready = rag_chain is not None

    if is_ready:
        logger.debug("헬스 체크: 정상")
    else:
        logger.warning("헬스 체크: RAG 체인 미초기화")

    return HealthResponse(
        message="쀼메이트 - 신혼부부 지원정책 RAG 서버가 실행 중입니다!",
        status="healthy",
        rag_chain_ready=is_ready,
    )


@app.post(
    "/query",
    response_model=QueryResponse,
    tags=["RAG"],
    status_code=status.HTTP_200_OK,
)
@limiter.limit(f"{settings.RATE_LIMIT_PER_MINUTE}/minute")
async def query_rag(query_request: QueryRequest, request: Request) -> QueryResponse:
    """RAG 기반 질의응답 엔드포인트 (Enhanced with Clarification & User Context).

    워크플로우:
    1. 질문 명확화 체크 (모호한 질문인 경우 re-ask)
    2. 사용자 컨텍스트(지역, 주거형태) 통합
    3. Retrieve → Grade → Rerank
    4. 답변 생성 (필요 시 웹 검색)

    Args:
        query_request: 질문, 지역, 주거형태, 명확화 정보 포함
        request: FastAPI Request 객체 (rate limiting용)

    Returns:
        QueryResponse: 답변 또는 명확화 질문

    Raises:
        HTTPException: RAG 체인 미초기화 또는 처리 중 오류 발생
    """
    # 체인 초기화 확인
    use_new_adaptive = adaptive_rag_chain is not None
    use_legacy_adaptive = grader is not None and reranker is not None

    if not use_new_adaptive and not use_legacy_adaptive and rag_chain is None:
        logger.error("RAG 체인이 초기화되지 않은 상태에서 쿼리 요청 수신")
        record_error("rag_chain_not_initialized")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG 체인이 초기화되지 않았습니다. Chroma DB를 먼저 생성하세요.",
        )

    if retriever is None:
        logger.error("Retriever가 초기화되지 않은 상태에서 쿼리 요청 수신")
        record_error("retriever_not_initialized")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Retriever가 초기화되지 않았습니다. Chroma DB를 먼저 생성하세요.",
        )

    logger.info("=" * 80)
    logger.info("Query Received: %s...", query_request.question[:50])
    if query_request.region:
        logger.info("Region: %s", query_request.region)
    if query_request.residence_type:
        logger.info("Residence Type: %s", query_request.residence_type)

    if use_new_adaptive:
        logger.info("Using New Adaptive RAG Chain (with clarification)")
    elif use_legacy_adaptive:
        logger.info("Using Legacy Adaptive RAG (fallback)")
    else:
        logger.info("Using Basic RAG (no adaptive features)")

    with query_duration.time():
        try:
            # 입력 검증 및 sanitization
            try:
                sanitized_question = sanitize_query(query_request.question)
                validate_query_length(sanitized_question)
            except RAGException as e:
                logger.warning("Input validation failed: %s", e)
                record_error("validation_error")
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=str(e),
                )

            # New Adaptive RAG Chain (권장)
            if use_new_adaptive:
                # Clarification 세션이 있는 경우 질문 재구성
                final_question = sanitized_question
                if query_request.clarification_session_id and query_request.clarification_answer:
                    logger.info("Refining question with clarification answer")
                    try:
                        final_question = adaptive_rag_chain.refine_question_with_clarification(
                            query_request.clarification_session_id,
                            query_request.clarification_answer
                        )
                        logger.info("Question refined: %s", final_question)
                    except ValueError as ve:
                        logger.error("Invalid clarification session: %s", ve)
                        raise HTTPException(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            detail=str(ve),
                        )

                # Adaptive RAG 파이프라인 실행
                result = await adaptive_rag_chain.ainvoke(
                    question=final_question,
                    region=query_request.region,
                    residence_type=query_request.residence_type
                )

                # 명확화가 필요한 경우
                if result.get("needs_clarification"):
                    logger.info("Clarification needed. Returning clarification questions.")
                    record_query_success()
                    return QueryResponse(
                        answer=None,
                        answer_md=None,
                        answer_html=None,
                        sources=None,
                        needs_clarification=True,
                        clarification_questions=result.get("clarification_questions"),
                        clarification_session_id=result.get("session_id"),
                    )

                # 답변 생성 완료
                answer = result.get("answer")
                source_type = result.get("source")

                # 실제 컨텍스트 문서 가져오기 (sources 추출용)
                # Note: adaptive_rag_chain의 결과에서 직접 docs를 가져올 수 없으므로 retriever 재호출
                retrieved_docs = await retriever.ainvoke(final_question)
                sources_data = extract_sources_from_docs(retrieved_docs)

                # 답변 포맷팅
                answer_md = format_answer_to_markdown(answer)
                answer_html = format_answer_to_html(answer)

                # 출처 정보 변환
                sources = [Source(**source) for source in sources_data]

                # 메트릭 기록
                record_query_success()
                record_llm_call(success=True)

                logger.info("Query completed successfully (source: %s)", source_type)
                logger.info("=" * 80)

                return QueryResponse(
                    answer=answer,
                    answer_md=answer_md,
                    answer_html=answer_html,
                    sources=sources,
                    needs_clarification=False,
                    clarification_questions=None,
                    clarification_session_id=None,
                )

            # Legacy Adaptive RAG (fallback)
            elif use_legacy_adaptive:
                # Adaptive RAG 로직 (app.py 내부에 직접 구현)
                logger.info("=" * 60)
                logger.info("Step 1: Retrieving documents from vector DB")

                step1_start = time.time()
                retrieved_docs = await retriever.ainvoke(sanitized_question)
                step1_duration = time.time() - step1_start

                logger.info("Retrieved %d documents from DB", len(retrieved_docs))
                logger.info("  Step 1 Duration: %.3f seconds", step1_duration)
                for i, doc in enumerate(retrieved_docs):
                    logger.debug("  Doc %d: %s (length: %d chars)", i+1, doc.metadata.get('source', 'unknown'), len(doc.page_content))

                logger.info("Step 2: Grading document relevance")

                step2_start = time.time()
                relevant_docs_with_scores, avg_confidence = await grader.afilter_relevant_documents_with_scores(
                    sanitized_question, retrieved_docs
                )
                step2_duration = time.time() - step2_start

                logger.info("Grading complete: %d/%d documents are relevant", len(relevant_docs_with_scores), len(retrieved_docs))
                logger.info("Average confidence score: %.3f (threshold: %.3f)", avg_confidence, settings.CONFIDENCE_THRESHOLD)
                logger.info("  Step 2 Duration: %.3f seconds", step2_duration)
                for i, (doc, score) in enumerate(relevant_docs_with_scores):
                    logger.debug("  Relevant doc %d: score=%.3f, source=%s", i+1, score, doc.metadata.get('source', 'unknown'))

                logger.info("Step 3: Reranking documents by relevance score")

                step3_start = time.time()
                reranked_docs = reranker.rerank(relevant_docs_with_scores)
                step3_duration = time.time() - step3_start

                logger.info("Reranked to top %d documents", len(reranked_docs))
                logger.info("  Step 3 Duration: %.3f seconds", step3_duration)
                for i, doc in enumerate(reranked_docs):
                    logger.debug("  Reranked doc %d: %s", i+1, doc.metadata.get('source', 'unknown'))

                use_db_docs = len(reranked_docs) > 0 and avg_confidence >= settings.CONFIDENCE_THRESHOLD

                logger.info("Step 4: Decision making")
                logger.info("  - Reranked docs count: %d", len(reranked_docs))
                logger.info("  - Average confidence: %.3f", avg_confidence)
                logger.info("  - Confidence threshold: %.3f", settings.CONFIDENCE_THRESHOLD)
                logger.info("  - Use DB docs: %s", use_db_docs)

                step4_start = time.time()

                if use_db_docs:
                    logger.info("DECISION: Using DB documents (%d reranked docs, confidence=%.3f >= %.3f)",
                        len(reranked_docs), avg_confidence, settings.CONFIDENCE_THRESHOLD
                    )
                    context_docs = reranked_docs
                    source_type = "database"
                else:
                    if not settings.USE_WEB_SEARCH or web_search is None:
                        logger.warning("DECISION: Insufficient relevant documents but web search is disabled")
                        logger.warning("  Using retrieved documents as fallback (%d docs)", len(retrieved_docs))
                        context_docs = retrieved_docs
                        source_type = "database_fallback"
                    else:
                        logger.info("DECISION: Insufficient relevant documents (%d docs, confidence=%.3f < %.3f)",
                            len(reranked_docs), avg_confidence, settings.CONFIDENCE_THRESHOLD
                        )
                        logger.info("  Falling back to web search...")

                        logger.info("Step 4a: Rewriting query for web search")
                        step4a_start = time.time()
                        rewritten_query = await rewriter.arewrite(sanitized_question)
                        step4a_duration = time.time() - step4a_start

                        logger.info("  Original query: %s", sanitized_question)
                        logger.info("  Rewritten query: %s", rewritten_query)
                        logger.info("  Step 4a Duration: %.3f seconds", step4a_duration)

                        logger.info("Step 4b: Searching web with rewritten query")
                        step4b_start = time.time()
                        web_docs = await web_search.asearch(rewritten_query)
                        step4b_duration = time.time() - step4b_start

                        if web_docs:
                            context_docs = web_docs
                            source_type = "web_search"
                            logger.info("Found %d web search results", len(web_docs))
                            for i, doc in enumerate(web_docs):
                                logger.debug("  Web doc %d: %s", i+1, doc.metadata.get('source', 'unknown'))
                        else:
                            logger.warning("Web search returned no results")
                            logger.warning("  Using original retrieved documents as fallback (%d docs)", len(retrieved_docs))
                            context_docs = retrieved_docs
                            source_type = "database_fallback"

                        logger.info("  Step 4b Duration: %.3f seconds", step4b_duration)

                step4_duration = time.time() - step4_start
                logger.info("  Step 4 Total Duration: %.3f seconds", step4_duration)

                logger.info("Step 5: Generating answer from %s", source_type.upper())
                logger.info("  Context documents count: %d", len(context_docs))

                step5_start = time.time()

                # 답변 생성
                context_text = format_docs(context_docs)
                logger.debug("  Context text length: %d chars", len(context_text))

                if source_type == "web_search":
                    context_text = (
                        "[웹 검색 결과]\n"
                        f"{context_text}\n\n"
                        "※ 위 정보는 웹에서 검색한 최신 정보입니다."
                    )

                chain = prompt | llm | StrOutputParser()
                answer = await chain.ainvoke({"context": context_text, "question": sanitized_question})

                step5_duration = time.time() - step5_start

                logger.info("Answer generated successfully (source: %s)", source_type.upper())
                logger.info("  Step 5 Duration: %.3f seconds", step5_duration)
                logger.debug("  Answer length: %d chars", len(answer))

                # sources는 실제 사용된 문서에서 추출
                sources_data = extract_sources_from_docs(context_docs)
                logger.info("  Extracted %d sources from context docs", len(sources_data))
                logger.info("=" * 60)
            else:
                # 기본 RAG 사용
                logger.info("Step 1: Retrieving documents from vector DB")

                step1_start = time.time()
                retrieved_docs = await retriever.ainvoke(sanitized_question)
                step1_duration = time.time() - step1_start

                logger.info("Retrieved %d documents", len(retrieved_docs))
                logger.info("  Step 1 Duration: %.3f seconds", step1_duration)

                logger.info("Step 2: Generating answer with basic RAG")

                step2_start = time.time()
                answer = await rag_chain.ainvoke(sanitized_question)
                step2_duration = time.time() - step2_start

                logger.info("Answer generated successfully (basic RAG)")
                logger.info("  Step 2 Duration: %.3f seconds", step2_duration)

                sources_data = extract_sources_from_docs(retrieved_docs)

            # 답변을 다양한 형식으로 변환
            answer_md = format_answer_to_markdown(answer)
            answer_html = format_answer_to_html(answer)

            # 출처 정보 변환
            sources = [Source(**source) for source in sources_data]

            # 메트릭 기록
            record_query_success()
            record_llm_call(success=True)

            logger.info("✅ Query completed successfully (legacy/basic)")
            logger.info("=" * 80)

            return QueryResponse(
                answer=answer,
                answer_md=answer_md,
                answer_html=answer_html,
                sources=sources,
                needs_clarification=False,
                clarification_questions=None,
                clarification_session_id=None,
            )

        except (RetrievalError, DatabaseError) as e:
            logger.error("문서 검색 중 오류: %s", e)
            record_query_error()
            record_error("retrieval_error")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"문서 검색 중 오류가 발생했습니다: {str(e)}",
            )

        except GenerationError as e:
            logger.error("답변 생성 중 오류: %s", e)
            record_query_error()
            record_llm_call(success=False)
            record_error("generation_error")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"답변 생성 중 오류가 발생했습니다: {str(e)}",
            )

        except RAGException as e:
            logger.error("RAG 처리 중 오류: %s", e)
            record_query_error()
            record_error("rag_error")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"요청 처리 중 오류가 발생했습니다: {str(e)}",
            )

        except Exception as e:  # pylint: disable=broad-except
            # 모든 예외를 처리하여 서버가 계속 실행되도록 함
            logger.error("예상치 못한 오류 발생: %s", e, exc_info=True)
            record_query_error()
            record_error("unexpected_error")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="서버 내부 오류가 발생했습니다. 로그를 확인하세요.",
            )


@app.get("/metrics", tags=["Monitoring"])
def metrics() -> Response:
    """Prometheus 메트릭 엔드포인트.

    시스템 메트릭을 Prometheus 형식으로 반환합니다.

    Returns:
        Response: Prometheus 형식의 메트릭 데이터.
    """
    return Response(generate_latest(registry), media_type="text/plain")
