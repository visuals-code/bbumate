import os
from typing import Optional, List
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

# 환경변수 불러오기
load_dotenv()
UPSTAGE_API_KEY = os.getenv("UPSTAGE_API_KEY")

if not UPSTAGE_API_KEY:
    raise ValueError("Please set UPSTAGE_API_KEY in your .env file")

# FastAPI 앱 인스턴스 생성
app = FastAPI(title="RAG API - 신혼부부 지원정책 상담")

# CORS
origins = [
    "http://localhost:8080",
    "http://127.0.0.1:8080",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 루트 경로
@app.get("/")
def root():
    return {"message": "신혼부부 지원정책 RAG 서버가 실행 중입니다!"}


# LangGraph 통합 버전 사용 (langgraph/chain.py)
from src.langgraph.chain import answer_question
from src.ingestion.index import ingest
from src.api.d002.api_d002 import (
    markdown_to_text,
    format_to_markdown,
    markdown_to_html,
    format_sources,
)


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500, description="사용자 질문")
    region: Optional[str] = Field(
        None, description="사전 선택 지역 (예: 인천, 서울, 경기)"
    )
    housing_type: Optional[str] = Field(
        None, description="사전 선택 주거형태 (예: 전세, 월세, 자가, 매매)"
    )

    @field_validator("question")
    @classmethod
    def validate_question(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("질문은 비어있을 수 없습니다")
        return v

    @field_validator("region", "housing_type", mode="before")
    @classmethod
    def filter_empty_strings(cls, v: Optional[str]) -> Optional[str]:
        """빈 문자열을 None으로 변환 (필터링)."""
        if v is None:
            return None
        if isinstance(v, str) and v.strip() == "":
            return None
        return v.strip() if isinstance(v, str) else v


class SourceItem(BaseModel):
    title: str
    url: str | None = None
    source: str


class QueryResponse(BaseModel):
    answer: str
    answer_md: str
    answer_html: str
    sources: list[SourceItem]


class IngestRequest(BaseModel):
    domain: str = "all"  # d001, d002, d003, d004, d005, or "all" for all domains
    data_dir: str | None = None
    mode: str = "pdf"
    chunk_size: int = 1000
    chunk_overlap: int = 200


class IngestResponse(BaseModel):
    message: str
    domain: str
    status: str


# API 엔드포인트
@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    """신혼부부 정책 관련 질문에 답변합니다.

    - **question**: 질문 내용 (1~500자)
    - **region**: 사전 선택 지역 (예: 인천, 서울, 경기)
    - **housing_type**: 사전 선택 주거형태 (예: 전세, 월세, 자가, 매매)
    """
    import time
    import logging

    logger = logging.getLogger(__name__)
    start_time = time.perf_counter()

    # LangGraph 통합 기반 RAG 파이프라인 실행
    res = answer_question(
        question=request.question,
        k=3,
        use_grade=True,  # Grade 활성화 (관련 문서만 필터링)
        use_validation=True,  # validation 활성화
        region=request.region,
        housing_type=request.housing_type,
        verbose=True,
    )

    answer = res.get("answer", "답변 생성 실패")
    sources_list = res.get("sources", [])
    clarification_needed = res.get("clarification_needed", False)
    duration_ms = res.get("duration_ms", 0)
    num_docs = res.get("num_docs", 0)
    web_search_used = res.get("web_search_used", False)

    # 로그로 메타데이터 출력
    elapsed_time = time.perf_counter() - start_time
    logger.info(
        f"Query completed - "
        f"Duration: {duration_ms}ms (API: {elapsed_time*1000:.2f}ms), "
        f"Docs: {num_docs}, "
        f"WebSearch: {web_search_used}, "
        f"ClarificationNeeded: {clarification_needed}"
    )

    # LLM이 마크다운 형식으로 생성하므로 변환
    answer_text = markdown_to_text(answer)  # 순수 텍스트
    answer_md = format_to_markdown(answer)  # 마크다운 (정리)
    answer_html = markdown_to_html(answer)  # HTML 변환

    # sources 객체 배열로 변환
    # format_sources는 내부에서 모든 도메인(d001-d005)의 링크를 자동으로 처리
    # domain 파라미터는 기본값으로만 사용되며, 실제로는 경로에서 도메인을 자동 감지
    sources = format_sources(sources_list)

    return {
        "answer": answer_text,
        "answer_md": answer_md,
        "answer_html": answer_html,
        "sources": sources,
    }


# Ingestion 엔드포인트 (관리자용)
@app.post("/ingest", response_model=IngestResponse)
def run_ingest(request: IngestRequest):
    """도메인별 문서 ingestion 실행 (관리자용).

    주의: ingestion은 시간이 오래 걸릴 수 있으며, 벡터 DB를 재생성합니다.
    """
    try:
        ingest(
            domain=request.domain,
            data_dir=request.data_dir,
            mode=request.mode if request.domain == "d003" else "pdf",
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
        )
        return {
            "message": f"Ingestion completed successfully for domain {request.domain}",
            "domain": request.domain,
            "status": "success",
        }
    except Exception as e:
        return {
            "message": f"Ingestion failed: {str(e)}",
            "domain": request.domain,
            "status": "error",
        }
