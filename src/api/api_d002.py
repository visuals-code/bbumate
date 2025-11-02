import logging
from fastapi import FastAPI, APIRouter, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from src.chains.rag_chain_d002 import run_rag


# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


app = FastAPI(
    title="신혼부부 정책 RAG 테스트 서버",
    description="신혼부부 지원정책 관련 질문에 답변하는 RAG 시스템",
)

# CORS 설정 (필요시)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프로덕션에서는 특정 도메인만 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

router = APIRouter()


# 요청 Body 모델 정의
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500, description="사용자 질문")

    @field_validator("query")
    @classmethod
    def validate_query(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("질문은 비어있을 수 없습니다")
        return v


class QueryResponse(BaseModel):
    answer: str = Field(..., description="생성된 답변")
    sources: list[str] = Field(default=[], description="참고 문서 출처")
    latency: str = Field(..., description="응답 시간")
    num_docs: int = Field(default=0, description="검색된 문서 수")


# --- 헬스체크 ---
@app.get("/")
def health_check():
    """서버 상태 확인"""
    return {"status": "ok", "service": "신혼부부 정책 RAG API", "version": "0.1.0"}


@router.post("/ask", response_model=QueryResponse)
def ask_question(request: QueryRequest):
    """
    신혼부부 정책 관련 질문에 답변합니다.

    - **query**: 질문 내용 (1~500자)
    """
    try:
        logger.info(f"질문 수신: {request.query[:50]}...")

        res = run_rag(request.query)

        response = QueryResponse(
            answer=res.get("answer", "답변 생성 실패"),
            sources=res.get("sources", []),
            latency=f"{res.get('duration_ms', 0) / 1000:.2f}s",
            num_docs=res.get("num_docs", 0),
        )

        logger.info(f"응답 완료 ({response.latency}, {response.num_docs}개 문서)")
        return response

    except ValueError as e:
        logger.warning(f"입력 오류: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"서버 오류: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"서버 오류: {str(e)}")


# 라우터 등록
app.include_router(router, prefix="/api", tags=["RAG"])


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.api_d002:app", host="0.0.0.0", port=8000, reload=True, log_level="info"
    )
