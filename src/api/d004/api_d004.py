# -*- coding: utf-8 -*-
import os
import sys
from pathlib import Path
from typing import Optional, List, Dict
from contextlib import asynccontextmanager
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# 프로젝트 경로 설정 (src/api/d004/api_d004.py 기준)
current_file = Path(__name__).resolve()
api_d004_dir = current_file.parent  # src/api/d004
api_dir = api_d004_dir.parent  # src/api
src_dir = api_dir.parent  # src
project_root = src_dir.parent  # 프로젝트 루트

# d004 관련 모듈 경로 추가
chains_d004_path = src_dir / "chains" / "d004"
generation_d004_path = src_dir / "generation" / "d004"
retrieval_d004_path = src_dir / "retrieval" / "d004"


try:
    from src.chains.d004.chain import AdvancedRAGChain

    print("✅ AdvancedRAGChain import 성공!")
except ImportError as e:
    print(f"❌ Import 실패: {e}")
    print(f"📍 다음 경로를 확인하세요: {chains_d004_path / 'chain.py'}")
    raise

# 환경 변수 로드
load_dotenv()

# RAG 체인 초기화 (전역 변수)
rag_chain: Optional[AdvancedRAGChain] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """애플리케이션 시작/종료 시 실행되는 lifespan 함수"""
    global rag_chain
    try:
        print("🚀 RAG 체인 초기화 중...")
        rag_chain = AdvancedRAGChain(max_rewrite_attempts=1)
        print("✅ RAG 체인 초기화 성공")
    except Exception as e:
        print(f"❌ RAG 체인 초기화 실패: {e}")
        raise

    yield

    print("🔚 애플리케이션 종료")


# FastAPI 앱 생성
app = FastAPI(
    title="신혼부부 혜택 정보 RAG API (D004)",
    description="d004 프로젝트 전용 RAG API - 신혼부부를 위한 혜택 정보 검색",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==============================================================================
# Pydantic 모델
# ==============================================================================


class QueryRequest(BaseModel):
    question: str = Field(..., description="사용자 질문", min_length=1, max_length=500)
    region: Optional[str] = Field(None, description="거주 지역 (예: 서울)")
    housing_type: Optional[str] = Field(None, description="주거 형태 (예: 아파트)")

    model_config = {
        "json_schema_extra": {
            "example": {
                "question": "신혼부부 전세자금대출 조건 알려줘",
                "region": "서울",
                "housing_type": "아파트",
            }
        }
    }


class Source(BaseModel):
    title: str = Field(..., description="문서 제목")
    url: Optional[str] = Field(None, description="원본 URL")
    source: str = Field(..., description="소스 파일 또는 출처 식별자")


class QueryResponse(BaseModel):
    answer: str = Field(..., description="사용자 질문에 대한 답변")
    sources: List[Source] = Field(..., description="답변에 사용된 문서 출처 리스트")
    metadata: Dict = Field(..., description="RAG 실행 관련 메타데이터")

    model_config = {
        "json_schema_extra": {
            "example": {
                "answer": "신혼부부 전세자금대출의 주요 조건은...",
                "sources": [
                    {
                        "title": "버팀목 전세자금",
                        "url": "https://gov.kr/policy",
                        "source": "gov_policy.pdf",
                    }
                ],
                "metadata": {"routing_status": "CLEAR", "documents_retrieved": 5},
            }
        }
    }


class VectorStoreStatus(BaseModel):
    status: str = Field(..., description="벡터스토어 상태")
    document_count: int = Field(0, description="저장된 문서 수")


class HealthResponse(BaseModel):
    status: str = Field(..., description="API 상태")
    vectorstore: VectorStoreStatus
    message: str


# ==============================================================================
# 엔드포인트
# ==============================================================================


@app.get("/", response_model=HealthResponse)
async def health_check():
    """API 상태 확인"""
    if rag_chain is None:
        return HealthResponse(
            status="error",
            vectorstore=VectorStoreStatus(status="not_initialized", document_count=0),
            message="RAG 체인이 초기화되지 않았습니다",
        )

    try:
        collection = rag_chain.retriever.vectorstore.get()
        doc_count = len(collection["ids"])

        return HealthResponse(
            status="healthy",
            vectorstore=VectorStoreStatus(status="active", document_count=doc_count),
            message="API가 정상적으로 작동 중입니다",
        )
    except Exception as e:
        return HealthResponse(
            status="degraded",
            vectorstore=VectorStoreStatus(status="error", document_count=0),
            message=f"벡터스토어 확인 중 오류: {str(e)}",
        )


@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """사용자 질문을 처리하고 답변 및 출처를 반환"""
    if rag_chain is None:
        raise HTTPException(status_code=503, detail="RAG 체인이 초기화되지 않았습니다")

    try:
        result = rag_chain.invoke(
            question=request.question,
            region=request.region,
            housing_type=request.housing_type,
        )

        return QueryResponse(
            answer=result["answer"],
            sources=[
                Source(title=src["title"], url=src.get("url"), source=src["source"])
                for src in result.get("sources", [])
            ],
            metadata={
                "original_question": result.get("original_question"),
                "final_question": result.get("final_question"),
                "routing_status": result.get("routing_status"),
                "documents_retrieved": result.get("documents_retrieved"),
                "relevant_documents": result.get("relevant_documents"),
                "source": result.get("source"),
                "rewrite_count": result.get("rewrite_count"),
            },
        )

    except Exception as e:
        print(f"🚨 RAG 처리 중 오류: {type(e).__name__}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"처리 중 오류 발생: {str(e)}")


@app.get("/stats")
async def get_stats():
    """벡터스토어 통계 정보"""
    if rag_chain is None:
        raise HTTPException(status_code=503, detail="RAG 체인이 초기화되지 않았습니다")

    try:
        collection = rag_chain.retriever.vectorstore.get()
        metadatas = collection.get("metadatas", [])
        source_files = set()

        for meta in metadatas:
            if "source_file" in meta:
                source_files.add(Path(meta["source_file"]).name)

        return {
            "total_documents": len(collection["ids"]),
            "unique_sources": len(source_files),
            "source_files": sorted(list(source_files)),
            "collection_name": rag_chain.collection_name,
            "db_path": rag_chain.db_path,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"통계 조회 중 오류: {str(e)}")


@app.get("/search")
async def search_documents(
    query: str = Query(..., description="검색 쿼리", min_length=1),
    k: int = Query(3, description="반환할 문서 수", ge=1, le=10),
):
    """벡터스토어 직접 검색 (테스트용)"""
    if rag_chain is None:
        raise HTTPException(status_code=503, detail="RAG 체인이 초기화되지 않았습니다")

    try:
        docs = rag_chain.retriever.vectorstore.similarity_search(query, k=k)

        return {
            "query": query,
            "count": len(docs),
            "documents": [
                {
                    "content": doc.page_content[:200] + "...",
                    "metadata": doc.metadata,
                    "has_url": "url" in doc.metadata,
                }
                for doc in docs
            ],
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"검색 중 오류: {str(e)}")


# ==============================================================================
# 실행
# uvicorn src.api.d004.api_d004:app --reload
# ==============================================================================
