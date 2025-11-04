import os
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

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


from src.chains.d003.chain import answer_question
from src.generation.d003.prompting import (
    extract_link_info,
    format_answer_md,
    format_answer_html,
)


class D003QueryRequest(BaseModel):
    question: str


class SourceItem(BaseModel):
    title: str
    url: str | None = None
    source: str


class D003QueryResponse(BaseModel):
    answer: str
    answer_md: str
    answer_html: str
    sources: list[SourceItem]


# API 엔드포인트
@app.post("/query", response_model=D003QueryResponse)
def query_d003(request: D003QueryRequest):
    # d003 체인 모듈을 사용해 답변 생성
    answer, docs = answer_question(request.question, k=3)

    # 출처 정보 구성
    sources = []
    for d in docs:
        title, url, src = extract_link_info(d)
        sources.append({"title": title, "url": url, "source": src})

    # 최종 출력용 포맷(마크다운/HTML) 생성
    answer_md = format_answer_md(answer)
    answer_html = format_answer_html(answer)

    return {
        "answer": answer,
        "answer_md": answer_md,
        "answer_html": answer_html,
        "sources": sources,
    }
