import os
import html
from typing import List, Literal
import re
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_upstage import ChatUpstage, UpstageEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# 1. 환경변수 불러오기
load_dotenv()
UPSTAGE_API_KEY = os.getenv("UPSTAGE_API_KEY")
CHROMA_DB_DIR = os.getenv("CHROMA_DB_DIR", "./chroma_storage")
EMBEDDING_MODEL_NAME = os.getenv("UPSTAGE_EMBEDDING_MODEL", "solar-embedding-1-large")
CHAT_MODEL_NAME = os.getenv("UPSTAGE_CHAT_MODEL", "solar-1-mini-chat")

if not UPSTAGE_API_KEY:
    raise ValueError("Please set UPSTAGE_API_KEY in your .env file")

# 2. FastAPI 앱 인스턴스 생성
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

# 3. 임베딩 & LLM 초기화
embedding_model = UpstageEmbeddings(api_key=UPSTAGE_API_KEY, model=EMBEDDING_MODEL_NAME)

llm_model = ChatUpstage(api_key=UPSTAGE_API_KEY, model=CHAT_MODEL_NAME)

# 4. ChromaDB 초기화
vectorstore = Chroma(
    persist_directory=CHROMA_DB_DIR, embedding_function=embedding_model
)

# 5. Retriever 설정
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


# 6. RAG 체인 구성 (LangChain 0.3 Runnables API)
def _format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "주어진 컨텍스트를 사용하여 사용자 질문에 간결하고 정확하게 답변하세요.\n"
            "모르면 모른다고 답하세요.\n"
            "컨텍스트:\n{context}",
        ),
        ("human", "질문: {question}"),
    ]
)

rag_chain = (
    {"context": retriever | _format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm_model
    | StrOutputParser()
)


# 7. 요청 모델
class QueryRequest(BaseModel):
    question: str


# 8. 루트 경로
@app.get("/")
def root():
    return {"message": "신혼부부 지원정책 RAG 서버가 실행 중입니다!"}


# TODO: D003에서 최종본으로 변경하고 나서 삭제
from src.generation.d003.prompting import build_chat_prompt, format_docs_for_context


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


def _extract_link_info(doc) -> tuple[str, str | None, str]:
    meta = doc.metadata or {}
    src = meta.get("source", "unknown")
    title = meta.get("title") or os.path.basename(str(src)) or "unknown"
    url = meta.get("url") or meta.get("source_url")
    return title, url, src


def _format_markdown_answer(answer: str, docs) -> str:
    prepared = insert_line_breaks_korean(answer.strip())
    content = emphasize_price_terms_md(prepared)
    return content


def _format_html_answer(answer: str, docs) -> str:
    prepared = insert_line_breaks_korean(answer.strip())
    escaped = html.escape(prepared)
    emphasized = emphasize_price_terms_html_escaped(escaped)
    return "<div>" + emphasized.replace("\n", "<br/>") + "</div>"


def emphasize_price_terms_md(text: str) -> str:
    patterns = [
        r"(?:(?:매?월|연)\s*)?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?\s*원",
        r"\d+(?:\.\d+)?\s*%",
    ]

    def repl(m: re.Match) -> str:
        return f"**{m.group(0)}**"

    for p in patterns:
        text = re.sub(p, repl, text)
    return text


def emphasize_price_terms_html_escaped(escaped_text: str) -> str:
    patterns = [
        r"(?:(?:매?월|연)\s*)?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?\s*원",
        r"\d+(?:\.\d+)?\s*%",
    ]

    def repl(m: re.Match) -> str:
        return f"<strong>{m.group(0)}</strong>"

    for p in patterns:
        escaped_text = re.sub(p, repl, escaped_text)
    return escaped_text


def insert_line_breaks_korean(text: str) -> str:
    # Insert breaks after sentence-ending patterns without lookbehind
    text = re.sub(r"(다|요|니다)\.(\s+)", r"\1.\n\n", text)
    text = re.sub(r"\s+(또한|그리고|한편|추가로|더불어|다만|참고로),", r"\n\n\1,", text)
    return text


# 9. API 엔드포인트
@app.post("/query", response_model=D003QueryResponse)
def query_d003(request: D003QueryRequest):
    # Retrieve documents first (to keep sources)
    docs = retriever.invoke(request.question)

    # Build prompt with formatted context
    prompt_d003 = build_chat_prompt()
    context = format_docs_for_context(docs)
    chain = (
        {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
        | prompt_d003
        | llm_model
        | StrOutputParser()
    )
    answer = chain.invoke({"question": request.question, "context": context})

    # Build sources payload
    sources = []
    for d in docs:
        title, url, src = _extract_link_info(d)
        sources.append({"title": title, "url": url, "source": src})

    # Prepare formatted answers
    answer_md = _format_markdown_answer(answer, docs)
    answer_html = _format_html_answer(answer, docs)

    return {
        "answer": answer,
        "answer_md": answer_md,
        "answer_html": answer_html,
        "sources": sources,
    }
