import os
from dotenv import load_dotenv
from fastapi import FastAPI
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

# 3. 임베딩 & LLM 초기화
embedding_model = UpstageEmbeddings(api_key=UPSTAGE_API_KEY, model=EMBEDDING_MODEL_NAME)

llm_model = ChatUpstage(api_key=UPSTAGE_API_KEY, model=CHAT_MODEL_NAME)

# 4. ChromaDB 초기화
vectorstore = Chroma(
    persist_directory=CHROMA_DB_DIR, embedding_function=embedding_model
)

# 5. Retriever 설정
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


# 6. RAG 체인 구성 (main.py와 동일하게 수정)
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

# main.py와 동일한 RAG 체인 구성
rag_chain = (
    {"context": retriever | _format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm_model
    | StrOutputParser()
)  # (2)


# 7. 요청 모델
class QueryRequest(BaseModel):
    question: str


# 8. API 엔드포인트
@app.post("/query")
def query_rag(request: QueryRequest):
    # API 요청 시, request.question 값을 rag_chain에 입력으로 전달 (RunnablePassthrough가 처리)
    answer = rag_chain.invoke(request.question)

    answer = rag_chain.invoke({"question": request.question})
    return {"answer": answer}


# 9. 루트 경로
@app.get("/")
def root():
    return {"message": "신혼부부 지원정책 RAG 서버가 실행 중입니다!"}
