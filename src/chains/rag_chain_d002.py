import os
import time
from typing import List, Dict, Any

from langchain_chroma import Chroma
from langchain_upstage import ChatUpstage, UpstageEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv


def load_vector_db(domain: str = "d002") -> Chroma:
    """도메인별 Chroma VectorDB 로드 (Upstage 임베딩 일관화)."""
    # .env 로드 (프로젝트 루트의 .env 파일)
    load_dotenv()
    persist_dir = f"data/{domain}/vector_store"

    api_key = os.getenv("UPSTAGE_API_KEY")
    if not api_key:
        raise ValueError("UPSTAGE_API_KEY 환경변수가 필요합니다")

    embedding_model = os.getenv("UPSTAGE_EMBEDDING_MODEL", "embedding-query")
    embeddings = UpstageEmbeddings(api_key=api_key, model=embedding_model)

    return Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings,
        collection_name=domain,
    )


def load_llm() -> ChatUpstage:
    # .env 로드 (한 번 더 보장)
    load_dotenv()
    api_key = os.getenv("UPSTAGE_API_KEY")
    if not api_key:
        raise ValueError("UPSTAGE_API_KEY 환경변수가 필요합니다")

    model = os.getenv("UPSTAGE_CHAT_MODEL", "solar-1-mini-chat")

    return ChatUpstage(api_key=api_key, model=model)


def _format_docs(docs: List[Any]) -> str:
    lines = []
    for i, d in enumerate(docs, 1):
        source = d.metadata.get("source", "unknown")
        content = (d.page_content or "").strip()

        if len(content) > 2000:
            content = content[:2000] + "..."

        lines.append(f"[문서 {i}] 출처: {source}\n{content}")

    return "\n\n---\n\n".join(lines) if lines else "제공된 문서 없음"


def build_rag_chain(domain: str = "d002"):
    vectordb = load_vector_db(domain)
    llm = load_llm()

    retriever = vectordb.as_retriever(search_kwargs={"k": 5})

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
         당신은 신혼부부 지원정책 도메인 전문가입니다.
         컨텍스트에 근거하지 않은 정보는 답변하지 말고, 모르면 모른다고 답하세요.
         답변 끝에 참고한 출처를 나열하세요.
         컨텍스트:\n{context}
         """.strip(),
            ),
            ("human", "질문: {question}"),
        ]
    )

    chain = (
        {
            "context": retriever | _format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, retriever


def run_rag(query: str, domain: str = "d002", verbose: bool = False) -> Dict[str, Any]:
    start = time.perf_counter()
    chain, retriever = build_rag_chain(domain)
    answer = chain.invoke(query)
    # 출처 수집: 동일 retriever로 다시 호출하여 소스 확보
    docs = retriever.invoke(query)
    sources = list({d.metadata.get("source", "unknown") for d in docs})
    duration_ms = int((time.perf_counter() - start) * 1000)

    if verbose:
        print("🧩 [질문]", query)
        print("⏱️  [소요(ms)]", duration_ms)
        print("💬 [답변]", answer)
        print("📚 [출처]", sources)

    return {"answer": answer, "sources": sources, "duration_ms": duration_ms}


## 실행 테스트
# python -c "from src.chains.rag_chain_d002 import run_rag; res = run_rag('신혼부부 전세자금대출 조건 알려줘','d002'); print(res['answer']); print(res['sources']); print(str(res['duration_ms']) + ' ms')"
