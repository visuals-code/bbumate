import os
from dotenv import load_dotenv
from pathlib import Path


from langchain_upstage import ChatUpstage, UpstageEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# .env 파일 로드
load_dotenv()

# 환경 변수 설정
UPSTAGE_API_KEY = os.getenv("UPSTAGE_API_KEY")
CHROMA_DB_DIR = os.getenv("CHROMA_DB_DIR", "./chroma_storage")
EMBEDDING_MODEL_NAME = os.getenv("UPSTAGE_EMBEDDING_MODEL", "solar-embedding-1-large")
CHAT_MODEL_NAME = os.getenv("UPSTAGE_CHAT_MODEL", "solar-1-mini-chat")


if not UPSTAGE_API_KEY:
    raise ValueError("UPSTAGE_API_KEY가 .env 파일에 설정되지 않았습니다.")

# 1. 임베딩 & LLM 초기화
embedding_model = UpstageEmbeddings(api_key=UPSTAGE_API_KEY, model=EMBEDDING_MODEL_NAME)
llm_model = ChatUpstage(api_key=UPSTAGE_API_KEY, model=CHAT_MODEL_NAME)

# 2. ChromaDB 초기화 (d005 파이프라인이 생성한 DB 로드)
vectorstore = Chroma(
    persist_directory=CHROMA_DB_DIR, embedding_function=embedding_model
)

# 3. Retriever 설정
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


# 4. RAG 체인 구성 (오류 수정된 로직 적용)
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

# Retriever에 문자열만 입력되도록 (lambda x: x["question"]) 사용
rag_chain = (
    {
        "context": (lambda x: x["question"]) | retriever | _format_docs,
        "question": (lambda x: x["question"]),
    }
    | prompt
    | llm_model
    | StrOutputParser()
)


def run_test_queries(queries):
    """
    쿼리를 실행하고 LangSmith에 결과를 기록합니다.
    """
    print("=" * 60)
    print("RAG 체인 테스트 시작 (LangSmith 트레이싱 활성화)")
    print("=" * 60)

    for i, query in enumerate(queries):
        # LangSmith에 질문 전체가 단일 "run"으로 기록됩니다.
        print(f"\n[{i+1}/{len(queries)}] 질문: {query}")

        try:

            result = rag_chain.invoke({"question": query})
            print("응답:")
            print(result)
        except Exception as e:
            print(f"**오류 발생**: {e}")


if __name__ == "__main__":
    # 테스트할 질문 목록
    test_queries = [
        "신혼부부 백화점",
        "신혼부부 백화점 혜택"
        "이 문서는 어디에서 왔는지 출처를 알려줄 수 있어?",  # RAG의 한계 테스트
    ]

    run_test_queries(test_queries)
