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
    """문서들을 컨텍스트 형식으로 포맷팅 (요약형 사용)."""
    lines = []
    for i, d in enumerate(docs, 1):
        source = d.metadata.get("source", "unknown")
        content = (d.page_content or "").strip()

        # 성능 개선: 문서 길이 제한 (요약형)
        if len(content) > 1500:
            content = content[:1500] + "..."

        lines.append(f"[문서 {i}] 출처: {source}\n{content}")

    return "\n\n---\n\n".join(lines) if lines else "제공된 문서 없음"


def check_domain_relevance(question: str, llm_model) -> bool:
    """질문이 신혼부부 지원정책 도메인과 관련 있는지 체크 (관련 있으면 True)."""
    domain_check_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "당신은 질문 분류 전문가입니다. "
                "질문이 신혼부부 지원정책(주거, 대출, 전세자금, 구매자금, 신혼부부 혜택 등)과 관련이 있으면 'Y', "
                "날씨, 요리, 일반 뉴스 등 무관한 주제면 'N'으로 답하세요.",
            ),
            ("human", "질문: {question}\n\n신혼부부 지원정책 관련 여부 (Y/N):"),
        ]
    )
    domain_check_chain = domain_check_prompt | llm_model | StrOutputParser()
    
    try:
        result = domain_check_chain.invoke({"question": question})
        return "Y" in result.upper()
    except Exception:
        # 평가 실패 시 관련 있다고 가정 (안전장치)
        return True


def check_question_ambiguity(question: str, llm_model) -> bool:
    """질문의 모호성을 체크 (모호하면 True, 명확하면 False)."""
    ambiguity_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "당신은 질문의 명확성을 판단하는 평가자입니다. "
                "질문이 매우 모호하거나 거의 아무 정보도 없는 경우에만 'Y', 그 외에는 'N'으로 답하세요. "
                "예시 - 명확한 질문 (N): '신혼부부 전세자금대출 조건', '전세자금 대출 한도', '신혼부부 대출 금리' 등 "
                "예시 - 모호한 질문 (Y): '대출', '조건 알려줘', '도와줘' 등 "
                "일반적인 조건/정보 문의는 명확한 것으로 판단하세요. 개인 맞춤형 답변을 위해 지역/소득 정보가 필요한 경우에만 모호하다고 판단하세요.",
            ),
            ("human", "질문: {question}"),
        ]
    )
    ambiguity_chain = ambiguity_prompt | llm_model | StrOutputParser()
    
    try:
        result = ambiguity_chain.invoke({"question": question})
        return "Y" in result.upper()
    except Exception:
        # 평가 실패 시 명확하다고 가정 (안전장치)
        return False


def clarify_question(question: str, llm_model) -> str:
    """모호한 질문을 명확화하기 위한 질문을 생성 (Re-ask)."""
    clarify_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "당신은 신혼부부 지원정책 상담사입니다. "
                "사용자의 모호한 질문에 대해, 답변에 꼭 필요한 핵심 정보 1-2가지만 간결하게 물어보세요. "
                "가능하면 한 문장으로, 최대 2개의 핵심 정보만 요청하세요. "
                "예: '거주 지역(서울/수도권/지방)과 주거형태(전세/매매)를 알려주세요.' "
                "너무 구체적이거나 여러 질문을 나열하지 마세요.",
            ),
            ("human", "모호한 질문: {question}\n\n간결한 명확화 질문(1-2개 핵심 정보만):"),
        ]
    )
    clarify_chain = clarify_prompt | llm_model | StrOutputParser()
    
    try:
        clarified = clarify_chain.invoke({"question": question})
        return clarified.strip()
    except Exception:
        return "답변에 필요한 핵심 정보(지역, 주거형태 등)를 알려주세요."


def check_docs_can_answer(question: str, docs: List[Any], llm_model) -> bool:
    """검색된 문서들이 질문에 답할 수 있는지 판단 (Y/N)."""
    if not docs:
        return False
    
    # 문서 내용 요약
    docs_summary = "\n\n".join([doc.page_content[:500] for doc in docs[:3]])
    
    answer_check_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "당신은 검색된 문서가 사용자 질문에 충분히 답할 수 있는지를 판단하는 평가자입니다. "
                "문서 내용이 질문에 대한 답을 제공할 수 있으면 'Y', 충분한 정보가 없으면 'N'으로 답하세요.",
            ),
            ("human", "질문: {question}\n\n검색된 문서 요약:\n{docs_summary}\n\n답변 가능 여부 (Y/N):"),
        ]
    )
    answer_check_chain = answer_check_prompt | llm_model | StrOutputParser()
    
    try:
        result = answer_check_chain.invoke({"question": question, "docs_summary": docs_summary})
        return "Y" in result.upper()
    except Exception:
        # 평가 실패 시 답할 수 있다고 가정 (안전장치)
        return True


def rewrite_query(question: str, llm_model) -> str:
    """웹 검색에 적합하도록 쿼리를 재작성 (Re-write query)."""
    rewrite_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "당신은 웹 검색 쿼리 전문가입니다. "
                "사용자의 질문을 웹 검색에 최적화된 검색 쿼리로 재작성하세요. "
                "핵심 키워드를 포함하고, 자연어 질문을 검색 친화적인 형태로 변환하세요.",
            ),
            ("human", "원래 질문: {question}\n\n웹 검색 쿼리로 재작성:"),
        ]
    )
    rewrite_chain = rewrite_prompt | llm_model | StrOutputParser()
    
    try:
        rewritten = rewrite_chain.invoke({"question": question})
        return rewritten.strip()
    except Exception:
        return question


def web_search(query: str) -> str:
    """웹 검색 실행 (현재는 더미 구현, 향후 실제 웹 검색 API 통합 가능)."""
    # TODO: 실제 웹 검색 API 통합 (예: Tavily, Serper, Google Custom Search 등)
    # 현재는 더미 응답 반환
    return f"[웹 검색 결과] '{query}'에 대한 최신 정보를 찾을 수 없습니다. 현재 시스템에는 저장된 문서만 있습니다."


def generate_with_web_context(question: str, web_results: str, llm_model) -> str:
    """웹 검색 결과를 컨텍스트로 사용하여 답변 생성."""
    web_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "당신은 신혼부부 지원정책 도메인 전문가입니다. "
                "웹 검색 결과를 참고하여 질문에 답변하세요. "
                "정보가 충분하지 않으면 모른다고 답하세요.",
            ),
            ("human", "질문: {question}\n\n웹 검색 결과:\n{web_results}"),
        ]
    )
    web_chain = web_prompt | llm_model | StrOutputParser()
    
    try:
        answer = web_chain.invoke({"question": question, "web_results": web_results})
        return answer.strip()
    except Exception:
        return "죄송합니다. 웹 검색 결과를 기반으로 답변을 생성할 수 없습니다."


def _grade_single_doc(question: str, doc_content: str, grade_chain) -> bool:
    """단일 문서의 관련성을 평가 (캐싱된 함수)."""
    try:
        grade = grade_chain.invoke({"question": question, "context": doc_content})
        return "Y" in grade.upper()
    except Exception:
        # 평가 실패 시 관련 있다고 가정 (안전장치)
        return True


def grade_docs(question: str, docs: List[Any], llm_model) -> List[Any]:
    """retrieved 문서 중 관련도 높은 것만 필터링 (Grade 단계)."""
    if not docs:
        return []

    # Grade 프롬프트 및 체인 구성
    grade_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "당신은 검색된 문서가 질문과 얼마나 관련 있는지를 판단하는 평가자입니다. "
                "문서가 질문과 관련이 있으면 'Y', 관련이 없으면 'N'으로 답하세요.",
            ),
            ("human", "질문: {question}\n\n문서 내용:\n{context}"),
        ]
    )
    grade_chain = grade_prompt | llm_model | StrOutputParser()

    filtered = []
    for doc in docs:
        # 문서 내용 요약 (성능 개선)
        content = (doc.page_content or "").strip()[:1500]

        # 캐싱 키 생성 (질문 + 문서 해시)
        cache_key = f"{question[:100]}|{hash(content)}"

        # 평가 실행
        is_relevant = _grade_single_doc(question, content, grade_chain)

        if is_relevant:
            filtered.append(doc)

    return filtered


def build_rag_chain(domain: str = "d002", use_grade: bool = True):
    """RAG 체인 구성 (Grade 단계 선택 가능)."""
    vectordb = load_vector_db(domain)
    llm = load_llm()

    # 성능 개선: k 값 줄이기 (5 → 3)
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})

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

    return chain, retriever, llm, use_grade


def run_rag(
    query: str,
    domain: str = "d002",
    verbose: bool = False,
    use_grade: bool = True,
    use_clarification: bool = True,
) -> Dict[str, Any]:
    """RAG 파이프라인 실행 (Clarification → Retrieve → Grade → Generate)."""
    start = time.perf_counter()
    chain, retriever, llm, grade_enabled = build_rag_chain(domain, use_grade=use_grade)

    # Domain Check: 신혼부부 지원정책 관련 질문인지 체크
    is_relevant = check_domain_relevance(query, llm)
    if not is_relevant:
        if verbose:
            print("[도메인 관련성 없음]")
        
        return {
            "answer": "죄송합니다. 신혼부부 지원정책(주거, 대출, 전세자금, 구매자금 등) 관련 질문만 답변드릴 수 있습니다. 다른 주제의 질문은 처리할 수 없습니다.",
            "sources": [],
            "duration_ms": int((time.perf_counter() - start) * 1000),
            "num_docs": 0,
            "clarification_needed": False,
            "web_search_used": False,
        }
    
    # Clarification: 질문 모호성 체크 및 명확화
    clarified_query = query
    clarification_needed = False
    
    if use_clarification:
        is_ambiguous = check_question_ambiguity(query, llm)
        if is_ambiguous:
            clarification_question = clarify_question(query, llm)
            clarification_needed = True
            
            if verbose:
                print("[질문 모호성 감지]")
                print(f"[명확화 요청]: {clarification_question}")
            
            # 명확화 질문을 답변으로 반환하고 종료
            return {
                "answer": f"질문을 더 명확히 해주세요.\n\n{clarification_question}",
                "sources": [],
                "duration_ms": int((time.perf_counter() - start) * 1000),
                "num_docs": 0,
                "clarification_needed": True,
                "web_search_used": False,
            }

    # Retrieve: 초기 문서 검색
    initial_docs = retriever.invoke(clarified_query)

    # Grade: 관련성 높은 문서만 필터링
    if grade_enabled and initial_docs:
        graded_docs = grade_docs(clarified_query, initial_docs, llm)
        if not graded_docs:
            # 모든 문서가 관련 없다고 판단되면 초기 문서 사용 (안전장치)
            graded_docs = initial_docs[:2]  # 최대 2개만
    else:
        graded_docs = initial_docs

    # Can retrieved documents answer?: 문서로 답변 가능한지 판단
    use_web_search = False
    if graded_docs:
        can_answer = check_docs_can_answer(clarified_query, graded_docs, llm)
        
        if not can_answer:
            # 문서로 답변 불가능 → Re-write query → Web Search 경로
            use_web_search = True
            
            if verbose:
                print("[문서로 답변 불가능 감지]")
            
            # Re-write query
            rewritten_query = rewrite_query(clarified_query, llm)
            
            if verbose:
                print(f"[쿼리 재작성]: {rewritten_query}")
            
            # Web Search
            web_results = web_search(rewritten_query)
            
            if verbose:
                print(f"[웹 검색 완료]")
            
            # Generate with Web Search results
            answer = generate_with_web_context(clarified_query, web_results, llm)
            sources = ["웹 검색"]
        else:
            # Generate: 필터링된 문서로 답변 생성
            context = _format_docs(graded_docs)
            generate_prompt = ChatPromptTemplate.from_messages(
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
            generate_chain = generate_prompt | llm | StrOutputParser()
            answer = generate_chain.invoke({"question": clarified_query, "context": context})
            sources = list({d.metadata.get("source", "unknown") for d in graded_docs})
    else:
        answer = "관련된 문서를 찾을 수 없습니다."
        sources = []

    # sources는 위에서 이미 설정됨 (웹 검색 경로 또는 문서 경로)
    duration_ms = int((time.perf_counter() - start) * 1000)

    if verbose:
        print("[질문]", query)
        if clarification_needed:
            print("[명확화 완료]")
        print(f"[검색 문서] {len(initial_docs)}개 → {len(graded_docs)}개 (Grade)")
        if use_web_search:
            print("[웹 검색 경로 사용]")
        print(f"[소요(ms)]", duration_ms)
        print("[답변]", answer)
        print("[출처]", sources)

    return {
        "answer": answer,
        "sources": sources,
        "duration_ms": duration_ms,
        "num_docs": len(graded_docs) if not use_web_search else 0,
        "clarification_needed": False,
        "web_search_used": use_web_search,
    }


## 실행 테스트
# python -c "from src.chains.rag_chain_d002 import run_rag; res = run_rag('신혼부부 전세자금대출 조건 알려줘','d002'); print(res['answer']); print(res['sources']); print(str(res['duration_ms']) + ' ms')"
