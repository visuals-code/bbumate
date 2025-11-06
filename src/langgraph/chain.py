"""LangGraph 기반 RAG 체인 정의.

기존 chains/d002/rag_chain.py의 LangChain 구조를 LangGraph로 변환.
retrieval, generation은 d002 폴더의 일반 함수들을 import해서 사용.
"""

import os
import time
from typing import TypedDict, List, Dict, Any, Optional, Literal
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_upstage import ChatUpstage, UpstageEmbeddings

# d002 폴더의 일반 함수들 import (기존과 동일)
from src.utils.d002.loaders import load_llm
from src.utils.d002.context_extraction import apply_region_housing_priority
from src.retrieval.d002.grader import grade_docs
from src.generation.d002.validation import is_question_clear, validate_question
from src.generation.d002.generator import (
    generate_with_docs_context,
    generate_with_web_context,
)
from src.generation.d002.web_search import rewrite_query, web_search

load_dotenv()


# State 정의
class RAGState(TypedDict):
    """RAG 파이프라인 상태."""
    # 입력 파라미터
    question: str
    region: Optional[str]
    housing_type: Optional[str]
    verbose: bool
    use_grade: bool
    use_validation: bool
    k: int

    # 중간 상태
    is_valid: bool
    validation_reason: str
    clarification_question: str
    initial_docs: List[Document]
    graded_docs: List[Document]
    context: str
    rewritten_query: str
    web_results: str
    web_metadata: List[Dict[str, str]]

    # 출력
    answer: str
    sources: List[Any]
    duration_ms: int
    num_docs: int
    clarification_needed: bool
    web_search_used: bool

    # 내부
    _start_time: float
    _retriever: Any
    _llm: Any
    _final_region: Optional[str]
    _final_housing_type: Optional[str]


def load_unified_vector_db() -> Chroma:
    """통합 벡터 DB 로드 (unified_rag_collection 사용)."""
    db_path = os.getenv("CHROMA_DB_DIR", "./chroma_storage")
    collection_name = os.getenv("COLLECTION_NAME", "unified_rag_collection")

    api_key = os.getenv("UPSTAGE_API_KEY")
    if not api_key:
        raise ValueError("UPSTAGE_API_KEY 환경변수가 필요합니다")

    embedding_model = os.getenv("UPSTAGE_EMBEDDING_MODEL", "solar-embedding-1-large")
    embeddings = UpstageEmbeddings(api_key=api_key, model=embedding_model)

    return Chroma(
        persist_directory=db_path,
        embedding_function=embeddings,
        collection_name=collection_name,
    )


def _format_docs(docs: List[Any]) -> str:
    """문서들을 컨텍스트 형식으로 포맷팅."""
    lines = []
    for i, d in enumerate(docs, 1):
        content = (d.page_content or "").strip()
        if len(content) > 1500:
            content = content[:1500] + "..."
        lines.append(f"[문서 {i}]\n{content}")
    return "\n\n---\n\n".join(lines) if lines else "제공된 문서 없음"


# Node 함수들 (기존 rag_chain.py의 로직을 Node로 분리)
def initialize_node(state: RAGState) -> RAGState:
    """초기화: retriever, llm 로드 및 컨텍스트 처리 (통합 DB 사용)."""
    if state.get("verbose"):
        print("\n[Initialize] 초기화 시작...")

    # 통합 DB 직접 로드
    vectordb = load_unified_vector_db()
    llm = load_llm()
    retriever = vectordb.as_retriever(search_kwargs={"k": state.get("k", 3)})

    # 지역/주거형태 우선순위 적용
    final_region, final_housing_type = apply_region_housing_priority(
        state["question"], state.get("region"), state.get("housing_type")
    )

    if state.get("verbose") and (final_region or final_housing_type):
        print(
            f"[컨텍스트] 지역: {final_region or '미지정'}, "
            f"주거형태: {final_housing_type or '미지정'}"
        )

    return {
        **state,
        "_start_time": time.perf_counter(),
        "_retriever": retriever,
        "_llm": llm,
        "_final_region": final_region,
        "_final_housing_type": final_housing_type,
        "is_valid": True,
        "validation_reason": "",
        "clarification_question": "",
        "clarification_needed": False,
        "web_search_used": False,
    }


def validate_node(state: RAGState) -> RAGState:
    """질문 검증: 도메인 관련성 + 명확성 (d002/validation.py 함수 사용)."""
    if state.get("verbose"):
        print("\n[Validate] 질문 검증 중...")

    if not state.get("use_validation", True):
        return state

    # is_question_clear: d002 폴더의 일반 함수
    if is_question_clear(state["question"]):
        if state.get("verbose"):
            print("[Validation 스킵: 명확한 질문]")
        return state

    # validate_question: d002 폴더의 일반 함수
    is_valid, reason, clarification_q = validate_question(
        state["question"], state["_llm"]
    )

    if not is_valid:
        if state.get("verbose"):
            if reason == "domain":
                print("[도메인 관련성 없음]")
            elif reason == "ambiguity":
                print("[질문 모호성 감지]")
                print(f"[명확화 요청]: {clarification_q}")

    return {
        **state,
        "is_valid": is_valid,
        "validation_reason": reason,
        "clarification_question": clarification_q,
        "clarification_needed": (not is_valid and reason == "ambiguity"),
    }


def retrieve_node(state: RAGState) -> RAGState:
    """문서 검색 (retriever 사용)."""
    if state.get("verbose"):
        print("\n[Retrieve] 문서 검색 중...")

    initial_docs = state["_retriever"].invoke(state["question"])

    if state.get("verbose") and initial_docs:
        print(f"[검색된 문서 (Grade 전)] {len(initial_docs)}개")
        for i, doc in enumerate(initial_docs[:3], 1):
            content_preview = (doc.page_content or "")[:200]
            source = doc.metadata.get("source", "unknown")
            print(f"  [{i}] {source}: {content_preview}...")

    return {
        **state,
        "initial_docs": initial_docs,
    }


def grade_node(state: RAGState) -> RAGState:
    """문서 평가: 관련성 필터링 (d002/grader.py 함수 사용)."""
    if state.get("verbose"):
        print("\n[Grade] 문서 관련성 평가 중...")

    initial_docs = state.get("initial_docs", [])

    if not state.get("use_grade", True) or not initial_docs:
        graded_docs = initial_docs
    else:
        # grade_docs: d002 폴더의 일반 함수
        graded_docs = grade_docs(state["question"], initial_docs, state["_llm"])
        if state.get("verbose"):
            print(f"[Grade 결과] {len(initial_docs)}개 → {len(graded_docs)}개")

    return {
        **state,
        "graded_docs": graded_docs,
        "num_docs": len(graded_docs),
    }


def generate_docs_node(state: RAGState) -> RAGState:
    """문서 기반 답변 생성 (d002/generator.py 함수 사용)."""
    if state.get("verbose"):
        print("\n[Generate Docs] 문서 기반 답변 생성 중...")

    graded_docs = state.get("graded_docs", [])
    context = _format_docs(graded_docs)

    # generate_with_docs_context: d002 폴더의 일반 함수
    answer = generate_with_docs_context(
        state["question"],
        context,
        state["_llm"],
        state.get("_final_region"),
        state.get("_final_housing_type"),
    )

    source_set = {d.metadata.get("source", "unknown") for d in graded_docs}
    sources = list(source_set)

    return {
        **state,
        "context": context,
        "answer": answer,
        "sources": sources,
        "web_search_used": False,
    }


def rewrite_node(state: RAGState) -> RAGState:
    """쿼리 재작성 (d002/web_search.py 함수 사용)."""
    if state.get("verbose"):
        print("\n[Rewrite] 쿼리 재작성 중...")

    # rewrite_query: d002 폴더의 일반 함수
    rewritten = rewrite_query(state["question"], state["_llm"])

    if state.get("verbose"):
        print(f"[쿼리 재작성]: {rewritten}")

    return {
        **state,
        "rewritten_query": rewritten,
    }


def web_search_node(state: RAGState) -> RAGState:
    """웹 검색 (d002/web_search.py 함수 사용)."""
    if state.get("verbose"):
        print("\n[Web Search] 웹 검색 중...")

    rewritten = state.get("rewritten_query", state["question"])

    # web_search: d002 폴더의 일반 함수
    web_results, web_metadata = web_search(rewritten)

    if state.get("verbose"):
        print("[웹 검색 완료]")

    return {
        **state,
        "web_results": web_results,
        "web_metadata": web_metadata,
        "web_search_used": True,
    }


def generate_web_node(state: RAGState) -> RAGState:
    """웹 검색 결과 기반 답변 생성 (d002/generator.py 함수 사용)."""
    if state.get("verbose"):
        print("\n[Generate Web] 웹 검색 결과 기반 답변 생성 중...")

    # generate_with_web_context: d002 폴더의 일반 함수
    answer = generate_with_web_context(
        state["question"],
        state.get("web_results", ""),
        state["_llm"],
        state.get("_final_region"),
        state.get("_final_housing_type"),
    )

    web_metadata = state.get("web_metadata", [])
    sources = (
        [{"title": item["title"], "url": item["url"]} for item in web_metadata]
        if web_metadata
        else ["웹 검색"]
    )

    return {
        **state,
        "answer": answer,
        "sources": sources,
    }


def finalize_node(state: RAGState) -> RAGState:
    """최종 결과 처리."""
    duration_ms = int((time.perf_counter() - state["_start_time"]) * 1000)

    if state.get("verbose"):
        print("\n[Finalize] 완료!")
        print(f"[질문] {state['question']}")
        print(f"[소요(ms)] {duration_ms}")
        print(f"[답변] {state.get('answer', '')}")
        print(f"[출처] {state.get('sources', [])}")

    return {
        **state,
        "duration_ms": duration_ms,
    }


# Conditional Edge 함수들 (기존 rag_chain.py의 if/else 로직을 조건부 Edge로 변환)
def should_continue_after_validate(
    state: RAGState,
) -> Literal["retrieve", "end"]:
    """Validation 후 라우팅 (기존: if not is_valid → return)."""
    if state.get("is_valid", True):
        return "retrieve"
    return "end"


def should_continue_after_grade(
    state: RAGState,
) -> Literal["generate_docs", "rewrite"]:
    """Grade 후 라우팅 (기존: if graded_docs → Generate, else → Web Search)."""
    graded_docs = state.get("graded_docs", [])
    if graded_docs:
        return "generate_docs"
    return "rewrite"


def should_continue_after_generate_docs(
    state: RAGState,
) -> Literal["rewrite", "finalize"]:
    """Generate Docs 후 라우팅 (기존: "정보 없음" 패턴 감지 → Web Search)."""
    answer = state.get("answer", "")

    no_info_patterns = [
        "제공된 문서에는 해당 정보가 없습니다",
        "제공된 문서에는",
        "해당 정보가 없습니다",
        "정보가 없습니다",
        "찾을 수 없습니다",
    ]

    answer_lower = answer.lower()
    if any(pattern in answer_lower for pattern in no_info_patterns):
        if state.get("verbose"):
            print("[Generate 결과: 정보 없음 → Web Search 경로로 전환]")
        return "rewrite"
    return "finalize"


def build_rag_graph() -> StateGraph:
    """RAG 파이프라인 그래프 구성.

    기존 rag_chain.py의 플로우:
    1. Question Validation → 2. Retrieve → 3. Grade → 4. Generate (또는 Web Search)
    """
    workflow = StateGraph(RAGState)

    # Nodes 추가
    workflow.add_node("initialize", initialize_node)
    workflow.add_node("validate", validate_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade", grade_node)
    workflow.add_node("generate_docs", generate_docs_node)
    workflow.add_node("rewrite", rewrite_node)
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("generate_web", generate_web_node)
    workflow.add_node("finalize", finalize_node)

    # Entry point
    workflow.set_entry_point("initialize")

    # Edges 추가 (기존 rag_chain.py의 플로우 그대로)
    workflow.add_edge("initialize", "validate")
    workflow.add_conditional_edges(
        "validate",
        should_continue_after_validate,
        {
            "retrieve": "retrieve",
            "end": "finalize",
        },
    )
    workflow.add_edge("retrieve", "grade")
    workflow.add_conditional_edges(
        "grade",
        should_continue_after_grade,
        {
            "generate_docs": "generate_docs",
            "rewrite": "rewrite",
        },
    )
    workflow.add_conditional_edges(
        "generate_docs",
        should_continue_after_generate_docs,
        {
            "rewrite": "rewrite",
            "finalize": "finalize",
        },
    )
    workflow.add_edge("rewrite", "web_search")
    workflow.add_edge("web_search", "generate_web")
    workflow.add_edge("generate_web", "finalize")
    workflow.add_edge("finalize", END)

    return workflow.compile()


def answer_question(
    question: str,
    k: int = 3,
    use_grade: bool = True,
    use_validation: bool = True,
    region: Optional[str] = None,
    housing_type: Optional[str] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """질문에 답변하고 문서 출처를 반환 (LangGraph 버전, 통합 DB 사용).

    기존 rag_chain.py의 run_rag 함수를 LangGraph로 변환.

    Args:
        question: 사용자 질문
        k: 검색할 문서 개수 (기본값: 3)
        use_grade: 문서 평가 사용 여부 (기본값: True)
        use_validation: 질문 검증 사용 여부 (기본값: True)
        region: 지역 정보 (선택)
        housing_type: 주거 형태 (선택)
        verbose: 상세 로그 출력 여부 (기본값: False)

    Returns:
        {
            "answer": str,
            "sources": List[str],
            "duration_ms": int,
            "num_docs": int,
            "clarification_needed": bool,
            "web_search_used": bool,
        }

    플로우 (기존 rag_chain.py와 동일):
        1. Initialize → Validate → Retrieve → Grade
        2. Grade 성공 → Generate Docs → (정보 있으면) Finalize
        3. Grade 실패 또는 Generate Docs 실패 → Rewrite → Web Search → Generate Web → Finalize
    """
    initial_state: RAGState = {
        "question": question,
        "region": region,
        "housing_type": housing_type,
        "verbose": verbose,
        "use_grade": use_grade,
        "use_validation": use_validation,
        "k": k,
        "is_valid": True,
        "validation_reason": "",
        "clarification_question": "",
        "initial_docs": [],
        "graded_docs": [],
        "context": "",
        "rewritten_query": "",
        "web_results": "",
        "web_metadata": [],
        "answer": "",
        "sources": [],
        "duration_ms": 0,
        "num_docs": 0,
        "clarification_needed": False,
        "web_search_used": False,
        "_start_time": 0.0,
        "_retriever": None,
        "_llm": None,
        "_final_region": None,
        "_final_housing_type": None,
    }

    graph = build_rag_graph()
    final_state = graph.invoke(initial_state)

    # Validation 실패 시 특별 처리 (기존 rag_chain.py와 동일)
    if not final_state.get("is_valid", True):
        reason = final_state.get("validation_reason", "")
        if reason == "domain":
            final_state["answer"] = (
                "죄송합니다. 신혼부부 지원정책(주거, 대출, 전세자금, 구매자금 등) "
                "관련 질문만 답변드릴 수 있습니다. 다른 주제의 질문은 처리할 수 없습니다."
            )
        elif reason == "ambiguity":
            clarification_q = final_state.get("clarification_question", "")
            final_state["answer"] = f"질문을 더 명확히 해주세요.\n\n{clarification_q}"

    return {
        "answer": final_state.get("answer", "답변 생성 실패"),
        "sources": final_state.get("sources", []),
        "duration_ms": final_state.get("duration_ms", 0),
        "num_docs": final_state.get("num_docs", 0),
        "clarification_needed": final_state.get("clarification_needed", False),
        "web_search_used": final_state.get("web_search_used", False),
    }


__all__ = [
    "build_rag_graph",
    "answer_question",
    "load_unified_vector_db",
]
