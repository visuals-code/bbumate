import os
import time
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_chroma import Chroma
from langchain_upstage import ChatUpstage, UpstageEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

# 모듈 레벨에서 .env 한 번만 로드
load_dotenv()

# Grade 결과 캐싱 (메모리 기반)
# 키: (질문 해시, 문서 해시) → 값: (is_relevant, doc)
_grade_cache: Dict[tuple[int, int], bool] = {}


def load_vector_db(domain: str = "d002") -> Chroma:
    """도메인별 Chroma VectorDB 로드 (Upstage 임베딩 일관화)."""
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
    api_key = os.getenv("UPSTAGE_API_KEY")
    if not api_key:
        raise ValueError("UPSTAGE_API_KEY 환경변수가 필요합니다")

    model = os.getenv("UPSTAGE_CHAT_MODEL", "solar-1-mini-chat")

    return ChatUpstage(api_key=api_key, model=model)


def is_question_clear(question: str) -> bool:
    """규칙 기반으로 질문이 명확한지 빠르게 판단 (LLM 호출 없이).
    
    명확한 질문의 특징:
    - 구체적인 키워드 포함 (전세자금, 대출, 세금, 주택 등)
    - 길이가 적절함 (5자 이상)
    - 단일 단어나 매우 짧은 질문 아님
    
    Returns:
        명확하면 True, 모호하면 False
    """
    question = question.strip()
    
    # 너무 짧으면 모호함
    if len(question) < 5:
        return False
    
    # 단일 단어만 있으면 모호함
    if len(question.split()) < 2:
        return False
    
    # 신혼부부 지원정책 관련 키워드 체크
    domain_keywords = [
        "신혼부부", "전세", "자금", "대출", "주택", "구입", "매매",
        "세금", "세액", "공제", "혜택", "청약", "공급", "전세자금",
        "구입자금", "주택청약", "특별공급", "버팀목", "디딤돌"
    ]
    
    # 도메인 키워드가 있으면 명확함
    if any(keyword in question for keyword in domain_keywords):
        return True
    
    # 기본 질문 패턴 체크
    question_patterns = [
        "조건", "한도", "금리", "혜택", "신청", "방법", "절차",
        "요건", "자격", "대상", "기간", "금액", "율"
    ]
    
    # 질문 패턴이 있으면 명확함
    if any(pattern in question for pattern in question_patterns):
        return True
    
    # 기본적으로는 모호함으로 판단
    return False


def _format_docs(docs: List[Any]) -> str:
    """문서들을 컨텍스트 형식으로 포맷팅 (출처 정보 제거하여 자연스러운 답변 유도)."""
    lines = []
    for i, d in enumerate(docs, 1):
        content = (d.page_content or "").strip()

        # 성능 개선: 문서 길이 제한 (요약형)
        if len(content) > 1500:
            content = content[:1500] + "..."

        lines.append(f"[문서 {i}]\n{content}")

    return "\n\n---\n\n".join(lines) if lines else "제공된 문서 없음"


def validate_question(question: str, llm_model) -> tuple[bool, str, str]:
    """질문 검증: 도메인 관련성 + 명확성 동시 체크.
    
    Returns:
        (is_valid, reason, clarification_question)
        - is_valid: 질문이 유효하면 True, 아니면 False
        - reason: 실패 이유 ('domain' 또는 'ambiguity')
        - clarification_question: 모호한 경우 명확화 질문 (도메인 외면 빈 문자열)
    """
    validation_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "당신은 질문 평가 전문가입니다. 다음 두 가지를 동시에 판단하세요:\n\n"
                "1. **도메인 관련성**: 질문이 신혼부부 지원정책(주거, 대출, 전세자금, 구매자금, 신혼부부 혜택 등)과 관련이 있으면 'DOMAIN_OK', "
                "날씨, 요리, 일반 뉴스 등 무관한 주제면 'DOMAIN_OUT'으로 답하세요.\n\n"
                "2. **질문 명확성**: 질문이 매우 모호하거나 거의 아무 정보도 없는 경우에만 'AMBIGUOUS', "
                "명확한 질문이면 'CLEAR'로 답하세요.\n\n"
                "예시 - 명확한 질문 (DOMAIN_OK, CLEAR): '신혼부부 전세자금대출 조건', '전세자금 대출 한도', '신혼부부 대출 금리' 등\n"
                "예시 - 모호한 질문 (DOMAIN_OK, AMBIGUOUS): '대출', '조건 알려줘', '도와줘' 등\n"
                "예시 - 도메인 외 (DOMAIN_OUT): '오늘 날씨는?', '파스타 만들기', '뉴스' 등\n\n"
                "일반적인 조건/정보 문의는 명확한 것으로 판단하세요. 개인 맞춤형 답변을 위해 지역/소득 정보가 필요한 경우에만 모호하다고 판단하세요.\n\n"
                "답변 형식: '도메인: [DOMAIN_OK/DOMAIN_OUT], 명확성: [CLEAR/AMBIGUOUS]'",
            ),
            ("human", "질문: {question}"),
        ]
    )
    validation_chain = validation_prompt | llm_model | StrOutputParser()
    
    try:
        result = validation_chain.invoke({"question": question}).upper()
        
        # 도메인 체크
        is_domain_ok = "DOMAIN_OK" in result
        if not is_domain_ok:
            return (False, "domain", "")
        
        # 명확성 체크
        is_ambiguous = "AMBIGUOUS" in result
        if is_ambiguous:
            clarification_question = clarify_question(question, llm_model)
            return (False, "ambiguity", clarification_question)
        
        # 통과
        return (True, "", "")
        
    except Exception:
        # 평가 실패 시 유효하다고 가정 (안전장치)
        return (True, "", "")


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
    """웹 검색 실행 (Tavily API 사용).
    
    Returns:
        검색 결과를 문자열로 반환. API 호출 실패 시 더미 응답 반환.
    """
    try:
        from tavily import TavilyClient
        
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            # API 키가 없으면 더미 응답 반환 (개발/테스트 환경)
            return f"[웹 검색 결과] '{query}'에 대한 최신 정보를 찾을 수 없습니다. TAVILY_API_KEY가 설정되지 않았습니다."
        
        client = TavilyClient(api_key=api_key)
        
        # Tavily API 호출
        # search_depth: "basic" (빠름) 또는 "advanced" (더 상세)
        response = client.search(
            query=query,
            search_depth="basic",
            max_results=3,  # 상위 3개 결과만 사용
        )
        
        # 검색 결과 포맷팅
        if response.get("results"):
            results_text = []
            for i, result in enumerate(response["results"], 1):
                title = result.get("title", "")
                content = result.get("content", "")
                url = result.get("url", "")
                
                results_text.append(
                    f"[결과 {i}] {title}\n{content}\n출처: {url}"
                )
            
            return "\n\n---\n\n".join(results_text)
        else:
            return f"[웹 검색 결과] '{query}'에 대한 검색 결과를 찾을 수 없습니다."
            
    except ImportError:
        # tavily-python 패키지가 설치되지 않은 경우
        return f"[웹 검색 결과] '{query}'에 대한 최신 정보를 찾을 수 없습니다. tavily-python 패키지가 설치되지 않았습니다."
    except Exception as e:
        # API 호출 실패 시 더미 응답 반환
        return f"[웹 검색 결과] '{query}'에 대한 최신 정보를 찾을 수 없습니다. 검색 중 오류 발생: {str(e)}"


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
    """단일 문서의 관련성을 평가 (캐싱 지원)."""
    # 캐시 키 생성
    question_hash = hash(question[:100])  # 질문 앞부분 해시
    doc_hash = hash(doc_content[:1500])  # 문서 앞부분 해시
    cache_key = (question_hash, doc_hash)
    
    # 캐시 확인
    if cache_key in _grade_cache:
        return _grade_cache[cache_key]
    
    # 캐시 미스 → LLM 평가
    try:
        grade = grade_chain.invoke({"question": question, "context": doc_content})
        is_relevant = "Y" in grade.upper()
    except Exception:
        # 평가 실패 시 관련 있다고 가정 (안전장치)
        is_relevant = True
    
    # 캐시 저장 (최대 1000개 항목으로 제한하여 메모리 관리)
    if len(_grade_cache) < 1000:
        _grade_cache[cache_key] = is_relevant
    
    return is_relevant


def grade_docs(question: str, docs: List[Any], llm_model) -> List[Any]:
    """retrieved 문서 중 관련도 높은 것만 필터링 (Grade 단계).
    
    병렬 처리로 성능 개선: k=3 문서를 동시에 평가하여 시간 절약.
    결과 품질은 동일하며, 순서는 원본 문서 순서 유지.
    """
    if not docs:
        return []

    # Grade 프롬프트 및 체인 구성
    grade_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "당신은 검색된 문서가 질문과 얼마나 관련 있는지를 판단하는 평가자입니다. "
                "문서 내용이 질문의 핵심 주제를 직접적으로 다루고 있으면 'Y', "
                "간접적으로만 관련되거나 주제가 다르면 'N'으로 답하세요. "
                "예시 - Y: '전세자금대출 조건' 질문에 '전세자금대출' 문서, '세금 혜택' 질문에 '세액공제' 문서 "
                "예시 - N: '재테크 방법' 질문에 '대출' 문서, '주거 문화' 질문에 '주택정책' 문서 "
                "문서 내용과 질문의 핵심 주제가 정확히 일치해야 합니다.",
            ),
            ("human", "질문: {question}\n\n문서 내용:\n{context}"),
        ]
    )
    grade_chain = grade_prompt | llm_model | StrOutputParser()

    # 병렬 처리로 문서 평가
    def evaluate_doc(doc_with_index: tuple[int, Any]) -> tuple[int, Any, bool]:
        """단일 문서 평가 함수 (병렬 처리용)."""
        index, doc = doc_with_index
        content = (doc.page_content or "").strip()[:1500]
        is_relevant = _grade_single_doc(question, content, grade_chain)
        return (index, doc, is_relevant)

    # ThreadPoolExecutor로 병렬 처리
    # k=3이므로 최대 3개의 스레드로 충분
    filtered = []
    with ThreadPoolExecutor(max_workers=min(len(docs), 3)) as executor:
        # 원본 인덱스와 함께 평가 작업 제출
        future_to_index = {
            executor.submit(evaluate_doc, (i, doc)): i
            for i, doc in enumerate(docs)
        }
        
        # 결과 수집 (원본 순서 유지)
        results = {}
        for future in as_completed(future_to_index):
            try:
                index, doc, is_relevant = future.result()
                results[index] = (doc, is_relevant)
            except Exception:
                # 평가 실패 시 관련 있다고 가정 (안전장치)
                index = future_to_index[future]
                results[index] = (docs[index], True)
        
        # 원본 순서대로 필터링
        for i in range(len(docs)):
            doc, is_relevant = results[i]
            if is_relevant:
                filtered.append(doc)

    return filtered


def build_rag_chain(domain: str = "d002", use_grade: bool = True):
    """RAG 체인 구성 요소 로드 (Grade 단계는 run_rag에서 직접 처리).
    
    Note: Grade 단계가 있어서 체인을 완전히 구성하기 어려움.
    대신 필요한 컴포넌트(vectordb, retriever, llm)만 반환하고,
    실제 RAG 체인은 run_rag에서 Grade 포함하여 구성함.
    """
    vectordb = load_vector_db(domain)
    llm = load_llm()

    # 성능 개선: k 값 줄이기 (5 → 3)
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})

    return retriever, llm, use_grade


def run_rag(
    query: str,
    domain: str = "d002",
    verbose: bool = False,
    use_grade: bool = True,
    use_validation: bool = True,
) -> Dict[str, Any]:
    """RAG 파이프라인 실행 (Question Validation → Retrieve → Grade → Generate).
    
    플로우:
    1. Question Validation: 도메인 관련성 + 명확성 동시 체크
       - 통과 못하면 → Re-ask/Clarification 또는 도메인 외 에러
    2. Retrieve: 문서 검색
    3. Grade: 관련성 높은 문서만 필터링
       - 문서 있으면(Yes) → Generate (RAG 체인 사용)
       - 문서 없으면(No) → Re-write query → Web search → Generate
    """
    start = time.perf_counter()
    retriever, llm, grade_enabled = build_rag_chain(domain, use_grade=use_grade)

    # Question Validation: 도메인 관련성 + 명확성 동시 체크 (선택적 스킵)
    if use_validation:
        # 명확한 질문이면 validation 스킵하여 성능 향상
        if is_question_clear(query):
            if verbose:
                print("[Validation 스킵: 명확한 질문]")
            # validation 통과로 간주
        else:
            # 모호한 질문이면 LLM으로 validation 실행
            is_valid, reason, clarification_question = validate_question(query, llm)
            
            if not is_valid:
                if reason == "domain":
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
                
                elif reason == "ambiguity":
                    if verbose:
                        print("[질문 모호성 감지]")
                        print(f"[명확화 요청]: {clarification_question}")
                    
                    return {
                        "answer": f"질문을 더 명확히 해주세요.\n\n{clarification_question}",
                        "sources": [],
                        "duration_ms": int((time.perf_counter() - start) * 1000),
                        "num_docs": 0,
                        "clarification_needed": True,
                        "web_search_used": False,
                    }

    # Retrieve: 문서 검색
    initial_docs = retriever.invoke(query)

    # Grade: 관련성 높은 문서만 필터링
    if grade_enabled and initial_docs:
        graded_docs = grade_docs(query, initial_docs, llm)
    else:
        graded_docs = initial_docs

    # Grade 결과로 바로 결정
    use_web_search = False
    if graded_docs:
        # Grade Yes: 관련 문서가 있음 → RAG 체인으로 Generate
        context = _format_docs(graded_docs)
        
        # RAG 체인 구성: Context + Question → Generate
        rag_prompt = ChatPromptTemplate.from_messages(
            [
                    (
                        "system",
                        """
당신은 신혼부부 지원정책 도메인 전문가입니다.

규칙:
1. 컨텍스트에 명확하게 나와있는 정보만 답변하세요. 컨텍스트에 없는 내용은 절대 추가하지 마세요.
2. 질문이 컨텍스트의 내용과 정확히 일치하지 않으면, "제공된 문서에는 해당 정보가 없습니다"라고 답하세요.
3. 답변은 자연스럽고 읽기 쉽게 작성하세요. "제공된 문서에는...", "○○ 문서에 따르면..." 같은 표현을 사용하지 마세요.
4. 문서명이나 파일명을 직접 언급하지 마세요.
5. 컨텍스트에 나와있는 정보를 연결하거나 확장하지 마세요. 정확히 명시된 내용만 답변하세요.

예시:
- 질문: "재테크 방법 알려줘" → 답변: "제공된 문서에는 재테크 방법에 대한 정보가 없습니다. 신혼부부 지원정책(대출, 세금 혜택 등) 관련 질문만 답변드릴 수 있습니다."
- 질문: "전세자금대출 조건 알려줘" → 답변: 컨텍스트에 나와있는 조건을 정확히 답변

컨텍스트:\n{context}
         """.strip(),
                    ),
                ("human", "질문: {question}"),
            ]
        )
        
        # RAG 체인: 질문 + 컨텍스트 → LLM → 답변
        rag_chain = rag_prompt | llm | StrOutputParser()
        answer = rag_chain.invoke({"question": query, "context": context})
        
        # 답변에서 "정보가 없습니다" 패턴 감지 → Web Search 경로로 전환
        no_info_keywords = ["정보가 없습니다", "해당 정보가 없습니다", "없습니다", "찾을 수 없습니다"]
        if any(keyword in answer for keyword in no_info_keywords):
            # 문서로 답변 불가능 → Web Search 경로로 전환
            use_web_search = True
            
            if verbose:
                print("[Generate 결과: 정보 없음 → Web Search 경로로 전환]")
            
            # Re-write query
            rewritten_query = rewrite_query(query, llm)
            
            if verbose:
                print(f"[쿼리 재작성]: {rewritten_query}")
            
            # Web Search
            web_results = web_search(rewritten_query)
            
            if verbose:
                print("[웹 검색 완료]")
            
            # Generate with Web Search results
            answer = generate_with_web_context(query, web_results, llm)
            sources = ["웹 검색"]
        else:
            # 답변 성공
            sources = list({d.metadata.get("source", "unknown") for d in graded_docs})
    else:
        # Grade No: 관련 문서가 없음 → Re-write query → Web Search → Generate
        use_web_search = True
        
        if verbose:
            print("[Grade 결과: 관련 문서 없음]")
        
        # Re-write query
        rewritten_query = rewrite_query(query, llm)
        
        if verbose:
            print(f"[쿼리 재작성]: {rewritten_query}")
        
        # Web Search
        web_results = web_search(rewritten_query)
        
        if verbose:
            print("[웹 검색 완료]")
        
        # Generate with Web Search results
        answer = generate_with_web_context(query, web_results, llm)
        sources = ["웹 검색"]

    # sources는 위에서 이미 설정됨 (웹 검색 경로 또는 문서 경로)
    duration_ms = int((time.perf_counter() - start) * 1000)

    if verbose:
        print("[질문]", query)
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
