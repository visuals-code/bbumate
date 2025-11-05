"""d002 도메인 문서 평가(Grade) 모듈."""

from typing import List, Any, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Grade 결과 캐싱 (메모리 기반)
# 키: (질문 해시, 문서 해시) → 값: (is_relevant, doc)
_grade_cache: Dict[tuple[int, int], bool] = {}


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
            executor.submit(evaluate_doc, (i, doc)): i for i, doc in enumerate(docs)
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
