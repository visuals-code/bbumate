"""d002 도메인 질문 검증 모듈."""
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


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
        "신혼부부",
        "전세",
        "자금",
        "대출",
        "주택",
        "구입",
        "매매",
        "세금",
        "세액",
        "공제",
        "혜택",
        "청약",
        "공급",
        "전세자금",
        "구입자금",
        "주택청약",
        "특별공급",
        "버팀목",
        "디딤돌",
        # 가족/복지 정책 관련 키워드 추가 (미혼모, 한부모 등)
        "미혼모",
        "한부모",
        "모자",
        "부자",
        "지원",
        "정책",
        "복지",
    ]

    # 도메인 키워드가 있으면 명확함
    if any(keyword in question for keyword in domain_keywords):
        return True

    # 기본 질문 패턴 체크
    question_patterns = [
        "조건",
        "한도",
        "금리",
        "혜택",
        "신청",
        "방법",
        "절차",
        "요건",
        "자격",
        "대상",
        "기간",
        "금액",
        "율",
    ]

    # 질문 패턴이 있으면 명확함
    if any(pattern in question for pattern in question_patterns):
        return True

    # 기본적으로는 모호함으로 판단
    return False


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
                "1. **도메인 관련성**: 질문이 신혼부부 지원정책(주거, 대출, 전세자금, 구매자금, 신혼부부 혜택 등) 또는 가족/복지 정책(미혼모, 한부모, 모자/부자 가정 지원 등)과 관련이 있으면 'DOMAIN_OK', "
                "날씨, 요리, 일반 뉴스 등 무관한 주제면 'DOMAIN_OUT'으로 답하세요.\n\n"
                "2. **질문 명확성**: 질문이 매우 모호하거나 거의 아무 정보도 없는 경우에만 'AMBIGUOUS', "
                "명확한 질문이면 'CLEAR'로 답하세요.\n\n"
                "예시 - 명확한 질문 (DOMAIN_OK, CLEAR): '신혼부부 전세자금대출 조건', '전세자금 대출 한도', '신혼부부 대출 금리', '미혼모가 받을 수 있는 지원', '한부모 지원 정책' 등\n"
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
            (
                "human",
                "모호한 질문: {question}\n\n간결한 명확화 질문(1-2개 핵심 정보만):",
            ),
        ]
    )
    clarify_chain = clarify_prompt | llm_model | StrOutputParser()

    try:
        clarified = clarify_chain.invoke({"question": question})
        return clarified.strip()
    except Exception:
        return "답변에 필요한 핵심 정보(지역, 주거형태 등)를 알려주세요."

