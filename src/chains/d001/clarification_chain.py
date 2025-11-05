"""질문 명확화 체인 (Clarification Chain).

모호한 질문을 감지하고 사용자에게 명확화 질문을 제시하여
더 정확한 답변을 제공합니다.

워크플로우:
1. Question Analysis: 질문의 모호성 판단
2. Clarification Generation: 명확화 질문 생성
3. Question Refinement: 사용자 응답으로 질문 재구성
4. Retrieval: 명확해진 질문으로 문서 검색
"""

import json
from typing import Dict, List, Optional
from uuid import uuid4

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_upstage import ChatUpstage

from src.utils.d001.config import settings
from src.utils.d001.logger import get_logger

logger = get_logger(__name__)


class AmbiguityScore:
    """질문 모호성 평가 결과를 나타내는 클래스."""

    def __init__(
        self,
        is_ambiguous: bool,
        ambiguity_level: float,
        reasons: List[str],
        clarification_questions: List[str],
    ) -> None:
        """모호성 평가 결과를 초기화합니다.

        Args:
            is_ambiguous: 모호한 질문인지 여부.
            ambiguity_level: 모호성 점수 (0.0 ~ 1.0, 높을수록 모호함).
            reasons: 모호하다고 판단한 이유 리스트.
            clarification_questions: 명확화를 위한 질문 리스트.
        """
        self.is_ambiguous = is_ambiguous
        self.ambiguity_level = ambiguity_level
        self.reasons = reasons
        self.clarification_questions = clarification_questions


class ClarificationSession:
    """명확화 세션을 관리하는 클래스."""

    def __init__(
        self,
        session_id: str,
        original_question: str,
        clarification_questions: List[str],
    ) -> None:
        """명확화 세션을 초기화합니다.

        Args:
            session_id: 세션 고유 ID.
            original_question: 원본 질문.
            clarification_questions: 명확화 질문 리스트.
        """
        self.session_id = session_id
        self.original_question = original_question
        self.clarification_questions = clarification_questions
        self.clarification_answer: Optional[str] = None
        self.refined_question: Optional[str] = None


class ClarificationChain:
    """질문 명확화 체인.

    모호한 질문을 감지하고 사용자에게 명확화 질문을 제시합니다.
    """

    # 세션 저장소 (실제 운영에서는 Redis 등 사용 권장)
    _sessions: Dict[str, ClarificationSession] = {}

    def __init__(self, ambiguity_threshold: float = 0.6) -> None:
        """명확화 체인을 초기화합니다.

        Args:
            ambiguity_threshold: 모호성 임계값 (0.0 ~ 1.0).
                                 이 값 이상이면 명확화 필요.
        """
        self.ambiguity_threshold = ambiguity_threshold
        self.llm = ChatUpstage(
            model=settings.UPSTAGE_CHAT_MODEL,
            temperature=0.0,
        )
        self.analysis_prompt = self._create_analysis_prompt()
        self.refinement_prompt = self._create_refinement_prompt()

    def _create_analysis_prompt(self) -> ChatPromptTemplate:
        """질문 분석 프롬프트를 생성합니다.

        Returns:
            ChatPromptTemplate: 질문 분석용 프롬프트 템플릿.
        """
        system_template = """당신은 질문의 모호성을 평가하는 전문가입니다.

사용자의 질문을 분석하여 다음을 판단하세요:

1. **모호성 판단 기준**:
   - 핵심 주제나 명사가 명확하지 않음
   - 지시대명사만 사용 ("이거", "저거", "그것", "이것")
   - 맥락 없이 의문사만 사용 ("뭐야?", "어떻게?", "왜?")
   - 질문이 너무 짧거나 구체성이 부족함
   - 여러 주제가 섞여 있어 의도가 불분명함

2. **모호하지 않은 경우 (명확한 질문)**:
   - 구체적인 주제/대상이 명시됨 ("신혼부부 대출", "행복주택 신청")
   - 충분한 맥락과 정보가 포함됨
   - 질문 의도가 명확함

3. **출력 형식** (JSON):
{{
  "is_ambiguous": true/false,
  "ambiguity_level": 0.0~1.0,
  "reasons": ["이유1", "이유2"],
  "clarification_questions": ["질문1", "질문2", "질문3"]
}}

**중요**:
- is_ambiguous가 false면 clarification_questions는 빈 리스트 []
- is_ambiguous가 true면 3-5개의 구체적인 명확화 질문 제공
- clarification_questions는 사용자가 선택할 수 있는 옵션 형태로 제공
  (예: "주택 관련 질문인가요?", "대출 관련 질문인가요?", "복지 혜택 관련 질문인가요?")

사용자 질문:
{question}

JSON 형식으로만 답변하세요:"""

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_template),
            ]
        )

        return prompt

    def _create_refinement_prompt(self) -> ChatPromptTemplate:
        """질문 재구성 프롬프트를 생성합니다.

        Returns:
            ChatPromptTemplate: 질문 재구성용 프롬프트 템플릿.
        """
        system_template = """당신은 모호한 질문을 명확하게 재구성하는 전문가입니다.

원본 질문과 사용자의 명확화 응답을 결합하여 명확하고 구체적인 질문으로 재구성하세요.

**재구성 규칙**:
1. 원본 질문의 의도를 유지
2. 명확화 응답의 정보를 자연스럽게 통합
3. 구체적이고 명확한 질문 형태로 변환
4. 불필요한 설명 없이 질문만 출력

원본 질문: {original_question}
명확화 응답: {clarification_answer}

재구성된 질문만 출력하세요:"""

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_template),
            ]
        )

        return prompt

    def _rule_based_check(self, question: str) -> Optional[AmbiguityScore]:
        """규칙 기반 모호성을 체크합니다.

        간단한 규칙으로 빠르게 판단할 수 있는 경우 처리합니다.

        Args:
            question: 사용자 질문.

        Returns:
            Optional[AmbiguityScore]: 모호성 평가 결과 (규칙으로 판단 불가시 None).
        """
        question_stripped = question.strip()

        # 1. 너무 짧은 질문 (5자 미만)
        if len(question_stripped) < 5:
            return AmbiguityScore(
                is_ambiguous=True,
                ambiguity_level=0.9,
                reasons=["질문이 너무 짧습니다"],
                clarification_questions=[
                    "주택 관련 질문인가요?",
                    "대출 관련 질문인가요?",
                    "복지 혜택 관련 질문인가요?",
                    "기타 지원정책 관련 질문인가요?",
                ],
            )

        # 2. 지시대명사만 사용
        pronouns = ["이거", "저거", "그거", "이것", "저것", "그것"]
        question_words = ["어떻게", "어디", "언제", "왜", "누가", "뭐", "무엇"]

        if any(p in question_stripped for p in pronouns):
            # 다른 명사가 있는지 확인 (의문사는 제외)
            words = question_stripped.split()
            meaningful_words = [
                w
                for w in words
                if len(w) > 1 and w not in question_words and not w.endswith("?")
            ]
            # 지시대명사 제외한 의미있는 단어 개수
            has_other_words = (
                len([w for w in meaningful_words if w not in pronouns]) > 0
            )

            if not has_other_words:
                return AmbiguityScore(
                    is_ambiguous=True,
                    ambiguity_level=0.85,
                    reasons=["지시대명사만 사용되어 구체적인 대상이 불명확합니다"],
                    clarification_questions=[
                        "어떤 주제에 대해 궁금하신가요? (주택/대출/복지 등)",
                        "신혼부부 지원정책 관련 질문인가요?",
                        "특정 지역이나 기관의 정책인가요?",
                    ],
                )

        # 3. 의문사만 있는 질문
        question_words_only = ["뭐야?", "왜?", "어떻게?", "언제?", "어디?", "누가?"]
        if question_stripped in question_words_only:
            return AmbiguityScore(
                is_ambiguous=True,
                ambiguity_level=0.95,
                reasons=["의문사만 사용되어 질문의 주제가 불명확합니다"],
                clarification_questions=[
                    "무엇에 대해 궁금하신가요?",
                    "신혼부부 지원정책 관련 질문인가요?",
                    "구체적인 주제를 알려주세요",
                ],
            )

        # 규칙으로 판단 불가 - LLM 분석 필요
        return None

    def analyze_question(self, question: str) -> AmbiguityScore:
        """질문의 모호성을 분석합니다.

        Args:
            question: 사용자 질문.

        Returns:
            AmbiguityScore: 모호성 평가 결과.
        """
        logger.info("Analyzing question ambiguity: %s...", question[:50])

        # Step 1: 규칙 기반 빠른 체크
        rule_result = self._rule_based_check(question)
        if rule_result is not None:
            logger.info(
                "Rule-based check: ambiguous=%s, level=%.2f",
                rule_result.is_ambiguous,
                rule_result.ambiguity_level,
            )
            return rule_result

        # Step 2: LLM 기반 분석
        try:
            chain = self.analysis_prompt | self.llm | StrOutputParser()
            response = chain.invoke({"question": question})

            # LLM 응답에서 JSON 추출 (```json...``` 또는 {...} 형태)
            response_cleaned = response.strip()
            if "```json" in response_cleaned:
                response_cleaned = (
                    response_cleaned.split("```json")[1].split("```")[0].strip()
                )
            elif "```" in response_cleaned:
                response_cleaned = (
                    response_cleaned.split("```")[1].split("```")[0].strip()
                )

            result = json.loads(response_cleaned)

            ambiguity_score = AmbiguityScore(
                is_ambiguous=result.get("is_ambiguous", False),
                ambiguity_level=result.get("ambiguity_level", 0.0),
                reasons=result.get("reasons", []),
                clarification_questions=result.get("clarification_questions", []),
            )

            logger.info(
                "LLM analysis: ambiguous=%s, level=%.2f",
                ambiguity_score.is_ambiguous,
                ambiguity_score.ambiguity_level,
            )

            return ambiguity_score

        except Exception as e:  # pylint: disable=broad-except
            # 분석 실패 시에도 폴백 처리하여 작업 계속 진행
            logger.error("Question analysis failed: %s", e)

            # 폴백: 보수적으로 명확하다고 판단
            return AmbiguityScore(
                is_ambiguous=False,
                ambiguity_level=0.0,
                reasons=["분석 실패로 인해 보수적으로 명확하다고 판단"],
                clarification_questions=[],
            )

    def create_session(
        self, original_question: str, clarification_questions: List[str]
    ) -> str:
        """명확화 세션을 생성합니다.

        Args:
            original_question: 원본 질문.
            clarification_questions: 명확화 질문 리스트.

        Returns:
            str: 세션 ID.
        """
        session_id = str(uuid4())
        session = ClarificationSession(
            session_id=session_id,
            original_question=original_question,
            clarification_questions=clarification_questions,
        )
        self._sessions[session_id] = session

        logger.info("Created clarification session: %s", session_id)
        return session_id

    def refine_question(self, session_id: str, clarification_answer: str) -> str:
        """사용자 응답으로 질문을 재구성합니다.

        Args:
            session_id: 세션 ID.
            clarification_answer: 사용자의 명확화 응답.

        Returns:
            str: 재구성된 질문.

        Raises:
            ValueError: 세션을 찾을 수 없는 경우.
        """
        if session_id not in self._sessions:
            raise ValueError(f"Session not found: {session_id}")

        session = self._sessions[session_id]
        session.clarification_answer = clarification_answer

        logger.info("Refining question for session %s", session_id)

        try:
            chain = self.refinement_prompt | self.llm | StrOutputParser()
            refined_question = chain.invoke(
                {
                    "original_question": session.original_question,
                    "clarification_answer": clarification_answer,
                }
            )

            session.refined_question = refined_question.strip()
            logger.info("Refined question: %s", session.refined_question)

            return session.refined_question

        except Exception as e:  # pylint: disable=broad-except
            # 재구성 실패 시 폴백 처리하여 작업 계속 진행
            logger.error("Question refinement failed: %s", e)

            # 폴백: 원본 질문 + 명확화 응답 결합
            fallback_question = f"{session.original_question} ({clarification_answer})"
            session.refined_question = fallback_question
            return fallback_question

    def get_session(self, session_id: str) -> Optional[ClarificationSession]:
        """세션을 조회합니다.

        Args:
            session_id: 세션 ID.

        Returns:
            Optional[ClarificationSession]: 세션 객체 (없으면 None).
        """
        return self._sessions.get(session_id)

    def delete_session(self, session_id: str) -> None:
        """세션을 삭제합니다.

        Args:
            session_id: 세션 ID.
        """
        if session_id in self._sessions:
            del self._sessions[session_id]
            logger.info("Deleted session: %s", session_id)


def create_clarification_chain(ambiguity_threshold: float = 0.6) -> ClarificationChain:
    """Clarification Chain을 생성하는 헬퍼 함수입니다.

    Args:
        ambiguity_threshold: 모호성 임계값.

    Returns:
        ClarificationChain: 명확화 체인 인스턴스.
    """
    return ClarificationChain(ambiguity_threshold=ambiguity_threshold)
