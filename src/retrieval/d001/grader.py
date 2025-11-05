"""문서 관련성 평가 모듈 (Document Grader).

검색된 문서가 사용자 질문과 얼마나 관련있는지 평가합니다.
"""

import asyncio
from typing import List, Literal, Tuple

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_upstage import ChatUpstage

from src.utils.d001.config import settings
from src.utils.d001.logger import get_logger

logger = get_logger(__name__)


class GradeScore(BaseModel):
    """문서 관련성 평가 결과."""

    score: Literal["relevant", "irrelevant"] = Field(
        description="문서가 질문과 관련있으면 'relevant', 아니면 'irrelevant'"
    )
    confidence: float = Field(description="평가 확신도 (0.0 ~ 1.0)", ge=0.0, le=1.0)


class DocumentGrader:
    """문서 관련성 평가기.

    LLM을 사용하여 검색된 문서가 사용자 질문에 답변하는데
    유용한지 평가합니다.
    """

    def __init__(self, relevance_threshold: float = 0.6) -> None:
        """DocumentGrader를 초기화합니다.

        Args:
            relevance_threshold: 관련성 임계값 (0.0 ~ 1.0).
                                 이 값 이상이면 relevant로 판단.
        """
        self.relevance_threshold = relevance_threshold
        self.llm = ChatUpstage(
            model=settings.UPSTAGE_CHAT_MODEL,
            temperature=0.0,  # 평가는 일관성 있게
        )
        self.grader_prompt = self._create_grader_prompt()

    def _create_grader_prompt(self) -> ChatPromptTemplate:
        """문서 평가용 프롬프트를 생성합니다.

        Returns:
            문서 평가용 프롬프트 템플릿.
        """
        system_template = """당신은 문서의 관련성을 평가하는 전문가입니다.

주어진 문서가 사용자의 질문에 답변하는데 도움이 될 수 있는 정보를 포함하고 있는지 평가하세요.

평가 기준:
1. 문서에 질문의 핵심 주제가 언급되어 있는가?
2. 문서의 내용이 질문에 답변하는데 유용한가?
3. 질문과 관련된 키워드나 개념이 포함되어 있는가?
4. 특정 지역/기관/연도가 질문에 명시된 경우, 문서도 유사한 맥락을 다루고 있는가?

예시:
- 질문: "오늘 날씨는?" / 문서: "신혼부부 주택 정책..." → irrelevant (완전히 다른 주제)
- 질문: "신혼부부 대출?" / 문서: "신혼부부 전세자금대출 조건..." → relevant
- 질문: "신혼부부 전세대출?" / 문서: "신혼부부 주거지원 정책..." → relevant (관련 정보 포함)
- 질문: "신혼부부 전세대출?" / 문서: "청년 대출 상품 안내..." → relevant (유사한 맥락)

**반드시 'relevant' 또는 'irrelevant' 중 하나만 답변하세요.**

문서 내용:
{document}

사용자 질문:
{question}

이 문서가 위 질문에 답변하는데 도움이 됩니까? relevant 또는 irrelevant로만 답하세요."""

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_template),
            ]
        )

        return prompt

    async def agrade_document(self, question: str, document: Document) -> GradeScore:
        """단일 문서를 비동기적으로 평가합니다.

        Args:
            question: 사용자 질문.
            document: 평가할 문서.

        Returns:
            평가 결과 (GradeScore).
        """
        try:
            # LLM 호출하여 실제 평가
            chain = self.grader_prompt | self.llm
            response = await chain.ainvoke(
                {
                    "question": question,
                    "document": document.page_content[:500],  # 토큰 절약을 위해 500자만
                }
            )

            response_text = response.content.lower()

            # 응답 파싱
            if "relevant" in response_text and "irrelevant" not in response_text[:20]:
                score = "relevant"
                if (
                    "매우" in response_text
                    or "확실" in response_text
                    or "명확" in response_text
                ):
                    confidence = 0.9
                elif "보통" in response_text or "어느정도" in response_text:
                    confidence = 0.7
                else:
                    confidence = 0.8
            elif (
                "irrelevant" in response_text
                or "관련없" in response_text
                or "관련 없" in response_text
            ):
                score = "irrelevant"
                confidence = 0.8
            else:
                # Fallback for unclear responses
                content_lower = document.page_content.lower()
                question_lower = question.lower()
                question_keywords = [w for w in question_lower.split() if len(w) > 1]
                matches = sum(1 for kw in question_keywords if kw in content_lower)
                match_ratio = (
                    matches / len(question_keywords) if question_keywords else 0
                )
                score = "relevant" if match_ratio >= 0.5 else "irrelevant"
                confidence = 0.6 if score == "relevant" else 0.7

            logger.debug(
                "Document grading: %s (confidence: %.2f) for question: %s...",
                score,
                confidence,
                question[:30],
            )
            return GradeScore(score=score, confidence=confidence)

        except Exception as e:
            logger.warning(
                "Async document grading failed: %s, using conservative evaluation", e
            )
            content_lower = document.page_content.lower()
            question_lower = question.lower()
            question_keywords = [w for w in question_lower.split() if len(w) > 1]
            matches = sum(1 for kw in question_keywords if kw in content_lower)
            match_ratio = matches / len(question_keywords) if question_keywords else 0
            return (
                GradeScore(score="relevant", confidence=0.6)
                if match_ratio >= 0.6
                else GradeScore(score="irrelevant", confidence=0.7)
            )

    async def agrade_documents(
        self, question: str, documents: List[Document]
    ) -> List[Tuple[Document, GradeScore]]:
        """여러 문서를 비동기적으로 평가합니다.

        Args:
            question: 사용자 질문.
            documents: 평가할 문서 리스트.

        Returns:
            (문서, 평가결과) 튜플 리스트.
        """
        tasks = [self.agrade_document(question, doc) for doc in documents]
        grades = await asyncio.gather(*tasks)
        return list(zip(documents, grades))

    async def afilter_relevant_documents_with_scores(
        self, question: str, documents: List[Document]
    ) -> Tuple[List[Tuple[Document, float]], float]:
        """관련있는 문서만 비동기적으로 필터링합니다 (점수 포함).

        Args:
            question: 사용자 질문.
            documents: 평가할 문서 리스트.

        Returns:
            ((문서, 확신도) 튜플 리스트, 평균 확신도) 튜플.
        """
        graded_docs = await self.agrade_documents(question, documents)

        relevant_docs_with_scores = []
        confidences = []

        for doc, grade in graded_docs:
            if (
                grade.score == "relevant"
                and grade.confidence >= self.relevance_threshold
            ):
                relevant_docs_with_scores.append((doc, grade.confidence))
                confidences.append(grade.confidence)

        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        logger.info(
            "Filtered %d/%d relevant documents with scores (avg confidence: %.2f)",
            len(relevant_docs_with_scores),
            len(documents),
            avg_confidence,
        )

        return relevant_docs_with_scores, avg_confidence

    def grade_document(self, question: str, document: Document) -> GradeScore:
        """단일 문서를 평가합니다.

        Args:
            question: 사용자 질문.
            document: 평가할 문서.

        Returns:
            평가 결과 (GradeScore).
        """
        try:
            # LLM 호출하여 실제 평가
            chain = self.grader_prompt | self.llm
            response = chain.invoke(
                {
                    "question": question,
                    "document": document.page_content[:500],  # 토큰 절약을 위해 500자만
                }
            )

            response_text = response.content.lower()

            # 응답 파싱
            # LLM이 "relevant", "irrelevant" 또는 유사한 답변을 할 것임
            if "relevant" in response_text and "irrelevant" not in response_text[:20]:
                score = "relevant"
                # 확신도 추출 시도
                if (
                    "매우" in response_text
                    or "확실" in response_text
                    or "명확" in response_text
                ):
                    confidence = 0.9
                elif "보통" in response_text or "어느정도" in response_text:
                    confidence = 0.7
                else:
                    confidence = 0.8
            elif (
                "irrelevant" in response_text
                or "관련없" in response_text
                or "관련 없" in response_text
            ):
                score = "irrelevant"
                confidence = 0.8
            else:
                # 불명확한 경우 내용 기반 간단 평가
                content_lower = document.page_content.lower()
                question_lower = question.lower()

                # 핵심 키워드 추출 (조사 제거)
                question_keywords = [w for w in question_lower.split() if len(w) > 1]

                # 키워드 포함 여부 확인
                matches = sum(1 for kw in question_keywords if kw in content_lower)
                match_ratio = (
                    matches / len(question_keywords) if question_keywords else 0
                )

                if match_ratio >= 0.5:
                    score = "relevant"
                    confidence = 0.6
                else:
                    score = "irrelevant"
                    confidence = 0.7

            logger.debug(
                "Document grading: %s (confidence: %.2f) for question: %s...",
                score,
                confidence,
                question[:30],
            )

            return GradeScore(score=score, confidence=confidence)

        except Exception as e:  # pylint: disable=broad-except
            # 폴백 전략: LLM 실패 시에도 키워드 기반 평가로 처리
            logger.warning(
                "Document grading failed: %s, using conservative evaluation", e
            )

            # 폴백: 보수적 평가 (간단한 키워드 매칭)
            content_lower = document.page_content.lower()
            question_lower = question.lower()

            # 핵심 키워드만 추출 (2글자 이상)
            question_keywords = [w for w in question_lower.split() if len(w) > 1]

            matches = sum(1 for kw in question_keywords if kw in content_lower)
            match_ratio = matches / len(question_keywords) if question_keywords else 0

            # 더 엄격한 기준
            if match_ratio >= 0.6:  # 60% 이상 매칭
                return GradeScore(score="relevant", confidence=0.6)
            else:
                return GradeScore(score="irrelevant", confidence=0.7)

    def grade_documents(
        self, question: str, documents: List[Document]
    ) -> List[Tuple[Document, GradeScore]]:
        """여러 문서를 평가합니다.

        Args:
            question: 사용자 질문.
            documents: 평가할 문서 리스트.

        Returns:
            (문서, 평가결과) 튜플 리스트.
        """
        results = []
        for doc in documents:
            grade = self.grade_document(question, doc)
            results.append((doc, grade))

        return results

    def filter_relevant_documents(
        self, question: str, documents: List[Document]
    ) -> Tuple[List[Document], float]:
        """관련있는 문서만 필터링합니다.

        Args:
            question: 사용자 질문.
            documents: 평가할 문서 리스트.

        Returns:
            (관련 문서 리스트, 평균 확신도) 튜플.
        """
        graded_docs = self.grade_documents(question, documents)

        relevant_docs = []
        confidences = []

        for doc, grade in graded_docs:
            if (
                grade.score == "relevant"
                and grade.confidence >= self.relevance_threshold
            ):
                relevant_docs.append(doc)
                confidences.append(grade.confidence)

        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        logger.info(
            "Filtered %d/%d relevant documents (avg confidence: %.2f)",
            len(relevant_docs),
            len(documents),
            avg_confidence,
        )

        return relevant_docs, avg_confidence

    def filter_relevant_documents_with_scores(
        self, question: str, documents: List[Document]
    ) -> Tuple[List[Tuple[Document, float]], float]:
        """관련있는 문서만 필터링합니다 (점수 포함).

        Args:
            question: 사용자 질문.
            documents: 평가할 문서 리스트.

        Returns:
            ((문서, 확신도) 튜플 리스트, 평균 확신도) 튜플.
        """
        graded_docs = self.grade_documents(question, documents)

        relevant_docs_with_scores = []
        confidences = []

        for doc, grade in graded_docs:
            if (
                grade.score == "relevant"
                and grade.confidence >= self.relevance_threshold
            ):
                relevant_docs_with_scores.append((doc, grade.confidence))
                confidences.append(grade.confidence)

        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        logger.info(
            "Filtered %d/%d relevant documents with scores (avg confidence: %.2f)",
            len(relevant_docs_with_scores),
            len(documents),
            avg_confidence,
        )

        return relevant_docs_with_scores, avg_confidence


def create_grader(relevance_threshold: float = 0.6) -> DocumentGrader:
    """Document Grader 생성 헬퍼 함수.

    Args:
        relevance_threshold: 관련성 임계값.

    Returns:
        문서 평가기 인스턴스.
    """
    return DocumentGrader(relevance_threshold=relevance_threshold)
