"""커스텀 예외 클래스 정의.

RAG 시스템 전반에서 사용할 예외 클래스들을 정의합니다.
"""


class RAGException(Exception):
    """RAG 시스템의 기본 예외 클래스.

    모든 RAG 관련 예외의 부모 클래스입니다.
    하위 클래스를 통해 세분화된 예외 처리를 제공합니다.
    """

    pass


class ConfigurationError(RAGException):
    """설정 관련 오류.

    환경 변수 누락, 잘못된 설정 값 등 설정 문제 발생 시 사용됩니다.

    Examples:
        - API 키 미설정
        - 잘못된 모델명
        - 필수 환경 변수 누락
    """

    pass


class EmbeddingError(RAGException):
    """임베딩 초기화 또는 생성 실패.

    임베딩 모델 로드 실패나 임베딩 벡터 생성 오류 시 사용됩니다.

    Examples:
        - Upstage 임베딩 모델 초기화 실패
        - 임베딩 API 호출 오류
        - 벡터 생성 타임아웃
    """

    pass


class DatabaseError(RAGException):
    """벡터 데이터베이스 작업 실패.

    ChromaDB 연결 실패, 컬렉션 접근 오류 등 DB 관련 문제 발생 시 사용됩니다.

    Examples:
        - ChromaDB 초기화 실패
        - 컬렉션 로드 오류
        - 데이터 저장/검색 실패
    """

    pass


class RetrievalError(RAGException):
    """문서 검색 실패.

    벡터 검색 실패, Retriever 오류 등 문서 검색 과정에서 발생하는 문제에 사용됩니다.

    Examples:
        - 벡터 유사도 검색 오류
        - Retriever 초기화 실패
        - 검색 결과 없음 (빈 결과)
    """

    pass


class GenerationError(RAGException):
    """LLM 답변 생성 실패.

    LLM 호출 실패, 응답 파싱 오류 등 답변 생성 과정에서 발생하는 문제에 사용됩니다.

    Examples:
        - Upstage API 호출 실패
        - LLM 응답 타임아웃
        - 응답 형식 파싱 오류
    """

    pass


class IngestionError(RAGException):
    """데이터 수집 및 처리 실패.

    PDF 로드, 청킹, 임베딩 저장 등 데이터 수집 과정의 오류에 사용됩니다.

    Examples:
        - PDF 파일 읽기 실패
        - 텍스트 청킹 오류
        - 벡터 DB 저장 실패
    """

    pass
