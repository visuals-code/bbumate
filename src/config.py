"""중앙화된 설정 관리 모듈.

환경 변수를 로드하고 애플리케이션 전역 설정을 관리합니다.
"""

import os
import sys
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

# 환경 변수 로드 (프로젝트 루트의 .env 파일)
load_dotenv()


class Settings:
    """애플리케이션 설정 클래스"""

    # 환경 설정
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")

    # Upstage API 설정
    UPSTAGE_API_KEY: Optional[str] = os.getenv("UPSTAGE_API_KEY")
    UPSTAGE_EMBEDDING_MODEL: str = os.getenv(
        "UPSTAGE_EMBEDDING_MODEL", "solar-embedding-1-large"
    )
    UPSTAGE_CHAT_MODEL: str = os.getenv("UPSTAGE_CHAT_MODEL", "solar-1-mini-chat")

    # Vector DB 설정
    CHROMA_DB_DIR: str = os.getenv("CHROMA_DB_DIR", "./chroma_storage")
    CHROMA_COLLECTION_NAME: str = "housing_reports"

    # PDF 데이터 경로
    PDF_DIRECTORY: str = os.getenv("PDF_DIRECTORY", "data/d001/housing")

    # RAG 설정
    DEFAULT_RETRIEVAL_K: int = int(os.getenv("DEFAULT_RETRIEVAL_K", "3"))
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))

    # LLM 설정
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.0"))

    # 로깅 설정
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # API 설정
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    API_WORKERS: int = int(os.getenv("API_WORKERS", "4"))

    # CORS 설정
    CORS_ORIGINS: str = os.getenv("CORS_ORIGINS", "http://localhost:8080,http://127.0.0.1:8080")

    # Rate Limiting 설정
    RATE_LIMIT_PER_MINUTE: int = int(os.getenv("RATE_LIMIT_PER_MINUTE", "10"))

    # Adaptive RAG 설정
    USE_ADAPTIVE_RAG: bool = os.getenv("USE_ADAPTIVE_RAG", "true").lower() == "true"
    RELEVANCE_THRESHOLD: float = float(os.getenv("RELEVANCE_THRESHOLD", "0.4"))
    CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.85"))
    USE_WEB_SEARCH: bool = os.getenv("USE_WEB_SEARCH", "true").lower() == "true"
    USE_MOCK_WEB_SEARCH: bool = os.getenv("USE_MOCK_WEB_SEARCH", "false").lower() == "true"

    @classmethod
    def is_production(cls) -> bool:
        """프로덕션 환경인지 확인.

        Returns:
            bool: 프로덕션 환경이면 True, 아니면 False
        """
        return cls.ENVIRONMENT.lower() == "production"

    @classmethod
    def is_development(cls) -> bool:
        """개발 환경인지 확인.

        Returns:
            bool: 개발 환경이면 True, 아니면 False
        """
        return cls.ENVIRONMENT.lower() == "development"

    @classmethod
    def validate(cls) -> None:
        """필수 설정 값 검증.

        Raises:
            ValueError: 필수 설정이 누락된 경우
        """
        errors: List[str] = []

        if not cls.UPSTAGE_API_KEY:
            errors.append("UPSTAGE_API_KEY가 설정되지 않았습니다.")

        if not cls.UPSTAGE_EMBEDDING_MODEL:
            errors.append("UPSTAGE_EMBEDDING_MODEL이 설정되지 않았습니다.")

        if not cls.UPSTAGE_CHAT_MODEL:
            errors.append("UPSTAGE_CHAT_MODEL이 설정되지 않았습니다.")

        if cls.DEFAULT_RETRIEVAL_K < 1:
            errors.append("DEFAULT_RETRIEVAL_K는 1 이상이어야 합니다.")

        if cls.CHUNK_SIZE < 100:
            errors.append("CHUNK_SIZE는 100 이상이어야 합니다.")

        if cls.CHUNK_OVERLAP < 0:
            errors.append("CHUNK_OVERLAP은 0 이상이어야 합니다.")

        if errors:
            error_msg = "\n".join([f"  - {err}" for err in errors])
            raise ValueError(
                f"설정 검증 실패:\n{error_msg}\n\n.env 파일을 확인하고 필수 환경 변수를 설정하세요."
            )

    @classmethod
    def get_chroma_persist_directory(cls) -> Path:
        """Chroma DB 저장 경로를 Path 객체로 반환.

        Returns:
            Path: Chroma DB 저장 디렉토리 경로
        """
        return Path(cls.CHROMA_DB_DIR)

    @classmethod
    def get_pdf_directory(cls) -> Path:
        """PDF 디렉토리를 Path 객체로 반환.

        Returns:
            Path: PDF 파일이 저장된 디렉토리 경로
        """
        return Path(cls.PDF_DIRECTORY)

    @classmethod
    def get_cors_origins_list(cls) -> List[str]:
        """CORS 허용 오리진을 리스트로 반환.

        Returns:
            List[str]: CORS 허용 오리진 리스트
        """
        if cls.CORS_ORIGINS == "*":
            return ["*"]
        return [origin.strip() for origin in cls.CORS_ORIGINS.split(",")]


# 전역 설정 인스턴스
settings = Settings()

# 시작 시 설정 검증 (테스트 환경이 아닌 경우)
if "pytest" not in sys.modules and "unittest" not in sys.modules:
    try:
        settings.validate()
    except ValueError as e:
        print(f"❌ Configuration Error:\n{e}", file=sys.stderr)
        sys.exit(1)
