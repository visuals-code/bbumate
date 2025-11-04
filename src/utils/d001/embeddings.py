"""임베딩 유틸리티.

Upstage 임베딩 모델을 초기화하고 제공하는 공통 유틸리티입니다.
"""

from langchain_upstage import UpstageEmbeddings

from src.config import settings
from src.exceptions import ConfigurationError, EmbeddingError
from src.utils.d001.logger import get_logger

logger = get_logger(__name__)


def get_embeddings() -> UpstageEmbeddings:
    """Upstage 임베딩 객체를 생성하여 반환.

    환경 변수에서 API 키와 모델 이름을 자동으로 로드합니다.

    Returns:
        UpstageEmbeddings: 초기화된 임베딩 모델

    Raises:
        ConfigurationError: API 키 또는 모델 이름이 설정되지 않은 경우
        EmbeddingError: 임베딩 모델 초기화 실패
    """
    try:
        # 설정 검증
        if not settings.UPSTAGE_API_KEY:
            raise ConfigurationError(
                "UPSTAGE_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요."
            )

        if not settings.UPSTAGE_EMBEDDING_MODEL:
            raise ConfigurationError(
                "UPSTAGE_EMBEDDING_MODEL이 설정되지 않았습니다. .env 파일을 확인하세요."
            )

        logger.info(f"Upstage 임베딩 모델 '{settings.UPSTAGE_EMBEDDING_MODEL}' 초기화")

        # UpstageEmbeddings는 UPSTAGE_API_KEY 환경 변수를 자동으로 사용
        return UpstageEmbeddings(model=settings.UPSTAGE_EMBEDDING_MODEL)

    except ConfigurationError:
        # 설정 오류는 그대로 전파
        raise
    except Exception as e:
        logger.error(f"임베딩 모델 초기화 실패: {e}")
        raise EmbeddingError(f"임베딩 모델 초기화 중 오류 발생: {e}") from e
