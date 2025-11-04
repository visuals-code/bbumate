"""Upstage 임베딩 모듈."""

import os

from langchain_upstage import UpstageEmbeddings


def get_embeddings() -> UpstageEmbeddings:
    """Upstage 임베딩 객체를 생성하여 반환합니다.

    환경 변수에서 API 키를 자동 로드합니다.

    Returns:
        초기화된 UpstageEmbeddings 객체.

    Raises:
        ValueError: 환경 변수가 설정되지 않은 경우.
    """
    upstage_model = os.getenv("UPSTAGE_EMBEDDING_MODEL")

    if not os.getenv("UPSTAGE_API_KEY") or not upstage_model:
        raise ValueError(
            "환경 변수 (UPSTAGE_API_KEY 또는 UPSTAGE_EMBEDDING_MODEL)가 설정되지 않았습니다. "
            "`.env` 파일을 확인하세요."
        )

    # UpstageEmbeddings는 UPSTAGE_API_KEY 환경 변수를 자동으로 찾아서 사용합니다.
    return UpstageEmbeddings(model=upstage_model)
