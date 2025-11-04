"""로깅 설정 유틸리티.

애플리케이션 전역에서 사용할 로거를 설정하고 제공합니다.
"""

import logging
import sys
from typing import Optional

from src.config import settings


def setup_logging(level: Optional[str] = None) -> None:
    """로깅 시스템 초기화.

    Args:
        level: 로그 레벨 (DEBUG, INFO, WARNING, ERROR, CRITICAL).
               None인 경우 settings.LOG_LEVEL 사용
    """
    log_level = level or settings.LOG_LEVEL
    log_level_value = getattr(logging, log_level.upper(), logging.INFO)

    # 루트 로거 설정
    logging.basicConfig(
        level=log_level_value,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # 외부 라이브러리 로그 레벨 조정
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """모듈별 로거 반환.

    Args:
        name: 로거 이름 (일반적으로 __name__ 사용)

    Returns:
        logging.Logger: 설정된 로거 인스턴스
    """
    return logging.getLogger(name)
