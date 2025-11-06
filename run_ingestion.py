"""Ingestion 실행 스크립트

배포 시 벡터 DB를 생성하기 위한 스크립트입니다.
- 서버 실행 전에 한 번만 실행합니다
- 통합 벡터 DB를 생성합니다 (모든 도메인: d001~d005)
- 기존 DB가 있으면 삭제 후 재생성합니다 (force_recreate=True)

주의:
- API 엔드포인트(/api/ingest)는 force_recreate=False가 기본값이므로 안전합니다
- 이 스크립트는 CLI에서 배포 시 한 번만 실행합니다
"""
from src.ingestion.index import ingest

if __name__ == "__main__":
    # 통합 벡터 DB 생성 (모든 도메인)
    # force_recreate=True: 기존 DB 삭제 후 재생성 (배포 시 사용)
    ingest("all", force_recreate=True)
    
    # 또는 특정 도메인만
    # ingest("d003", force_recreate=True)
    # ingest("d001", force_recreate=True)

