"""Ingestion 실행 스크립트"""
from src.ingestion.index import ingest

if __name__ == "__main__":
    # 통합 벡터 DB 생성 (모든 도메인)
    ingest("all")
    
    # 또는 특정 도메인만
    # ingest("d003")
    # ingest("d001")

