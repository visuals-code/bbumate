"""d002 도메인 로더 유틸리티."""
import os
import json
from pathlib import Path
from typing import Dict

from langchain_chroma import Chroma
from langchain_upstage import ChatUpstage, UpstageEmbeddings
from dotenv import load_dotenv

load_dotenv()

# 문서 출처 링크 매핑 (파일명 -> URL)
# data/d002/document_links.json 파일에서 로드하거나, 직접 딕셔너리로 정의 가능
_document_links: Dict[str, str] = {}


def load_document_links(domain: str = "d002") -> Dict[str, str]:
    """문서 출처 링크 매핑 로드.

    파일명 -> URL 매핑을 JSON 파일에서 로드하거나,
    없으면 빈 딕셔너리 반환.

    JSON 파일 형식 예시:
    {
        "tax_credit.html": "https://example.com/tax_credit",
        "newlywed_home_purchase_loan.html": "https://example.com/newlywed_loan"
    }
    """
    global _document_links

    if _document_links:
        return _document_links

    # JSON 파일 경로 확인
    links_file = Path(f"data/{domain}/document_links.json")

    if links_file.exists():
        try:
            with open(links_file, "r", encoding="utf-8") as f:
                _document_links = json.load(f)
            return _document_links
        except Exception:
            # JSON 로드 실패 시 빈 딕셔너리 반환
            return {}

    # JSON 파일이 없으면 빈 딕셔너리 반환
    return {}


def load_vector_db(domain: str = "d002") -> Chroma:
    """도메인별 Chroma VectorDB 로드 (Upstage 임베딩 일관화)."""
    persist_dir = f"data/{domain}/vector_store"

    api_key = os.getenv("UPSTAGE_API_KEY")
    if not api_key:
        raise ValueError("UPSTAGE_API_KEY 환경변수가 필요합니다")

    embedding_model = os.getenv("UPSTAGE_EMBEDDING_MODEL", "embedding-query")
    embeddings = UpstageEmbeddings(api_key=api_key, model=embedding_model)

    return Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings,
        collection_name=domain,
    )


def load_llm() -> ChatUpstage:
    """Upstage LLM 로드."""
    api_key = os.getenv("UPSTAGE_API_KEY")
    if not api_key:
        raise ValueError("UPSTAGE_API_KEY 환경변수가 필요합니다")

    model = os.getenv("UPSTAGE_CHAT_MODEL", "solar-1-mini-chat")

    return ChatUpstage(api_key=api_key, model=model)

