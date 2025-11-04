import os
import argparse
import logging
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_upstage import UpstageEmbeddings


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)
load_dotenv()


def load_vector_db(domain: str) -> Chroma:
    """도메인별 저장된 Chroma 벡터DB 로드"""
    base_dir = Path("data") / domain / "vector_store"

    if not base_dir.exists():
        raise FileNotFoundError(f"벡터 스토어가 없습니다: {base_dir}")

    api_key = os.getenv("UPSTAGE_API_KEY")
    if not api_key:
        raise ValueError("UPSTAGE_API_KEY 환경변수가 필요합니다")

    embedding_model = os.getenv("UPSTAGE_EMBEDDING_MODEL", "embedding-query")
    embeddings = UpstageEmbeddings(api_key=api_key, model=embedding_model)

    vectordb = Chroma(
        persist_directory=str(base_dir),
        embedding_function=embeddings,
        collection_name=domain,
    )

    logger.info(f"✅ VectorDB 로드: {base_dir}")
    return vectordb


def search_documents(
    vectordb: Chroma, query: str, k: int = 5, with_score: bool = False
):
    """벡터DB에서 문서 검색"""

    if with_score:
        results = vectordb.similarity_search_with_score(query, k=k)
        if not results:
            logger.warning("검색 결과 없음")
            return []

        logger.info(f"📊 {len(results)}개 검색됨\n")
        for i, (doc, score) in enumerate(results, 1):
            print(f"[{i}] {doc.metadata.get('source', 'unknown')} (score: {score:.3f})")
            print(f"    {doc.page_content[:200]}...\n")
        return results

    else:
        results = vectordb.similarity_search(query, k=k)
        if not results:
            logger.warning("검색 결과 없음")
            return []

        logger.info(f"📊 {len(results)}개 검색됨\n")
        for i, doc in enumerate(results, 1):
            print(f"[{i}] {doc.metadata.get('source', 'unknown')}")
            print(f"    {doc.page_content[:200]}...\n")
        return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", required=True, help="도메인 (예: d002)")
    parser.add_argument("--query", required=True, help="검색 쿼리")
    parser.add_argument("--k", type=int, default=5, help="결과 개수")
    parser.add_argument("--score", action="store_true", help="유사도 점수 표시")
    args = parser.parse_args()

    try:
        vectordb = load_vector_db(args.domain)
        logger.info(f"🔍 '{args.query}' 검색 중...\n")
        search_documents(vectordb, args.query, k=args.k, with_score=args.score)
    except Exception as e:
        logger.error(f"오류: {e}")
        exit(1)


if __name__ == "__main__":
    main()


# # 기본 검색
# python src/retrieval/retrieve_d002.py --domain d002 --query "신혼부부 전세자금대출 조건"

# # 점수 포함
# python src/retrieval/retrieve_d002.py --domain d002 --query "신혼부부 전세자금대출 조건" --score

# # 결과 개수 조정
# python src/retrieval/retrieve_d002.py --domain d002 --query "신혼부부 전세자금대출 조건" --k 10
