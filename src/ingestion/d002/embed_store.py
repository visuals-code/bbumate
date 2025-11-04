import os
import argparse
import time
from pathlib import Path
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import logging
from tqdm import tqdm

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_upstage import UpstageEmbeddings
from langchain_community.vectorstores import Chroma


# 로깅 설정
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# .env 로드
load_dotenv()

# 청킹 파라미터 상수
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 100


def extract_text_from_html(html_content: str) -> str:
    """HTML에서 순수 텍스트만 추출.

    - 스크립트/스타일 제거 후 화면에 보이는 텍스트만 뽑아 임베딩 품질을 높인다.
    - 태그를 보존하지 않음: 벡터화 대상은 의미 텍스트 중심.
    """
    try:
        soup = BeautifulSoup(html_content, "html.parser")
        # script, style 태그 제거
        for script in soup(["script", "style"]):
            script.decompose()

        text = soup.get_text(separator="\n", strip=True)
        return text

    except Exception as e:
        logger.warning(f"HTML 파싱 실패, raw text 사용: {e}")
        return html_content


def load_html_documents(input_dir: Path) -> list[Document]:
    """HTML 파일들을 Document 객체로 로드.

    - 파일 단위 실패는 전체 배치를 중단하지 않고 로그만 남기고 계속 진행한다.
    - 메타데이터에 원본 경로/길이를 저장해 추적성과 디버깅을 돕는다.
    """

    if not input_dir.exists():
        raise FileNotFoundError(f"입력 디렉토리가 없습니다: {input_dir}")

    html_files = list(input_dir.glob("*.html"))
    if not html_files:
        raise ValueError(f"HTML 파일이 없습니다: {input_dir}")

    documents = []
    failed_files = []

    logger.info(f"{len(html_files)}개 HTML 파일 발견")
    for file_path in tqdm(html_files, desc="HTML 로딩"):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                html_content = f.read()

            # HTML 태그 제거하고 순수 텍스트만 추출
            text = extract_text_from_html(html_content)

            if not text.strip():
                logger.warning(f"빈 문서 건너뜀: {file_path.name}")
                continue

            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": file_path.name,
                        "file_path": str(file_path),
                        "total_chars": len(text),
                    },
                )
            )

        except Exception as e:
            logger.error(f"{file_path.name} 로드 실패: {e}")
            failed_files.append(file_path.name)

    if failed_files:
        logger.warning(
            f"실패한 파일 ({len(failed_files)}개): {', '.join(failed_files[:5])}"
            + (f" 외 {len(failed_files)-5}개" if len(failed_files) > 5 else "")
        )

    return documents


def embed_from_html(
    input_dir: Path,
    persist_dir: Path,
    collection_name: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
):
    """HTML 파일들을 임베딩하여 VectorDB에 저장.

    - 텍스트 정제 → 분할 → 임베딩 → VectorDB 저장 순서로 처리한다.
    - 진행률/요약 로깅을 통해 대량 처리 시 가시성을 확보한다.
    """

    # HTML 문서 로드
    logger.info(f"HTML 파일 로드 중: {input_dir}")
    documents = load_html_documents(input_dir)
    logger.info(f"{len(documents)}개 HTML 파일 로드 완료")

    if not documents:
        raise ValueError("처리할 문서가 없습니다.")

    # 텍스트 분할
    logger.info("텍스트 분할 중...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )
    splits = splitter.split_documents(documents)

    # 청크 인덱스 메타데이터 추가
    # - 검색 결과가 원문에서 어느 위치인지 추적하기 위한 최소 정보
    for i, split in enumerate(splits):
        split.metadata["chunk_index"] = i

    logger.info(
        f"총 {len(splits)}개 청크로 분할 완료 (chunk={chunk_size}, overlap={chunk_overlap})"
    )

    # 예상 비용 안내 (선택사항)
    total_chars = sum(len(doc.page_content) for doc in splits)
    logger.info(f"총 {total_chars:,}자 임베딩 예정 (약 {len(splits)}회 API 호출)")

    # Upstage 임베딩 초기화
    api_key = os.getenv("UPSTAGE_API_KEY")
    if not api_key:
        raise ValueError("환경변수 UPSTAGE_API_KEY가 없습니다 (.env 확인).")

    try:
        # 임베딩 모델은 환경변수로 교체 가능. 기본값은 Upstage 표준 임베딩 모델.
        embedding_model = os.getenv("UPSTAGE_EMBEDDING_MODEL", "embedding-query")
        embeddings = UpstageEmbeddings(api_key=api_key, model=embedding_model)
        logger.info(f"임베딩 모델: {embedding_model}")

    except Exception as e:
        raise ValueError(f"Upstage 임베딩 초기화 실패: {e}")

    # Chroma VectorDB 저장
    persist_dir.mkdir(parents=True, exist_ok=True)

    try:
        logger.info("임베딩 및 VectorDB 생성 중... (시간이 걸릴 수 있습니다)")
        embedding_start_time = time.perf_counter()

        # 배치 처리로 메모리 효율성 개선
        # - 대용량(수천 청크)에서 한 번에 from_documents를 호출하면 메모리/시간 급증
        # - 1회 생성 후 add_documents로 증분 추가하여 안정성 확보
        batch_size = 100
        if len(splits) > batch_size:
            logger.info(f"{batch_size}개씩 배치 처리")
            vectordb = None
            for i in tqdm(range(0, len(splits), batch_size), desc="임베딩"):
                batch = splits[i : i + batch_size]
                if vectordb is None:
                    vectordb = Chroma.from_documents(
                        documents=batch,
                        embedding=embeddings,
                        persist_directory=str(persist_dir),
                        collection_name=collection_name,
                    )
                else:
                    vectordb.add_documents(batch)
        else:
            vectordb = Chroma.from_documents(
                documents=splits,
                embedding=embeddings,
                persist_directory=str(persist_dir),
                collection_name=collection_name,
            )

        # from_documents는 생성 시 저장되며, add_documents는 자동 반영됨
        embedding_elapsed = time.perf_counter() - embedding_start_time
        print(f"[임베딩 완료] 총 소요 시간: {embedding_elapsed:.2f}초 ({embedding_elapsed/60:.2f}분)")

    except Exception as e:
        raise RuntimeError(f"VectorDB 생성 실패: {e}")

    # 결과 확인
    try:
        # 실제 문서의 단어로 검색 테스트
        # - "test" 같은 무의미 쿼리 대신, 첫 청크의 실제 토큰으로 저장 검증
        first_words = splits[0].page_content[:20].split()[0] if splits else "test"
        test_results = vectordb.similarity_search(first_words, k=1)
        doc_count = len(splits)

        logger.info("=" * 60)
        logger.info("임베딩 완료 및 저장 완료!")
        logger.info(f"VectorDB 위치: {persist_dir}")
        logger.info(f"컬렉션 이름: {collection_name}")
        logger.info(f"저장된 청크 수: {doc_count}")
        logger.info(f"원본 문서 수: {len(documents)}")

        if doc_count:
            logger.info(f"평균 청크 크기: {total_chars // doc_count:,}자")

        if test_results:
            logger.info(f"샘플 메타데이터:")
            logger.info(f"   • source: {test_results[0].metadata.get('source')}")
            logger.info(
                f"   • chunk_index: {test_results[0].metadata.get('chunk_index')}"
            )
            logger.info(
                f"   • total_chars: {test_results[0].metadata.get('total_chars')}"
            )
            logger.info(f"검색 테스트 성공 (쿼리: '{first_words}')")
        logger.info("=" * 60)

    except Exception as e:
        logger.warning(f"결과 확인 단계에서 오류 (저장은 완료됨): {e}")


def main():
    parser = argparse.ArgumentParser(
        description="HTML 파일들을 임베딩하여 Chroma VectorDB에 저장"
    )
    parser.add_argument("--domain", required=True, help="도메인 이름 (예: d002)")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=f"청크 크기 (기본값: {DEFAULT_CHUNK_SIZE})",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=DEFAULT_CHUNK_OVERLAP,
        help=f"청크 오버랩 (기본값: {DEFAULT_CHUNK_OVERLAP})",
    )
    args = parser.parse_args()

    base_dir = Path("data") / args.domain
    input_dir = base_dir / "htmls"
    persist_dir = base_dir / "vector_store"
    collection_name = args.domain

    try:
        embed_from_html(
            input_dir=input_dir,
            persist_dir=persist_dir,
            collection_name=collection_name,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
    except Exception as e:
        logger.error(f"처리 실패: {e}")
        exit(1)


if __name__ == "__main__":
    main()


# 실행
# 기본 설정으로 실행
# python src/ingestion/d002/embed_store.py --domain d002

# 커스텀 청크 크기로 실행
# python src/ingestion/d002/embed_store.py --domain d002 --chunk-size 1500 --chunk-overlap 150