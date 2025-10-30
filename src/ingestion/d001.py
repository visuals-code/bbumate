"""PDF 문서를 로드하고 Upstage 임베딩을 사용하여 Chroma DB에 저장하는 데이터 수집 파이프라인."""

import os
import shutil
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 최신 Upstage LangChain 통합을 사용
from langchain_upstage import UpstageEmbeddings

# --- 전역 설정 ---
# PDF 파일들이 있는 디렉토리 경로 (상대 경로 사용)
PDF_DIRECTORY = Path("data/d001/housing")  # Pathlib 사용
# Chroma DB 기본 저장 경로 (.env의 CHROMA_DB_DIR과 일치)
DEFAULT_CHROMA_DIR = "./chroma_storage"
# Chroma 컬렉션 이름
COLLECTION_NAME = "housing_reports"
# 검색 결과 미리보기 길이
PREVIEW_LENGTH = 300


# --- 1. 모든 PDF 파일 로딩 ---
def load_pdfs(directory_path: Path):
    """지정된 디렉토리에서 모든 PDF 파일을 로드합니다."""
    print(f"\n디렉토리: '{directory_path}'에서 PDF 로드를 시작합니다...")

    try:
        # DirectoryLoader는 문자열 경로를 받으므로 str()로 변환
        loader = DirectoryLoader(
            path=str(directory_path),
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,
            show_progress=True,
        )

        documents = loader.load()
        print(f"✅ 총 {len(documents)}개의 문서 페이지/청크를 로드했습니다.")
        return documents
    except FileNotFoundError:
        print(
            f"❌ 파일 로드 중 FileNotFoundError 발생: PDF 파일을 {directory_path}에 넣어주세요."
        )
        return []


# --- 2. 텍스트 분할 (청킹) ---
def split_documents(documents):
    """로드된 문서를 검색에 적합한 크기의 청크로 분할합니다"""

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    chunks = text_splitter.split_documents(documents)
    print(f"✅ 총 {len(chunks)}개의 청크로 분할되었습니다.")
    return chunks


# --- 3. Upstage 임베딩 객체 생성 (간소화 및 오류 처리 개선) ---
def get_embeddings():
    """Upstage 임베딩 객체를 생성하여 반환합니다. 환경 변수에서 API 키를 자동 로드합니다."""

    upstage_model = os.getenv("UPSTAGE_EMBEDDING_MODEL")

    if not os.getenv("UPSTAGE_API_KEY") or not upstage_model:
        raise ValueError(
            "환경 변수 (UPSTAGE_API_KEY 또는 UPSTAGE_EMBEDDING_MODEL)가 설정되지 않았습니다. `.env` 파일을 확인하세요."
        )
    # UpstageEmbeddings는 UPSTAGE_API_KEY 환경 변수를 자동으로 찾아서 사용합니다.
    return UpstageEmbeddings(model=upstage_model)


# --- 4. 벡터 DB 저장 (환경 변수 경로 사용 및 중복 방지) ---
def store_in_chroma(chunks):
    """청크를 Upstage 임베딩을 사용하여 Chroma DB에 저장합니다. 기존 DB는 삭제됩니다."""
    print("\n청크를 Upstage 임베딩을 사용하여 Chroma DB에 저장합니다...")

    try:
        # 임베딩 객체 생성
        embeddings = get_embeddings()
    except ValueError as e:
        print(f"❌ {e}")
        print("`.env` 파일이 올바른 위치에 있는지, 내용이 정확한지 확인하세요.")
        return None
    except Exception as e:
        print(f"❌ UpstageEmbeddings 초기화 중 오류: {e}")
        return None

    # 환경 변수에서 Chroma DB 저장 경로 로드 (기본값 설정)
    chroma_dir_str = os.getenv("CHROMA_DB_DIR", DEFAULT_CHROMA_DIR)
    CHROMA_PERSIST_DIRECTORY = Path(chroma_dir_str)
    print(
        f"📁 Chroma DB 저장 경로: '{CHROMA_PERSIST_DIRECTORY}' (CHROMA_DB_DIR 환경 변수 사용)"
    )

    # 기존 DB 디렉토리가 있으면 삭제하여 중복 저장 방지
    if CHROMA_PERSIST_DIRECTORY.exists():
        print(
            f"⚠️  기존 Chroma DB '{CHROMA_PERSIST_DIRECTORY}' 발견. 중복 방지를 위해 삭제합니다."
        )
        shutil.rmtree(CHROMA_PERSIST_DIRECTORY)
        print(f"🗑️ 기존 DB 삭제 완료.")

    # 디렉토리 생성 (Pathlib 사용)
    CHROMA_PERSIST_DIRECTORY.mkdir(parents=True, exist_ok=True)

    # Chroma 벡터스토어 생성 및 데이터 저장 (Pathlib 객체를 str로 변환하여 전달)
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_PERSIST_DIRECTORY),
        collection_name=COLLECTION_NAME,
    )

    print(
        f"✅ 데이터가 Chroma DB에 성공적으로 저장되었습니다. (경로: '{CHROMA_PERSIST_DIRECTORY}')"
    )
    return vector_db


# --- 5. 저장 확인 및 테스트 (환경 변수 경로 사용) ---
def verify_storage():
    """Chroma DB를 불러와 저장된 문서 개수를 확인하고 검색을 테스트합니다."""
    print("\n--- 🔎 Chroma DB 저장 확인 단계 ---")

    # 환경 변수에서 Chroma DB 저장 경로 로드
    chroma_dir_str = os.getenv("CHROMA_DB_DIR", DEFAULT_CHROMA_DIR)
    CHROMA_PERSIST_DIRECTORY = Path(chroma_dir_str)

    try:
        # 1. 임베딩 모델 재초기화 (검색 시에도 필요)
        embeddings = get_embeddings()
    except ValueError as e:
        print(f"❌ {e}")
        print("환경 변수가 설정되지 않아 확인을 건너뜜니다.")
        return
    except Exception as e:
        print(f"❌ 임베딩 초기화 중 오류: {e}")
        return

    try:
        # 2. 저장된 Chroma DB 로드 (Pathlib 객체를 str로 변환하여 전달)
        persisted_db = Chroma(
            persist_directory=str(CHROMA_PERSIST_DIRECTORY),
            embedding_function=embeddings,
            collection_name=COLLECTION_NAME,
        )

        # 3. 데이터 개수 확인
        count = persisted_db._collection.count()
        print(
            f"✅ Chroma DB '{COLLECTION_NAME}'에 저장된 청크(문서)의 총 개수: {count}개"
        )

        if count == 0:
            print(
                "🚨 경고: 저장된 청크가 0개입니다. PDF 파일이 올바르게 처리되었는지 확인하세요."
            )
            return

        # 4. 간단한 검색 테스트
        test_query = input(
            "\n테스트 검색어를 입력하세요 (Enter 시 기본값 사용): "
        ).strip()

        # 입력이 없으면 기본값 사용
        if not test_query:
            test_query = "지역별 신혼부부 주택공급 현황을 알려줘"
            print(f"   → 기본 검색어 사용: '{test_query}'")

        # .as_retriever()를 사용하여 검색
        retriever = persisted_db.as_retriever(search_kwargs={"k": 1})
        results = retriever.invoke(test_query)

        # 검색 결과 검증
        if not results or len(results) == 0:
            print(f"\n🔍 테스트 검색어: '{test_query}'")
            print("⚠️ 검색 결과가 없습니다. 데이터가 올바르게 저장되었는지 확인하세요.")
            return

        print(f"\n🔍 테스트 검색어: '{test_query}'")
        print(
            f"📄 최상위 검색 결과 (소스): {results[0].metadata.get('source', '소스 정보 없음')}"
        )
        print(" 최상위 검색 결과 (내용 일부): \n")
        # 검색 결과의 텍스트 내용을 PREVIEW_LENGTH만큼 출력
        print(results[0].page_content[:PREVIEW_LENGTH] + "...")
        print("---")
        print(
            "💡 위와 같이 검색 결과가 출력되면, DB 저장이 성공적으로 완료된 것입니다."
        )

    except Exception as e:
        print(f"❌ Chroma DB 로드 또는 검색 중 오류 발생: {e}")


# --- 메인 실행 함수 ---
def main_pipeline():
    """PDF 로드부터 벡터 DB 저장까지 전체 파이프라인을 실행합니다."""
    # 1. `.env` 파일 로드
    load_dotenv()
    print("`.env` 파일의 환경 변수를 로드했습니다.")

    # 2. PDF 디렉토리 존재 확인
    if not PDF_DIRECTORY.exists():
        print(f"PDF 파일을 '{PDF_DIRECTORY}' 안에 넣은 후 다시 실행해 주세요.")
        return

    # 3. PDF 로드
    documents = load_pdfs(PDF_DIRECTORY)

    if not documents:
        print("로드할 문서(PDF 파일)가 디렉토리 내에 없거나 로드에 실패했습니다.")
        return

    # 4. 텍스트 분할
    chunks = split_documents(documents)

    # 5. 벡터 DB 저장
    vector_db = store_in_chroma(chunks)

    if vector_db:
        print("\n--- 파이프라인 실행 완료 ---")

        # 6. 저장 확인 함수 호출
        verify_storage()


if __name__ == "__main__":
    main_pipeline()
