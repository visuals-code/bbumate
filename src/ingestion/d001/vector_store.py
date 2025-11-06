"""Chroma 벡터 스토어 관리 모듈."""

import os
import shutil
import time
from pathlib import Path
from typing import List, Optional

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from .embeddings import get_embeddings

# 기본 설정
DEFAULT_CHROMA_DIR = "./chroma_storage"
COLLECTION_NAME = "housing_reports"
PREVIEW_LENGTH = 300


def store_in_chroma(chunks: List[Document]) -> Optional[Chroma]:
    """청크를 Upstage 임베딩을 사용하여 Chroma DB에 저장합니다.

    기존 DB는 삭제됩니다.

    Args:
        chunks: 저장할 문서 청크 리스트.

    Returns:
        생성된 Chroma 벡터스토어 객체. 실패 시 None 반환.
    """
    print("\n청크를 Upstage 임베딩을 사용하여 Chroma DB에 저장합니다...")

    try:
        # 임베딩 객체 생성
        embeddings = get_embeddings()
    except ValueError as e:
        print(f"Error: {e}")
        print("`.env` 파일이 올바른 위치에 있는지, 내용이 정확한지 확인하세요.")
        return None
    except Exception as e:  # pylint: disable=broad-except
        # 다양한 임베딩 초기화 오류를 처리하기 위한 포괄적 예외 처리
        print(f"Error: UpstageEmbeddings 초기화 중 오류: {e}")
        return None

    # 환경 변수에서 Chroma DB 저장 경로 로드 (기본값 설정)
    chroma_dir_str = os.getenv("CHROMA_DB_DIR", DEFAULT_CHROMA_DIR)
    chroma_persist_directory = Path(chroma_dir_str)
    print(
        f"Chroma DB 저장 경로: '{chroma_persist_directory}' "
        "(CHROMA_DB_DIR 환경 변수 사용)"
    )

    # 기존 DB 디렉토리가 있으면 삭제하여 중복 저장 방지
    if chroma_persist_directory.exists():
        print(
            f"Warning: 기존 Chroma DB '{chroma_persist_directory}' 발견. "
            "중복 방지를 위해 삭제합니다."
        )
        shutil.rmtree(chroma_persist_directory)
        print("기존 DB 삭제 완료.")

    # 디렉토리 생성 (Pathlib 사용)
    chroma_persist_directory.mkdir(parents=True, exist_ok=True)

    # 임베딩 시작 시간 측정
    print("\n임베딩을 시작합니다...")
    start_time = time.time()

    # Chroma 벡터스토어 생성 및 데이터 저장 (Pathlib 객체를 str로 변환하여 전달)
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(chroma_persist_directory),
        collection_name=COLLECTION_NAME,
    )

    # 임베딩 종료 시간 측정
    end_time = time.time()
    elapsed_time = end_time - start_time

    print(
        f"\n데이터가 Chroma DB에 성공적으로 저장되었습니다. "
        f"(경로: '{chroma_persist_directory}')"
    )
    print(f"[임베딩 완료] 총 소요 시간: {elapsed_time:.2f}초 ({elapsed_time/60:.2f}분)")

    return vector_db


def verify_storage() -> None:
    """Chroma DB를 불러와 저장된 문서 개수를 확인하고 검색을 테스트합니다.

    Returns:
        None. 콘솔에 검증 결과를 출력합니다.
    """
    print("\n--- Chroma DB 저장 확인 단계 ---")

    # 환경 변수에서 Chroma DB 저장 경로 로드
    chroma_dir_str = os.getenv("CHROMA_DB_DIR", DEFAULT_CHROMA_DIR)
    chroma_persist_directory = Path(chroma_dir_str)

    try:
        # 1. 임베딩 모델 재초기화 (검색 시에도 필요)
        embeddings = get_embeddings()
    except ValueError as e:
        print(f"Error: {e}")
        print("환경 변수가 설정되지 않아 확인을 건너뜁니다.")
        return
    except Exception as e:  # pylint: disable=broad-except
        # 임베딩 초기화 실패 시 검증 스킵
        print(f"Error: 임베딩 초기화 중 오류: {e}")
        return

    try:
        # 2. 저장된 Chroma DB 로드 (Pathlib 객체를 str로 변환하여 전달)
        persisted_db = Chroma(
            persist_directory=str(chroma_persist_directory),
            embedding_function=embeddings,
            collection_name=COLLECTION_NAME,
        )

        # 3. 데이터 개수 확인
        count = persisted_db._collection.count()  # pylint: disable=protected-access
        print(
            f"Chroma DB '{COLLECTION_NAME}'에 저장된 청크(문서)의 총 개수: {count}개"
        )

        if count == 0:
            print(
                "Warning: 저장된 청크가 0개입니다. "
                "PDF 파일이 올바르게 처리되었는지 확인하세요."
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
            print(f"\n테스트 검색어: '{test_query}'")
            print("Warning: 검색 결과가 없습니다. 데이터가 올바르게 저장되었는지 확인하세요.")
            return

        print(f"\n테스트 검색어: '{test_query}'")
        print(
            f"최상위 검색 결과 (소스): "
            f"{results[0].metadata.get('source', '소스 정보 없음')}"
        )
        print(" 최상위 검색 결과 (내용 일부): \n")
        # 검색 결과의 텍스트 내용을 PREVIEW_LENGTH만큼 출력
        print(results[0].page_content[:PREVIEW_LENGTH] + "...")
        print("---")
        print(
            "위와 같이 검색 결과가 출력되면, DB 저장이 성공적으로 완료된 것입니다."
        )

    except Exception as e:  # pylint: disable=broad-except
        # 검증 단계의 다양한 오류 처리
        print(f"Error: Chroma DB 로드 또는 검색 중 오류 발생: {e}")
