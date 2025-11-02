"""임베딩 모델 테스트"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from langchain_upstage import UpstageEmbeddings

# 프로젝트 루트를 Python path에 추가
project_root = Path(__name__).parent.parent
sys.path.insert(0, str(project_root))


def test_embedding():
    """Upstage 임베딩 모델 테스트"""

    # 환경 변수 로드
    env_path = project_root / ".env"
    if not env_path.exists():
        env_path = project_root / ".env"
    load_dotenv(dotenv_path=env_path)

    # API Key 확인
    api_key = os.getenv("UPSTAGE_API_KEY")
    if not api_key:
        raise ValueError("UPSTAGE_API_KEY가 설정되지 않았습니다.")

    print("=" * 60)
    print("Upstage 임베딩 모델 테스트")
    print("=" * 60)

    # UpstageEmbeddings 인스턴스 생성
    print("\n[1단계] 임베딩 모델 초기화")
    embeddings = UpstageEmbeddings(
        api_key=api_key, model="solar-embedding-1-large-passage"
    )
    print("✅ 임베딩 모델 초기화 완료")

    # 테스트용 텍스트
    test_texts = [
        "벡터 데이터베이스에 넣을 텍스트 청크입니다.",
        "Passage 임베딩 모델 테스트.",
        "특별공급 신청 자격에 대한 내용입니다.",
    ]

    print(f"\n[2단계] 테스트 텍스트 임베딩 ({len(test_texts)}개)")
    for i, text in enumerate(test_texts, 1):
        print(f"  {i}. {text}")

    try:
        # 임베딩 실행
        embedded_vectors = embeddings.embed_documents(test_texts)

        print(f"\n✅ 임베딩 성공!")
        print(f"  - 벡터 개수: {len(embedded_vectors)}")
        print(f"  - 벡터 차원: {len(embedded_vectors[0])}")
        print(f"  - 첫 번째 벡터 샘플: {embedded_vectors[0][:5]}...")

        # 쿼리 임베딩 테스트
        print(f"\n[3단계] 쿼리 임베딩 테스트")
        query = "특별공급 신청 방법"
        query_vector = embeddings.embed_query(query)

        print(f"✅ 쿼리 임베딩 성공!")
        print(f"  - 쿼리: '{query}'")
        print(f"  - 벡터 차원: {len(query_vector)}")
        print(f"  - 벡터 샘플: {query_vector[:5]}...")

    except Exception as e:
        print(f"\n❌ 임베딩 실패: {e}")
        return False

    print("\n" + "=" * 60)
    print("✅ 모든 테스트 통과!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    test_embedding()
