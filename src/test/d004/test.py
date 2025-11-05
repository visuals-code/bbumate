import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_upstage import UpstageEmbeddings

load_dotenv()

# 설정
api_key = os.getenv("UPSTAGE_API_KEY")
db_path = os.getenv("CHROMA_DB_DIR", "./chroma_storage")
embedding_model = os.getenv("UPSTAGE_EMBEDDING_MODEL", "solar-embedding-1-large")
collection_name = os.getenv("COLLECTION_NAME", "pdf_promotion_chunks")

print(f"DB 경로: {db_path}")
print(f"컬렉션: {collection_name}")

# 임베딩 초기화
embeddings = UpstageEmbeddings(api_key=api_key, model=embedding_model)

# 벡터스토어 로드
vectorstore = Chroma(
    collection_name=collection_name,
    embedding_function=embeddings,
    persist_directory=db_path,
)

# 문서 개수 확인
collection_data = vectorstore.get()
count = len(collection_data["ids"])

print(f"\n저장 검증: {count}개 문서 확인됨")

if count == 0:
    print("\n❌ 벡터 DB가 비어있습니다!")
    print("   → pipeline.py를 실행하여 PDF를 임베딩하세요.")
else:
    # 샘플 출력
    print(f"\n첫 문서 ID: {collection_data['ids'][0]}")
    print(f"첫 문서 메타데이터: {collection_data['metadatas'][0]}")

    # 추가 샘플 3개
    print(f"\n--- 샘플 문서 3개 ---")
    for i in range(min(3, count)):
        metadata = collection_data["metadatas"][i]
        print(f"\n[{i+1}]")
        print(f"  ID: {collection_data['ids'][i]}")
        print(f"  source_file: {metadata.get('source_file', 'N/A')}")
        print(f"  heading: {metadata.get('heading', 'N/A')[:60]}...")
        if metadata.get("url"):
            print(f"  url: {metadata['url']}")

    # 검색 테스트
    print(f"\n--- 검색 테스트 ---")
    test_query = "신혼부부 전세 지원"
    print(f"쿼리: '{test_query}'")

    docs = vectorstore.similarity_search(test_query, k=3)
    print(f"검색 결과: {len(docs)}개 문서")

    if docs:
        print(f"\n첫 번째 검색 결과:")
        print(f"  heading: {docs[0].metadata.get('heading', 'N/A')[:60]}...")
        print(f"  source_file: {docs[0].metadata.get('source_file', 'N/A')}")
        print(f"  content: {docs[0].page_content[:100]}...")

print(f"\n✅ 벡터 DB 확인 완료")
