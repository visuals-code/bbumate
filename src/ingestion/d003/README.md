## [D003 가족 및 복지 제도] 파이프라인

이 디렉토리는 PDF 문서를 로드 → 텍스트 청킹 → 임베딩(Upstage) → Chroma 벡터DB에 저장하는 파이프라인을 구성합니다.

### 파이프라인 단계

1. 문서 로드 (PDF → 텍스트)
2. 텍스트 작은 조각(chunks)으로 분할
3. 임베딩(Upstage)으로 벡터 변환
4. 벡터DB(Chroma) 저장 및 영속화

### 구성 파일

- `loader.py`: PDF 로더HTML로 감싸서 반환
- `splitter.py`: 텍스트 청킹
- `embedder.py`: Upstage 임베딩 초기화 (env 기반)
- `vectorstore.py`: ChromaDB에 문서 벡터 영속 저장
- `pipeline.py`: CLI 파이프라인 실행(로드 → 청킹 → 임베딩 → 저장)

### 환경 변수 (.env)

프로젝트 루트의 `.env`에 다음 키가 필요합니다.

```dotenv
UPSTAGE_API_KEY=YOUR_UPSTAGE_API_KEY
UPSTAGE_EMBEDDING_MODEL=solar-embedding-1-large
CHROMA_DB_DIR=./chroma_storage
```

### 설치

프로젝트 루트에서 의존성을 설치하세요.

```bash
pip install -r requirements.txt
```

### 실행 방법

```bash
python -m src.ingestion.d003.pipeline
```
