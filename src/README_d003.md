## [D003 가족 및 복지 제도] RAG 파이프라인

### RAG 파이프라인 단계

1. 문서 로드 (PDF → 텍스트)
2. 텍스트 작은 조각(chunks)으로 분할
3. 임베딩(Upstage)으로 벡터 변환
4. 벡터DB(Chroma) 저장 및 영속화
5. 쿼리와 관련된 조각 검색(Retriever)
6. 조각을 프롬프트에 포함
7. LLM 체인 실행(프롬프트 + LLM + 체인)
8. 응답 출력

### 구성 파일

- [`loader.py`](ingestion/d003/loader.py): PDF 로더
- [`splitter.py`](ingestion/d003/splitter.py): 텍스트 청킹 (HTML 모드 시 BeautifulSoup으로 정제 후 분할)
- [`embedder.py`](ingestion/d003/embedder.py): Upstage 임베딩 초기화 (env 기반)
- [`vectorstore.py`](ingestion/d003/vectorstore.py): ChromaDB에 문서 벡터 영속 저장
- [`pipeline.py`](ingestion/d003/pipeline.py): 데이터전처리 파이프라인 실행 (로드 → 청킹 → 임베딩 → 저장)
- [`retriever.py`](retrieval/d003/retriever.py): ChromaDB에서 관련 청크 검색
- [`prompting.py`](generation/d003/prompting.py): 컨텍스트 포맷팅 + 프롬프트 정의
- [`qa_chain.py`](chains/d003/qa_chain.py): LLM 체인 정의/실행 (Runnables API)
- [`run_query.py`](test/d003/run_query.py): 실행 CLI (검색 결과와 답변 출력)

### 환경 변수 (.env)

- `UPSTAGE_API_KEY`: Upstage API 인증 키
- `UPSTAGE_EMBEDDING_MODEL`: 임베딩 모델명 (예: `solar-embedding-1-large`)
- `UPSTAGE_CHAT_MODEL`: 대화형 LLM 모델명 (예: `solar-1-mini-chat`)
- `CHROMA_DB_DIR`: Chroma 벡터 스토어 퍼시스트 디렉터리 경로 (예: `./chroma_storage`)

선택: LangSmith 추적을 원하면 아래를 추가하세요.

```dotenv
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_API_KEY=YOUR_LANGSMITH_API_KEY
LANGCHAIN_PROJECT=your-project-name
```

### 설치

프로젝트 루트에서 의존성을 설치하세요.

```bash
pip install -r requirements.txt
```

### 색인(ingestion)

PDF를 색인(임베딩 → Chroma 저장)합니다.

- 코드 기본값

```bash
python -m src.ingestion.d003.pipeline
```

- 가능한 옵션들(전체)

```bash
python -m src.ingestion.d003.pipeline \
  --mode html \
  --data-dir ./data \
  --chunk-size 1000 \
  --chunk-overlap 200
```

### 질의 테스트 (Retrieval + Generation)

관련 조각을 검색하고, 프롬프트에 포함해 LLM 응답을 출력합니다. (기본 top-k는 3입니다. 필요 시 `--k`로 변경 가능.)

```bash
python -m src.test.d003.run_query --question "신혼부부 전세자금대출 조건 알려줘"
```

출력 예시

- [Retrieved Chunks]: 상위 k개 조각의 `source`와 내용 일부
- [Answer]: LLM 생성 답변 문자열

### 참고/주의

- 현재 버전은 리포지토리 내 모든 `data/` 폴더를 재귀적으로 스캔합니다.
- HTML 모드를 유지하는 이유(요약): 확장성(HTML 소스 수용), 견고성(PyPDFLoader 실패 시 우회), 정규화(BeautifulSoup로 텍스트 정제), 미래 대응(OCR 등 후처리 수용).
- 이미지형(스캔) PDF는 `mode=html`만으로 부족할 수 있습니다. OCR 전처리를 검토하세요.
