# D001 (신혼부부 주택공급 정책) 실행 가이드

## 사전 준비

### 1. 환경 변수 설정
프로젝트 루트에 `.env` 파일을 생성하고 다음 변수를 설정하세요:

```dotenv
UPSTAGE_API_KEY=YOUR_UPSTAGE_API_KEY
UPSTAGE_EMBEDDING_MODEL=embedding-query
UPSTAGE_CHAT_MODEL=solar-1-mini-chat
TAVILY_API_KEY=YOUR_TAVILY_API_KEY  # 웹 검색 기능
```

### 2. 데이터 디렉토리 확인
`data/d001/` 디렉토리에 다음 구조가 있어야 합니다:

```
data/d001/
└── housing/          # PDF 원본 파일들
```

프로젝트 루트에 다음 디렉토리가 생성됩니다:
```
chroma_storage/       # VectorDB 저장 경로 (임베딩 실행 시 자동 생성)
```

---

## 실행 단계

### 1단계: PDF 임베딩 및 VectorDB 저장

PDF 파일들을 로드하여 임베딩하고 Chroma VectorDB에 저장합니다.

**실행 명령:**
```bash
python -m src.ingestion.d001.pipeline
```

**출력 예시:**
```
...(생략)
📊 [임베딩 완료] 총 소요 시간: 123.45초 (2.06분)

============================================================
--- 파이프라인 실행 완료 ---
📊 [전체 파이프라인] 총 소요 시간: 125.67초 (2.09분)
============================================================
...
```

## 전체 실행 순서 요약

```bash
# 1. PDF 임베딩 및 VectorDB 저장
python -m src.ingestion.d001.pipeline

# 2. API 서버 실행
python -m src.api.d001.app
# 또는
uvicorn src.api.d001.app:app --host 0.0.0.0 --port 8000 --reload
```

**Swagger UI 접속:**
```
http://127.0.0.1:8000/docs
```

---

## 주요 기능

### 1. Basic RAG
- 기본적인 검색-생성 파이프라인
- 벡터 DB에서 유사 문서를 검색하여 답변 생성

### 2. Adaptive RAG 
- **문서 관련성 평가 (Grading)**: 검색된 문서의 관련성을 평가
- **문서 재순위화 (Reranking)**: 관련성 높은 문서를 우선순위화
- **웹 검색 폴백**: 관련 문서가 충분하지 않을 때 웹 검색 수행
- **질문 명확화**: 모호한 질문에 대해 추가 정보 요청
- **사용자 컨텍스트**: 지역, 주거형태 등 사용자 정보 활용

### 3. 답변 포맷팅
- **순수 텍스트**: `answer` 필드
- **마크다운**: `answer_md` 필드 (금액/비율 강조, 줄바꿈 처리)
- **HTML**: `answer_html` 필드 (금액/비율 `<strong>`, 줄바꿈 `<br/>`)

### 4. 출처 정보
- 답변에 사용된 문서의 출처 정보 제공
- 파일 경로, 제목, URL (있는 경우) 포함

### 5. 모니터링
- Prometheus 메트릭 수집: `http://127.0.0.1:8000/metrics`
- 요청/응답 로깅
- 성능 메트릭 추적

---

## 환경 변수 전체 목록

```dotenv
# 필수 환경 변수
UPSTAGE_API_KEY=your_upstage_api_key
UPSTAGE_EMBEDDING_MODEL=embedding-query
UPSTAGE_CHAT_MODEL=solar-1-mini-chat

# 선택적 환경 변수
TAVILY_API_KEY=your_tavily_api_key
CHROMA_DB_DIR=./chroma_storage
USE_ADAPTIVE_RAG=true
USE_WEB_SEARCH=false
USE_MOCK_WEB_SEARCH=false
RELEVANCE_THRESHOLD=0.5
CONFIDENCE_THRESHOLD=0.5
DEFAULT_RETRIEVAL_K=5
RATE_LIMIT_PER_MINUTE=60
CORS_ORIGINS=http://localhost:3000,http://localhost:8080
```
