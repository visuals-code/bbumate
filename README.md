<p align="center">
  <img src="/public/thumbnail.png" alt="Thumbnail" width="800">
</p>

## 쀼메이트 👰🏻🤵🏻 - 신혼부부 챗봇 상담 서비스

> 📚 **상세 문서**: 전체 시스템 아키텍처 및 데이터 흐름은 [ARCHITECTURE.md](ARCHITECTURE.md)를 참고하세요.
> 📖 **설치 및 사용법**: [D001_GUIDE.md](d001docs/D001_GUIDE.md)에서 자세한 가이드를 확인할 수 있습니다.

### 1. 서비스 개요

- **목적**: "쀼메이트"는 결혼 후 주거·대출·복지·기업 혜택 등 각 기관에 흩어진 정보를 RAG 기반으로 통합하여, 사용자의 상황에 맞는 정책을 추천하는 AI 상담 플랫폼입니다.
<<<<<<< HEAD
- **사용 흐름**: 사용자가 지역, 거주 유형 등 조건을 입력 → 질문 → 관련 정책/대출/지원금 정보를 검색 → 요약·비교 → **출처가 포함된 답변** 제공합니다.
- **🆕 새로운 기능: 질문 명확화 (Clarification)** - 모호한 질문을 자동 감지하고 명확화 질문 제시 → 더 정확한 답변 제공 ([자세히 보기](d001docs/D001_GUIDE.md#3-질문-명확화-기능이-포함된-질의))
- **거주 유형 중복 선택**: 전세, 월세, 자가, 무주택 중 여러 개 선택 가능
=======
- **사용 흐름**: 사용자가 소득, 지역, 자녀 수 등 조건을 입력 → 관련 정책/대출/지원금 정보를 검색 → 요약·비교 → **출처가 포함된 답변** 제공합니다.
>>>>>>> main

### 2. 기획 의도

1. **정보의 파편화**: 주택도시기금/보건복지부/마이홈포털 등 여러 사이트를 오가며 정보를 수집해야 하며, 표현이 제각각이라 최신·정확한 이해가 어려움
2. **AI를 통한 통합 상담**: RAG로 공공데이터 API/정책문서/공고문을 벡터DB에 저장하고, 질의와 매칭된 문서를 기반으로 안전하게 답변 생성. 고시문·보도자료·공공데이터 링크 등 **근거 문서와 출처**를 함께 제시해 신뢰성 강화
3. **주요 기능** (v1.0)

- AI 챗봇 상담
- 출처 기반 근거 제시

### 3. 아키텍처/설계 (LangChain + Chroma + Upstage)
<<<<<<< HEAD
- **FastAPI**: 경량 REST API 서버
  - `/` 헬스체크
  - `/query` 기본 RAG 질의
  - `/adaptive/query` 적응형 RAG (문서 품질 평가 + 웹 검색)
  - **🆕 `/adaptive/query-with-clarification`** 명확화 기능 포함 질의
  - **🆕 `/adaptive/answer-with-clarification`** 명확화 응답 처리
=======

- **FastAPI**: 경량 REST API 서버 (`/` 헬스체크, `/query` 질의 엔드포인트)
>>>>>>> main
- **LangChain 0.3 (Runnables API)**: 체인 구성
  - Retriever(Chroma) → 컨텍스트 포맷 → Prompt → Upstage Chat → 문자열 파싱
- **벡터DB: Chroma**: 로컬 퍼시스턴스(`CHROMA_DB_DIR`) 사용
- **모델**: Upstage Embedding/Chat 모델을 `.env`로 관리
<<<<<<< HEAD
- **처리 흐름** ([상세 아키텍처 문서](ARCHITECTURE.md))
    1. **사용자 입력** (질문 + 지역 + 거주 유형)
       - 거주 유형: 전세, 월세, 자가, 무주택 (중복 선택 가능)
    2. **🆕 Clarification Check** (질문 명확성 판단 - 규칙 기반 + LLM 기반)
    3. **🆕 Re-ask / Clarification** (모호한 경우 명확화 질문 제시)
    4. **Retrieve** (VectorDB에서 관련 문서 검색, k개)
    5. **Grade** (LLM으로 문서 관련성 평가 + 점수 부여)
    6. **🆕 Rerank** (관련성 점수 기준으로 문서 재정렬, top_k개 선택)
    7. **Decision** (문서 품질 판단)
       - 충분함 → DB 문서로 답변 생성
       - 부족함 → Re-write Query → Web Search → 답변 생성
    8. **LLM Answer Generation** (Upstage Chat 모델로 답변 생성)
    9. **Answer** (출력)
=======
- **처리 흐름**
  1. Question (사용자 질문 입력)
  2. Clarification Check (질문 명확성 판단)
  3. Re-ask / Clarification (질문 재질문 또는 구체화)
  4. Retrieve (VectorDB에서 관련 문서 검색)
  5. Grade (문서 관련도 평가 및 재정렬)
  6. Re-write Query (쿼리 재작성)
  7. Web Search (외부 데이터 검색)
  8. LLM Answer Generation (Upstage Chat 모델로 답변 생성)
  9. Answer (출력)
>>>>>>> main

### 4. RAG 파이프라인

<p align="center">
  <img src="/public/rag_pipeline.png" alt="RAG Pipeline Diagram" width="800">
</p>

### 5. 폴더 구조 설계

```text
KDT_BE13_TOY_PROJECT4/
├── data/                 # PDF 및 텍스트 원본
├── scripts/              # 크롤링 및 전처리 스크립트
├── src/
│   ├── ingestion/        # 수집 → 청킹 → 임베딩 → 저장
│   ├── retrieval/        # 쿼리 기반 문서 검색
│   ├── generation/       # LLM 응답 생성
│   ├── chains/           # LangChain / LangGraph 체인 정의
│   │   ├── rag_chain.py
│   │   ├── adaptive_rag_chain.py
│   │   └── clarification_chain.py  🆕 질문 명확화 체인
│   ├── utils/            # 공통 유틸, 로깅, 캐싱
│   └── api/              # FastAPI 엔드포인트
├── .env.example
├── requirements.txt
├── main.py
└── README.md
```

### 6. 기술 스택

- **Language**: Python 3.13
- **Web**: FastAPI, Uvicorn
- **AI Orchestration**: LangChain 0.3.x, langchain-community 0.3.x, langchain-upstage
- **Vector DB**: ChromaDB
- **LLM/Embedding**: Upstage Solar 시리즈

### 7. 환경 변수 (.env)

프로젝트 루트에 `.env` 파일을 생성하고 아래 키를 설정하세요.

```dotenv
UPSTAGE_API_KEY=YOUR_UPSTAGE_API_KEY
UPSTAGE_EMBEDDING_MODEL=solar-embedding-1-large
UPSTAGE_CHAT_MODEL=solar-1-mini-chat
CHROMA_DB_DIR=./chroma_storage
```

- `UPSTAGE_EMBEDDING_MODEL` / `UPSTAGE_CHAT_MODEL`: 필요 시 모델명을 교체하여 실험 가능
- `CHROMA_DB_DIR`: Chroma 퍼시스트 디렉터리 경로

### 8. 설치 및 실행 방법

1. 가상환경 생성/활성화(선택)

```bash
python3 -m venv venv
source venv/bin/activate
```

2. 패키지 설치(버전 고정: LangChain 0.3)

```bash
pip install -r requirements.txt
```

3. `.env` 파일 작성(위 예시 참조)

4. `.env` 파일 확인

```bash
cat .env
```

5. 서버 실행

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 9. API 사용 방법

- **헬스체크**: `GET /`
- **질의**: `POST /query`
  - Request Body
    ```json
    { "question": "신혼부부 전세자금대출 조건 알려줘" }
    ```
  - Response 예시
    ```json
    { "answer": "...모델 응답 문자열..." }
    ```

### 10. 개발/운영 팁

- IDE에서 프로젝트의 **가상환경 인터프리터**를 선택하세요(예: `venv/bin/python`).
- `.env`를 수정한 경우 **서버 재시작** 또는 `--reload` 옵션 사용을 권장합니다.
- Chroma 인덱스가 비어있다면 검색이 정상적으로 이루어지지 않습니다. 사전 임베딩/적재를 수행하세요.

---

본 서비스는 신뢰 가능한 출처를 기반으로 신혼부부 맞춤형 정책 정보를 안전하게 안내하는 것을 목표로 합니다.
