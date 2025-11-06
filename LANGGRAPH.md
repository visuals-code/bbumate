# LangGraph 아키텍처 및 마이그레이션 가이드

## 📋 목차

1. [개요](#개요)
2. [아키텍처 설계 철학](#아키텍처-설계-철학)
3. [파일 구조](#파일-구조)
4. [LangGraph 구조 상세](#langgraph-구조-상세)
5. [실행 플로우](#실행-플로우)
6. [사용 방법](#사용-방법)
7. [환경 재설정 및 설치](#환경-재설정-및-설치)
8. [마이그레이션 가이드](#마이그레이션-가이드)

---

## 개요

기존 **LangChain 기반 절차적 RAG 파이프라인**을 **LangGraph State 기반 그래프 구조**로 전환했습니다.

### 주요 변경점

| 항목 | Before (LangChain) | After (LangGraph) |
|------|-------------------|-------------------|
| **구조** | 절차적 (if/else 분기) | State 기반 그래프 (Node + Edge) |
| **파일** | `chains/d002/rag_chain.py` | `src/langgraph/chain.py` |
| **상태 관리** | 수동 (함수 내 변수) | 명시적 (RAGState TypedDict) |
| **플로우 제어** | if/else 중첩 | Conditional Edge |
| **함수** | `run_rag()` | `answer_question()` |

### 기능 동일성 보장

✅ 모든 기존 기능 100% 유지:
- Question Validation (도메인 관련성 + 명확성)
- Document Retrieval & Grading
- Answer Generation (문서 기반 + 웹 검색)
- Region/Housing Type 컨텍스트 처리

---

## 아키텍처 설계 철학

### 왜 retrieval/generation은 일반 함수로 유지하고, chain.py만 LangGraph로 변환했나?

이 프로젝트는 **3-tier 아키텍처**를 따릅니다:

```
┌─────────────────────────────────────────────────────┐
│  🎯 Orchestration Layer (LangGraph)                 │  ← LangGraph 사용
│  - 복잡한 플로우 제어 (조건부 라우팅)                  │
│  - 상태 관리 (State Machine)                         │
│  - 여러 컴포넌트 조합 및 실행 순서 결정                │
│  📁 src/langgraph/chain.py                          │
└─────────────────────────────────────────────────────┘
                       ↓ 호출
┌─────────────────────────────────────────────────────┐
│  🧩 Component Layer (일반 함수)                      │  ← 일반 함수 유지
│  - 재사용 가능한 독립 함수                            │
│  - 단일 책임 원칙 (Single Responsibility)            │
│  - 테스트 용이 (Unit Test)                          │
│  📁 src/retrieval/d002/ (grader.py)                 │
│  📁 src/generation/d002/ (generator.py, etc)        │
└─────────────────────────────────────────────────────┘
                       ↓ 사용
┌─────────────────────────────────────────────────────┐
│  🔧 Utility Layer (헬퍼 함수)                        │  ← 일반 함수 유지
│  - 공통 유틸리티 (loaders, formatters)               │
│  - 설정 관리                                         │
│  📁 src/utils/d002/                                 │
└─────────────────────────────────────────────────────┘
```

### 각 계층별 역할 및 LangGraph 적용 여부

#### 1. Orchestration Layer (LangGraph) - `src/langgraph/chain.py`

**✅ LangGraph 사용 이유:**
- **복잡한 조건부 라우팅**: 3개 이상의 분기점 (Validate, Grade, Generate)
- **상태 관리 필요**: 여러 단계에서 state 공유 (question, docs, answer, sources)
- **동적 실행 경로**: Grade 결과에 따라 다른 경로 (문서 기반 vs 웹 검색)
- **확장 가능성**: Human-in-the-loop, Multi-agent 등 향후 추가 용이

**체크리스트:**
- ✅ 3개 이상의 조건부 분기점이 있는가?
- ✅ 여러 단계에서 상태를 공유해야 하는가?
- ✅ 실행 경로가 동적으로 변경되는가?
- ✅ 복잡한 플로우 시각화가 필요한가?

**결과:** → **LangGraph 적합**

#### 2. Component Layer (일반 함수) - `src/retrieval/`, `src/generation/`

**✅ 일반 함수로 유지 이유:**
- **단순 입출력**: 입력 → 처리 → 출력 (조건부 로직 없음)
- **재사용성**: RAG Chain뿐 아니라 다른 곳에서도 사용 가능
- **테스트 용이**: 독립적인 Unit Test 가능
- **명확한 인터페이스**: 함수 시그니처만으로 동작 이해 가능

**예시:**
```python
# src/retrieval/d002/grader.py
def grade_docs(question, docs, llm):
    """문서 관련성 평가 - 단순 입출력"""
    # 입력: question, docs, llm
    # 처리: LLM으로 관련성 판단
    # 출력: filtered_docs
    return filtered_docs

# src/generation/d002/generator.py
def generate_with_docs_context(question, context, llm, region, housing_type):
    """문서 기반 답변 생성 - 단순 입출력"""
    # 입력: question, context, llm, ...
    # 처리: LLM으로 답변 생성
    # 출력: answer
    return answer
```

**체크리스트:**
- ✅ 단순 입력 → 처리 → 출력 구조인가?
- ✅ 조건부 로직이 거의 없는가?
- ✅ 다른 곳에서 재사용되는 함수인가?
- ✅ 독립적으로 테스트 가능한가?

**결과:** → **일반 함수 유지**

#### 3. Utility Layer (헬퍼 함수) - `src/utils/`

**✅ 일반 함수로 유지 이유:**
- **공통 유틸리티**: 어디서나 사용되는 공통 기능
- **상태 없음 (Stateless)**: 입력만으로 출력 결정
- **단순 변환/로딩**: 복잡한 로직 없음

**예시:**
```python
# src/utils/d002/loaders.py
def load_llm():
    """LLM 로드 - 단순 유틸리티"""
    return ChatUpstage(...)

def load_vector_db(domain):
    """VectorDB 로드 - 단순 유틸리티"""
    return Chroma(...)
```

**결과:** → **일반 함수 유지**

### 설계 원칙 요약

| 계층 | LangGraph 사용 여부 | 기준 |
|------|-------------------|------|
| **Orchestration** | ✅ 사용 | 복잡한 플로우 제어, 조건부 라우팅, 상태 관리 |
| **Component** | ❌ 일반 함수 | 단순 입출력, 재사용성, 테스트 용이성 |
| **Utility** | ❌ 일반 함수 | 공통 유틸리티, Stateless |

**핵심 원칙:**
> **"LangGraph는 복잡한 플로우 제어가 필요한 오케스트레이션 레이어에만 사용하고,**
> **재사용 가능한 컴포넌트는 일반 함수로 유지하여 단순성과 재사용성을 확보한다."**

---

## 파일 구조

### 프로젝트 구조

```
KDT_BE13_TOY_PROJECT4/
├── src/
│   ├── langgraph/              # ✨ 새로 추가 (LangGraph)
│   │   ├── __init__.py
│   │   └── chain.py           # LangGraph 기반 RAG 체인 (522 lines)
│   │
│   ├── chains/                 # 기존 LangChain (레거시)
│   │   ├── index.py           # 통합 체인 (LangChain)
│   │   └── d002/
│   │       └── rag_chain.py   # LangChain 기반 RAG 체인
│   │
│   ├── retrieval/              # 일반 함수 (변경 없음)
│   │   └── d002/
│   │       ├── grader.py      # 문서 평가
│   │       └── retrieve_d002.py
│   │
│   ├── generation/             # 일반 함수 (변경 없음)
│   │   └── d002/
│   │       ├── generator.py   # 답변 생성
│   │       ├── validation.py  # 질문 검증
│   │       └── web_search.py  # 웹 검색
│   │
│   ├── utils/                  # 유틸리티 (변경 없음)
│   │   └── d002/
│   │       ├── loaders.py     # LLM/VectorDB 로더
│   │       └── context_extraction.py
│   │
│   └── api/
│       └── d002/
│           └── api_d002.py
│
├── main.py                     # FastAPI 서버 (import 변경)
├── requirements.txt
├── run_ingestion.py
└── LANGGRAPH.md               # 📖 이 문서
```

### 변경된 파일

| 파일 | 상태 | 설명 |
|------|------|------|
| `src/langgraph/chain.py` | ✅ 추가 | LangGraph 기반 RAG 체인 |
| `src/langgraph/__init__.py` | ✅ 추가 | 모듈 초기화 |
| `main.py` | ✏️ 수정 | import를 LangGraph로 변경 |
| `src/retrieval/d002/*` | ✔️ 유지 | 일반 함수 (변경 없음) |
| `src/generation/d002/*` | ✔️ 유지 | 일반 함수 (변경 없음) |
| `src/utils/d002/*` | ✔️ 유지 | 유틸리티 (변경 없음) |

---

## LangGraph 구조 상세

### State 정의 (RAGState)

```python
# src/langgraph/chain.py

class RAGState(TypedDict):
    """RAG 파이프라인 상태."""

    # 입력 파라미터
    question: str
    region: Optional[str]
    housing_type: Optional[str]
    verbose: bool
    use_grade: bool
    use_validation: bool
    k: int

    # 중간 상태
    is_valid: bool
    validation_reason: str
    clarification_question: str
    initial_docs: List[Document]
    graded_docs: List[Document]
    context: str
    rewritten_query: str
    web_results: str
    web_metadata: List[Dict[str, str]]

    # 출력
    answer: str
    sources: List[Any]
    duration_ms: int
    num_docs: int
    clarification_needed: bool
    web_search_used: bool

    # 내부 (언더스코어 prefix)
    _start_time: float
    _retriever: Any
    _llm: Any
    _final_region: Optional[str]
    _final_housing_type: Optional[str]
```

### Node 정의 (9개)

각 Node는 **d002 폴더의 일반 함수를 호출**합니다.

| Node | 역할 | 사용하는 d002 함수 |
|------|------|-------------------|
| `initialize_node` | retriever, llm 로드 | `load_llm()`, `apply_region_housing_priority()` |
| `validate_node` | 질문 검증 | `is_question_clear()`, `validate_question()` |
| `retrieve_node` | 문서 검색 | retriever.invoke() |
| `grade_node` | 문서 관련성 평가 | `grade_docs()` |
| `generate_docs_node` | 문서 기반 답변 생성 | `generate_with_docs_context()` |
| `rewrite_node` | 쿼리 재작성 | `rewrite_query()` |
| `web_search_node` | 웹 검색 | `web_search()` |
| `generate_web_node` | 웹 검색 기반 답변 생성 | `generate_with_web_context()` |
| `finalize_node` | 결과 처리 | - |

**예시 코드:**

```python
# Node 함수는 d002의 일반 함수를 호출
def validate_node(state: RAGState) -> RAGState:
    """질문 검증 Node."""

    # d002/validation.py의 일반 함수 호출
    if is_question_clear(state["question"]):
        return state

    # d002/validation.py의 일반 함수 호출
    is_valid, reason, clarification_q = validate_question(
        state["question"], state["_llm"]
    )

    return {
        **state,
        "is_valid": is_valid,
        "validation_reason": reason,
        "clarification_question": clarification_q,
    }
```

### Conditional Edge 정의 (3개)

기존 if/else 로직을 Conditional Edge로 변환:

```python
def should_continue_after_validate(state: RAGState) -> Literal["retrieve", "end"]:
    """Validation 후 라우팅."""
    if state.get("is_valid", True):
        return "retrieve"
    return "end"

def should_continue_after_grade(state: RAGState) -> Literal["generate_docs", "rewrite"]:
    """Grade 후 라우팅."""
    graded_docs = state.get("graded_docs", [])
    if graded_docs:
        return "generate_docs"
    return "rewrite"

def should_continue_after_generate_docs(state: RAGState) -> Literal["rewrite", "finalize"]:
    """Generate Docs 후 라우팅."""
    answer = state.get("answer", "")

    # "정보 없음" 패턴 감지
    no_info_patterns = [
        "제공된 문서에는 해당 정보가 없습니다",
        "정보가 없습니다",
        "찾을 수 없습니다",
    ]

    if any(pattern in answer.lower() for pattern in no_info_patterns):
        return "rewrite"  # 웹 검색 경로로
    return "finalize"
```

### 그래프 구성 (build_rag_graph)

```python
def build_rag_graph() -> StateGraph:
    """RAG 파이프라인 그래프 구성."""
    workflow = StateGraph(RAGState)

    # Nodes 추가
    workflow.add_node("initialize", initialize_node)
    workflow.add_node("validate", validate_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade", grade_node)
    workflow.add_node("generate_docs", generate_docs_node)
    workflow.add_node("rewrite", rewrite_node)
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("generate_web", generate_web_node)
    workflow.add_node("finalize", finalize_node)

    # Entry point
    workflow.set_entry_point("initialize")

    # Edges 추가
    workflow.add_edge("initialize", "validate")
    workflow.add_conditional_edges(
        "validate",
        should_continue_after_validate,
        {"retrieve": "retrieve", "end": "finalize"},
    )
    workflow.add_edge("retrieve", "grade")
    workflow.add_conditional_edges(
        "grade",
        should_continue_after_grade,
        {"generate_docs": "generate_docs", "rewrite": "rewrite"},
    )
    workflow.add_conditional_edges(
        "generate_docs",
        should_continue_after_generate_docs,
        {"rewrite": "rewrite", "finalize": "finalize"},
    )
    workflow.add_edge("rewrite", "web_search")
    workflow.add_edge("web_search", "generate_web")
    workflow.add_edge("generate_web", "finalize")
    workflow.add_edge("finalize", END)

    return workflow.compile()
```

---

## 실행 플로우

### 전체 플로우 다이어그램

```
[Start]
   ↓
┌─────────────┐
│ Initialize  │  retriever, llm 로드 (d002/loaders.py)
└─────────────┘
   ↓
┌─────────────┐
│  Validate   │  질문 검증 (d002/validation.py)
└─────────────┘
   ↓ (유효?)
   ├─ Yes ────────────────┐
   │                      ↓
   │              ┌─────────────┐
   │              │  Retrieve   │  문서 검색 (retriever)
   │              └─────────────┘
   │                      ↓
   │              ┌─────────────┐
   │              │    Grade    │  문서 평가 (d002/grader.py)
   │              └─────────────┘
   │                      ↓ (문서 있음?)
   │                      ├─ Yes ────────────┐
   │                      │                  ↓
   │                      │          ┌──────────────┐
   │                      │          │Generate Docs │  문서 기반 답변 (d002/generator.py)
   │                      │          └──────────────┘
   │                      │                  ↓ (정보 있음?)
   │                      │                  ├─ Yes ──────────┐
   │                      │                  │                ↓
   │                      │                  │         ┌──────────┐
   │                      │                  │         │Finalize  │
   │                      │                  │         └──────────┘
   │                      │                  │                ↓
   │                      │                  │             [END]
   │                      │                  │
   │                      │                  └─ No ───────────┐
   │                      │                                   ↓
   │                      └─ No ─────────────────────────────┐│
   │                                                          ↓↓
   │                                                  ┌─────────────┐
   │                                                  │   Rewrite   │  쿼리 재작성 (d002/web_search.py)
   │                                                  └─────────────┘
   │                                                          ↓
   │                                                  ┌─────────────┐
   │                                                  │ Web Search  │  웹 검색 (d002/web_search.py)
   │                                                  └─────────────┘
   │                                                          ↓
   │                                                  ┌──────────────┐
   │                                                  │Generate Web  │  웹 기반 답변 (d002/generator.py)
   │                                                  └──────────────┘
   │                                                          ↓
   │                                                   ┌──────────┐
   │                                                   │Finalize  │
   │                                                   └──────────┘
   │                                                          ↓
   │                                                       [END]
   │
   └─ No ────────────────────────────────────────────────────────────┐
                                                                     ↓
                                                              ┌──────────┐
                                                              │Finalize  │
                                                              └──────────┘
                                                                     ↓
                                                                  [END]
```

### 경로별 실행 예시

#### 경로 1: 문서 기반 답변 성공
```
Initialize → Validate (유효) → Retrieve → Grade (문서 3개 → 2개)
→ Generate Docs (정보 있음) → Finalize → END
```

#### 경로 2: 웹 검색 경로 (Grade 실패)
```
Initialize → Validate (유효) → Retrieve → Grade (문서 0개)
→ Rewrite → Web Search → Generate Web → Finalize → END
```

#### 경로 3: 웹 검색 경로 (Generate Docs 실패)
```
Initialize → Validate (유효) → Retrieve → Grade (문서 2개)
→ Generate Docs (정보 없음 패턴 감지)
→ Rewrite → Web Search → Generate Web → Finalize → END
```

#### 경로 4: Validation 실패
```
Initialize → Validate (도메인 무관) → Finalize (에러 메시지) → END
```

---

## 사용 방법

### main.py에서 사용

```python
# main.py (Line 39)
from src.langgraph.chain import answer_question

# API 엔드포인트 (Line 120)
res = answer_question(
    question=request.question,
    k=3,
    use_grade=True,
    use_validation=True,
    region=request.region,
    housing_type=request.housing_type,
    verbose=True,
)
```

### 직접 호출

```python
from src.langgraph.chain import answer_question

# 기본 사용
result = answer_question(
    question="신혼부부 전세자금대출 조건 알려줘"
)

# 고급 사용 (지역/주거형태 포함)
result = answer_question(
    question="전세자금대출 조건 알려줘",
    k=5,                    # 검색 문서 개수
    use_grade=True,         # 문서 평가 활성화
    use_validation=True,    # 질문 검증 활성화
    region="인천",
    housing_type="전세",
    verbose=True,           # 상세 로그 출력
)
```

### 반환값

```python
{
    "answer": "신혼부부 전세자금대출 조건은...",
    "sources": ["file1.html", "file2.html"],  # 또는 웹 검색 메타데이터
    "duration_ms": 1234,
    "num_docs": 2,
    "clarification_needed": False,
    "web_search_used": False,
}
```

---

## 환경 재설정 및 설치

### 가상환경 재설정이 필요한 이유

LangGraph 마이그레이션 후, 새로운 의존성(`langgraph>=0.2.45`)이 추가되었으므로 **기존 가상환경을 삭제하고 재설정**하는 것이 안정적입니다.

### 1단계: 기존 가상환경 삭제

```bash
# 가상환경 비활성화 (활성화 상태인 경우)
deactivate

# 기존 가상환경 폴더 삭제
rm -rf venv
rm -rf venv_stable
```

### 2단계: Python 버전 확인

**권장 Python 버전: 3.10 ~ 3.12**

```bash
# Python 버전 확인
python --version
# 또는
python3 --version

# 예상 출력: Python 3.11.x 또는 Python 3.12.x
```

**Python이 없거나 버전이 낮은 경우:**

```bash
# macOS (Homebrew)
brew install python@3.11

# Ubuntu/Debian
sudo apt update
sudo apt install python3.11 python3.11-venv

# Windows
# https://www.python.org/downloads/ 에서 다운로드
```

### 3단계: 새 가상환경 생성

```bash
# Python 3.10+를 사용하여 가상환경 생성
python3 -m venv venv

# 또는 특정 버전 지정
python3.11 -m venv venv
```

### 4단계: 가상환경 활성화

```bash
# macOS/Linux
source venv/bin/activate

# Windows (PowerShell)
venv\Scripts\Activate.ps1

# Windows (CMD)
venv\Scripts\activate.bat
```

**활성화 확인:**
```bash
# 프롬프트가 (venv)로 시작하는지 확인
(venv) user@hostname:~/project$

# Python 경로 확인 (가상환경 내부여야 함)
which python
# 예상 출력: /path/to/project/venv/bin/python
```

### 5단계: pip 업그레이드 및 의존성 설치

```bash
# pip 업그레이드 (최신 버전 사용)
pip install --upgrade pip

# setuptools, wheel 업그레이드 (안정성 향상)
pip install --upgrade setuptools wheel

# requirements.txt 설치 (안정적인 순서로)
pip install -r requirements.txt
```

**설치 진행 상황:**
```
Collecting fastapi>=0.115,<1
Collecting langchain==0.3.*
Collecting langgraph>=0.2.45  # ← 새로 추가된 패키지
...
Successfully installed langchain-0.3.x langgraph-0.2.45 ...
```

**설치 확인:**
```bash
# 설치된 패키지 확인
pip list | grep -E "langchain|langgraph"

# 예상 출력:
# langchain                 0.3.x
# langchain-chroma          0.1.x
# langchain-community       0.3.x
# langchain-core            0.3.x
# langchain-text-splitters  0.3.x
# langchain-upstage         0.1.x
# langgraph                 0.2.45  # ← 확인!
```

### 6단계: 환경 변수 설정

`.env` 파일이 있는지 확인하고, 없으면 생성:

```bash
# .env 파일 확인
cat .env

# 없으면 생성
cat > .env << 'EOF'
UPSTAGE_API_KEY=YOUR_UPSTAGE_API_KEY
UPSTAGE_EMBEDDING_MODEL=solar-embedding-1-large
UPSTAGE_CHAT_MODEL=solar-1-mini-chat
CHROMA_DB_DIR=./chroma_storage
COLLECTION_NAME=unified_rag_collection
TAVILY_API_KEY=YOUR_TAVILY_API_KEY
EOF
```

**환경 변수 확인:**
```bash
# .env 파일 읽기 테스트
python -c "from dotenv import load_dotenv; import os; load_dotenv(); print('UPSTAGE_API_KEY:', os.getenv('UPSTAGE_API_KEY')[:10] + '...')"
```

### 7단계: 데이터 Ingestion 실행

**VectorDB가 없거나 재생성이 필요한 경우:**

```bash
# run_ingestion.py 실행 (통합 DB 생성)
python run_ingestion.py

# 예상 출력:
# [Ingestion] 도메인: all (통합 모드)
# [Ingestion] d001 처리 중...
# [Ingestion] d002 처리 중...
# ...
# [Ingestion] 완료! 총 X개 문서, Y개 청크 저장
```

**선택적으로 특정 도메인만 실행:**
```bash
# d002만 실행
python -c "from src.ingestion.index import ingest; ingest(domain='d002')"
```

### 8단계: 서버 실행 및 테스트

```bash
# FastAPI 서버 실행
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# 예상 출력:
# INFO:     Started server process [12345]
# INFO:     Application startup complete.
# INFO:     Uvicorn running on http://0.0.0.0:8000
```

**테스트:**

```bash
# 헬스체크
curl http://localhost:8000/

# 질의 테스트
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "신혼부부 전세자금대출 조건 알려줘",
    "region": "인천",
    "housing_type": "전세"
  }'
```

**Swagger UI 테스트:**
```
http://localhost:8000/docs
```

### 문제 해결

#### Import 에러
```bash
# 에러: ModuleNotFoundError: No module named 'langgraph'
# 해결: langgraph 재설치
pip install langgraph>=0.2.45
```

#### 환경 변수 에러
```bash
# 에러: UPSTAGE_API_KEY 환경변수가 필요합니다
# 해결: .env 파일 확인 및 수정
cat .env
```

#### VectorDB 에러
```bash
# 에러: VectorDB가 비어있습니다
# 해결: run_ingestion.py 실행
python run_ingestion.py
```

---

## 마이그레이션 가이드

### 기존 코드에서 마이그레이션

#### Before (LangChain)

```python
# main.py (기존)
from src.chains.d002.rag_chain import run_rag

res = run_rag(
    query=request.question,
    domain="d002",
    verbose=True,
    use_grade=True,
    use_validation=True,
    region=request.region,
    housing_type=request.housing_type,
)
```

#### After (LangGraph)

```python
# main.py (신규)
from src.langgraph.chain import answer_question

res = answer_question(
    question=request.question,  # query → question
    k=3,
    use_grade=True,
    use_validation=True,
    region=request.region,
    housing_type=request.housing_type,
    verbose=True,
)
```

**변경 사항:**
- ✅ `run_rag()` → `answer_question()`
- ✅ `query` → `question`
- ✅ `domain` 파라미터 제거 (통합 DB 사용)
- ✅ `k` 파라미터 추가 (검색 문서 개수)

### 반환값 비교

**동일합니다!**

```python
# Before & After 모두 동일
{
    "answer": str,
    "sources": List[str] | List[Dict[str, str]],
    "duration_ms": int,
    "num_docs": int,
    "clarification_needed": bool,
    "web_search_used": bool,
}
```

---

## 향후 확장 가능성

LangGraph 기반으로 다음 기능을 쉽게 추가할 수 있습니다:

### 1. Human-in-the-loop
```python
# 사용자 피드백 Node 추가
workflow.add_node("human_feedback", human_feedback_node)
workflow.add_edge("generate_docs", "human_feedback")
workflow.add_conditional_edges(
    "human_feedback",
    should_continue_after_feedback,
    {"approved": "finalize", "rejected": "rewrite"},
)
```

### 2. Multi-agent
```python
# 여러 전문가 Agent 병렬 실행
workflow.add_node("expert_1", expert_1_node)
workflow.add_node("expert_2", expert_2_node)
workflow.add_node("synthesize", synthesize_node)

# Parallel execution
workflow.add_edge("retrieve", "expert_1")
workflow.add_edge("retrieve", "expert_2")
workflow.add_edge("expert_1", "synthesize")
workflow.add_edge("expert_2", "synthesize")
```

### 3. Memory
```python
# 대화 기록 관리
class RAGState(TypedDict):
    # ... 기존 필드
    conversation_history: List[Dict[str, str]]  # 추가

workflow.add_node("update_memory", update_memory_node)
workflow.add_edge("finalize", "update_memory")
```

### 4. Streaming
```python
# 답변 생성 중간 결과 스트리밍
async def generate_docs_node_streaming(state: RAGState):
    async for chunk in llm.astream(...):
        yield chunk  # 중간 결과 스트리밍
```

---

## 참고 자료

- **LangGraph 공식 문서**: https://langchain-ai.github.io/langgraph/
- **LangChain 0.3.x 문서**: https://python.langchain.com/docs/
- **Upstage API 문서**: https://console.upstage.ai/docs

---

## 요약

| 항목 | 내용 |
|------|------|
| **구조** | LangChain 절차적 → LangGraph State 기반 그래프 |
| **파일** | `src/langgraph/chain.py` (522 lines) |
| **Node** | 9개 (initialize, validate, retrieve, grade, generate_docs, rewrite, web_search, generate_web, finalize) |
| **d002 함수** | retrieval/generation 일반 함수 그대로 사용 |
| **설계 원칙** | Orchestration Layer만 LangGraph, Component/Utility Layer는 일반 함수 유지 |
| **기능** | 100% 동일 (기존 run_rag()와 동일한 플로우) |
| **확장성** | Human-in-the-loop, Multi-agent, Memory, Streaming 추가 용이 |

**핵심 메시지:**
> **LangGraph는 복잡한 플로우 제어가 필요한 오케스트레이션 레이어에만 사용하고,**
> **재사용 가능한 컴포넌트는 일반 함수로 유지하여 단순성과 재사용성을 확보합니다.**

---

**🤖 Generated with [Claude Code](https://claude.com/claude-code)**
