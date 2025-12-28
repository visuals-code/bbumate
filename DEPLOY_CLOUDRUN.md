# Google Cloud Run 배포 가이드

### 배포 아키텍처

```
┌─────────────────────────────┐
│   Cloud Run Container       │
│                             │
│  ┌──────────┐  ┌─────────┐  │
│  │ FastAPI  │  │ ChromaDB│  │
│  │  서버     │  │ (37MB)  │  │
│  └──────────┘  └─────────┘  │
└─────────────────────────────┘
```

**특징:**
- FastAPI 서버와 ChromaDB가 하나의 컨테이너에 포함
- 빌드 시점에 ChromaDB 자동 생성 (1,156 chunks)
- Stateless 컨테이너로 자동 스케일링
- 무료 티어 내에서 운영 가능

### 배포 플로우

```
코드 Push (main)
    ↓
GitHub Actions 시작
    ↓
Docker 빌드 + Ingestion 자동 실행
    ↓
ChromaDB 포함된 이미지 생성
    ↓
Cloud Run 배포 완료
```

---

## 🔧 사전 준비

### 필요한 것
- ✅ Google Cloud 계정
- ✅ 결제 정보 등록 (무료 티어 사용 가능, $300 크레딧 제공)
- ✅ GitHub 계정
- ✅ Upstage API 키 ([가입하기](https://console.upstage.ai))

### 로컬 환경
- Docker Desktop 설치
- Git 설치

---

## ☁️ Google Cloud 설정

### 1. Google Cloud Console 접속
👉 https://console.cloud.google.com

### 2. 새 프로젝트 생성
프로젝트 ID 예시: `bbumate-api-1`

### 3. Cloud Shell 열기
화면 우측 상단의 `>_` 아이콘 클릭 (또는 `Ctrl + \``)

### 4. 환경 설정 스크립트 실행

```bash
# 프로젝트 ID 설정 (your-project-id를 실제 ID로 변경)
export PROJECT_ID="your-project-id"
gcloud config set project $PROJECT_ID

# API 활성화
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable artifactregistry.googleapis.com

# Artifact Registry 생성
gcloud artifacts repositories create bbumate-api \
  --repository-format=docker \
  --location=asia-northeast3 \
  --description="Docker repository for Bbumate API"

# 서비스 계정 생성
gcloud iam service-accounts create github-actions \
  --display-name="GitHub Actions Deployer"

# 권한 부여
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:github-actions@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/run.admin"

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:github-actions@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/storage.admin"

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:github-actions@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/artifactregistry.admin"

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:github-actions@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/iam.serviceAccountUser"

# 서비스 계정 키 생성
gcloud iam service-accounts keys create key.json \
  --iam-account=github-actions@${PROJECT_ID}.iam.gserviceaccount.com

# 키 내용 확인
cat key.json
```

### 5. 서비스 계정 키 저장
- `key.json` 전체 내용을 복사
- 안전한 곳에 보관 (GitHub Secrets에 사용)

---

## 🔐 GitHub Secrets & Variables 설정

### 설정 경로
Repository → Settings → Secrets and variables → Actions

### 필수 Secrets (민감 정보)

**New repository secret** 클릭하여 추가:

| Secret 이름 | 값 예시 | 설명 |
|-------------|---------|------|
| `GCP_PROJECT_ID` | `bbumate-api-1` | Google Cloud 프로젝트 ID |
| `GCP_SA_KEY` | `key.json` 전체 내용 | 서비스 계정 키 (JSON 형식) |
| `UPSTAGE_API_KEY` | `up_xxxxxxxxxxxxx` | Upstage API 키 |

### 필수 Variables (설정 값)

**Variables** 탭 → **New repository variable** 클릭하여 추가:

| Variable 이름 | 값 | 설명 |
|---------------|-----|------|
| `UPSTAGE_EMBEDDING_MODEL` | `solar-embedding-1-large` | 임베딩 모델명 |
| `UPSTAGE_CHAT_MODEL` | `solar-1-mini-chat` | 채팅 모델명 |

### 선택 Secrets

| Secret 이름 | 필요 여부 | 설명 |
|-------------|----------|------|
| `TAVILY_API_KEY` | ❌ 불필요 | 현재 Mock 웹 검색 사용 (`USE_MOCK_WEB_SEARCH=true`) |

---

## 🧪 로컬 테스트

배포 전 로컬에서 Docker 이미지를 테스트합니다.

### 1. 환경 변수 로드
```bash
source .env
```

### 2. Docker 이미지 빌드
```bash
docker build \
  --build-arg UPSTAGE_API_KEY="$UPSTAGE_API_KEY" \
  --build-arg UPSTAGE_EMBEDDING_MODEL="$UPSTAGE_EMBEDDING_MODEL" \
  --build-arg UPSTAGE_CHAT_MODEL="$UPSTAGE_CHAT_MODEL" \
  -t bbumate-api:test .
```

**빌드 시 자동 실행:**
- `run_ingestion.py` 실행
- 5개 도메인(d001-d005) 데이터 처리
- ChromaDB 생성 (약 3분 소요)
- 총 1,156 chunks 생성

### 3. 컨테이너 실행
```bash
docker run -p 8080:8080 --env-file .env bbumate-api:test
```

### 4. 테스트
```bash
# 헬스체크
curl http://localhost:8080/api/health

# API 테스트
curl -X POST http://localhost:8080/api/query \
  -H "Content-Type: application/json" \
  -d '{"question": "신혼부부 전세자금 대출이 뭐야?"}'
```

### 5. 정리
```bash
# 컨테이너 중지
docker stop $(docker ps -q --filter ancestor=bbumate-api:test)
```

---

## 🚀 배포 실행

### 자동 배포 (추천)

main 브랜치에 push하면 GitHub Actions가 자동으로 배포합니다.

**배포 확인:**
1. GitHub Repository → **Actions** 탭에서 진행 상황 확인
2. 약 5-10분 소요 (빌드 3분 + 배포 2분)
3. 완료 후 Cloud Run 콘솔에서 서비스 URL 확인

---

## 🛠️ 운영 가이드

### 서비스 상태 확인
```bash
gcloud run services describe bbumate-api --region asia-northeast3
```

### 배포된 서비스 URL 확인
```bash
gcloud run services describe bbumate-api \
  --region asia-northeast3 \
  --format 'value(status.url)'
```

---

## 🆘 문제 해결

### 1. 빌드 실패: API 키 관련

**증상:**
```
ERROR: UPSTAGE_API_KEY not found
```

**해결:**
- GitHub Secrets에 `UPSTAGE_API_KEY` 등록 확인
- Secret 이름 대소문자 정확히 확인

### 2. 메모리 부족

**증상:**
```
Container failed to start. Failed to start and listen on the port
```

**해결:**
```bash
gcloud run services update bbumate-api \
  --memory 1Gi \
  --region asia-northeast3
```

### 3. Cold Start 느림

**증상:**
- 첫 요청 시 5-10초 걸림

**해결 방법 A (비용 증가):**
```bash
gcloud run services update bbumate-api \
  --min-instances 1 \
  --region asia-northeast3
```

**해결 방법 B (무료):**
- Cloud Scheduler로 5분마다 헬스체크 요청

### 4. ChromaDB 데이터 업데이트

**방법:**
1. `data/` 폴더의 PDF/HTML 파일 수정
2. GitHub에 push
3. 자동으로 재빌드 & 배포

**주의:**
- 빌드마다 ChromaDB가 새로 생성됨
- 일관된 데이터 보장
- `chroma_storage/` 폴더는 Git에 커밋 불필요

### 5. TAVILY_API_KEY 관련 경고

**증상:**
```
TAVILY_API_KEY not found
```

**해결:**
- 현재 `USE_MOCK_WEB_SEARCH=true` 설정으로 Mock 사용
- 실제 API 필요 없음, 무시해도 됨
- 실제 웹 검색 필요 시 Tavily API 키 등록

---

## ✅ 배포 완료 확인

### 1. Cloud Run 콘솔 확인
👉 https://console.cloud.google.com/run

### 2. 서비스 URL 접속
```bash
# URL 확인
gcloud run services describe bbumate-api \
  --region asia-northeast3 \
  --format 'value(status.url)'

# 헬스체크
curl https://your-service-url/api/health
```

### 3. API 테스트
```bash
curl -X POST https://your-service-url/api/query \
  -H "Content-Type: application/json" \
  -d '{"question": "신혼부부 전세자금 대출이 뭐야?"}'
```

**예상 응답:**
```json
{
  "answer": "신혼부부 전세자금 대출은...",
  "answer_md": "# 답변\n...",
  "answer_html": "<h1>답변</h1>...",
  "sources": [...]
}
```

---

## 📚 추가 자료

- [Cloud Run 공식 문서](https://cloud.google.com/run/docs)
- [Upstage API 문서](https://console.upstage.ai/docs)

---

## 🎓 참고: 아키텍처 상세

### ChromaDB 자동 생성 방식

**Dockerfile:**
```dockerfile
# 빌드 인자로 API 키 받기
ARG UPSTAGE_API_KEY

# Ingestion 자동 실행
RUN UPSTAGE_API_KEY=${UPSTAGE_API_KEY} \
    python run_ingestion.py

# 생성된 ChromaDB 확인
RUN ls -la chroma_storage/
```

**장점:**
- ✅ 로컬 환경에 의존하지 않음
- ✅ 빌드마다 일관된 데이터
- ✅ 팀원 간 데이터 불일치 문제 해결
- ✅ Git에 ChromaDB 커밋 불필요

**처리 데이터:**
- d001: 211 chunks (주거정책)
- d002: 184 chunks (대출정책)
- d003: 632 chunks (HTML)
- d004: 100 chunks (PDF)
- d005: 29 chunks (기타)
- **총: 1,156 chunks (37MB)**

---

**배포 완료를 축하합니다! 🎉**
