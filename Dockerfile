# Google Cloud Run용 Dockerfile
FROM python:3.11.9-slim

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 패키지 업데이트 및 필수 패키지 설치
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python 의존성 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드 및 데이터 파일 복사
COPY . .

# 빌드 시점에 Ingestion 실행을 위한 환경 변수 받기
ARG UPSTAGE_API_KEY
ARG UPSTAGE_EMBEDDING_MODEL=solar-embedding-1-large
ARG UPSTAGE_CHAT_MODEL=solar-1-mini-chat

# Ingestion 실행 (빌드 시점에 ChromaDB 생성)
RUN UPSTAGE_API_KEY=${UPSTAGE_API_KEY} \
    UPSTAGE_EMBEDDING_MODEL=${UPSTAGE_EMBEDDING_MODEL} \
    UPSTAGE_CHAT_MODEL=${UPSTAGE_CHAT_MODEL} \
    python run_ingestion.py

# 생성된 ChromaDB 확인
RUN ls -la chroma_storage/ && echo "ChromaDB built successfully!"

# 포트 설정 (Cloud Run은 PORT 환경 변수 사용)
ENV PORT=8080

# Non-root 사용자로 실행
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# 헬스체크 (선택사항)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT}/api/health || exit 1

# 애플리케이션 실행
CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT} --workers 1
