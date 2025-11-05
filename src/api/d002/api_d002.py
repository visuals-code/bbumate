import logging
import re
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from src.chains.d002.rag_chain import run_rag


# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


app = FastAPI(
    title="신혼부부 정책 RAG 테스트 서버",
    description="신혼부부 지원정책 관련 질문에 답변하는 RAG 시스템",
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080", "http://127.0.0.1:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 요청 Body 모델 정의
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500, description="사용자 질문")
    region: Optional[str] = Field(None, description="사전 선택 지역 (예: 인천, 서울, 경기)")
    housing_type: Optional[str] = Field(None, description="사전 선택 주거형태 (예: 전세, 월세, 자가, 매매)")

    @field_validator("question")
    @classmethod
    def validate_question(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("질문은 비어있을 수 없습니다")
        return v

    @field_validator("region", "housing_type", mode="before")
    @classmethod
    def filter_empty_strings(cls, v: Optional[str]) -> Optional[str]:
        """빈 문자열을 None으로 변환 (필터링)."""
        if v is None:
            return None
        if isinstance(v, str) and v.strip() == "":
            return None
        return v.strip() if isinstance(v, str) else v


# Source 모델
class SourceItem(BaseModel):
    title: str
    url: Optional[str] = None
    source: str


# 응답 Body 모델 정의
class QueryResponse(BaseModel):
    answer: str = Field(..., description="생성된 답변 (순수 텍스트)")
    answer_md: str = Field(..., description="생성된 답변 (마크다운)")
    answer_html: str = Field(..., description="생성된 답변 (HTML)")
    sources: List[SourceItem] = Field(default=[], description="참고 문서 출처")


def markdown_to_text(text: str) -> str:
    """마크다운 텍스트를 순수 텍스트로 변환.
    
    - 마크다운 문법 제거 (**, ## 등)
    """
    # 굵게 처리 제거: **text** -> text
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    
    # 헤더 제거: ## text -> text
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    
    # 리스트 기호 제거: - text -> text
    text = re.sub(r'^[-*+]\s+', '', text, flags=re.MULTILINE)
    
    # 링크 제거: [text](url) -> text
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    
    return text.strip()


def format_to_markdown(text: str) -> str:
    """마크다운 텍스트를 정리 및 보강.
    
    - 이미 마크다운 형식이므로 추가 보강만 수행
    - 연속된 줄바꿈 정리
    """
    # 연속된 줄바꿈 정리 (3개 이상 -> 2개)
    text = re.sub(r'\n{3,}', r'\n\n', text)
    
    return text.strip()


def markdown_to_html(text: str) -> str:
    """마크다운 텍스트를 HTML 형식으로 변환.
    
    - **text** -> <strong>text</strong>
    - 줄바꿈 -> <br/>
    - 문단 구분 유지
    """
    # 굵게 처리: **text** -> <strong>text</strong>
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    
    # 헤더 처리: ## text -> <h2>text</h2>
    text = re.sub(r'^### (.+?)$', r'<h3>\1</h3>', text, flags=re.MULTILINE)
    text = re.sub(r'^## (.+?)$', r'<h2>\1</h2>', text, flags=re.MULTILINE)
    text = re.sub(r'^# (.+?)$', r'<h1>\1</h1>', text, flags=re.MULTILINE)
    
    # 리스트 처리: - text -> <li>text</li>
    lines = text.split('\n')
    in_list = False
    html_lines = []
    
    for line in lines:
        if re.match(r'^[-*+]\s+', line):
            if not in_list:
                html_lines.append('<ul>')
                in_list = True
            list_item = re.sub(r'^[-*+]\s+', '', line)
            html_lines.append(f'<li>{list_item}</li>')
        else:
            if in_list:
                html_lines.append('</ul>')
                in_list = False
            if line.strip():
                html_lines.append(f'<p>{line}</p>')
            else:
                html_lines.append('<br/>')
    
    if in_list:
        html_lines.append('</ul>')
    
    text = '\n'.join(html_lines)
    
    # 문단 구분: 연속된 줄바꿈을 <br/><br/>로 변환
    text = re.sub(r'\n{2,}', r'<br/><br/>', text)
    
    # 단일 줄바꿈을 <br/>로 변환
    text = text.replace('\n', '<br/>')
    
    # 연속된 <br/> 정리
    text = re.sub(r'(<br/>){3,}', r'<br/><br/>', text)
    
    return f"<div>{text}</div>"


def format_sources(sources, domain: str = "d002") -> List[SourceItem]:
    """sources 리스트/딕셔너리를 SourceItem 객체 리스트로 변환.
    
    - sources는 문자열 리스트 또는 {"title": str, "url": str} 딕셔너리 리스트
    - URL이 있으면 url 필드에, 없으면 null
    - title은 파일명 또는 URL에서 추출
    - source는 원본 파일명 또는 "웹 검색"
    """
    from src.utils.d002.loaders import load_document_links
    
    formatted_sources = []
    links = load_document_links(domain)
    
    for source_item in sources:
        # 딕셔너리 형태 (웹 검색 결과)
        if isinstance(source_item, dict):
            formatted_sources.append(
                SourceItem(
                    title=source_item.get("title", "웹 검색 결과"),
                    url=source_item.get("url"),
                    source="웹 검색"
                )
            )
        # 문자열 형태 (문서 파일명 또는 "웹 검색")
        elif isinstance(source_item, str):
            if source_item == "웹 검색" or source_item.startswith("http"):
                formatted_sources.append(
                    SourceItem(
                        title="웹 검색 결과",
                        url=source_item if source_item.startswith("http") else None,
                        source="웹 검색"
                    )
                )
            else:
                # 문서인 경우
                url = links.get(source_item) if source_item in links else None
                # 파일명에서 확장자 제거하고 title로 사용
                title = source_item.replace(".html", "").replace(".pdf", "").replace("_", " ")
                
                formatted_sources.append(
                    SourceItem(
                        title=title,
                        url=url,
                        source=f"data/{domain}/{source_item}" if not source_item.startswith("data/") else source_item
                    )
                )
    
    return formatted_sources


# --- 헬스체크 ---
@app.get("/")
def health_check():
    """서버 상태 확인"""
    return {"status": "ok", "service": "신혼부부 정책 RAG API", "version": "0.1.0"}


@app.post("/query", response_model=QueryResponse)
def query_question(request: QueryRequest):
    """
    신혼부부 정책 관련 질문에 답변합니다.

    - **question**: 질문 내용 (1~500자)
    """
    try:
        import time
        start_time = time.perf_counter()
        
        logger.info(f"질문 수신: {request.question[:50]}...")

        res = run_rag(
            request.question,
            region=request.region,
            housing_type=request.housing_type,
        )

        answer = res.get("answer", "답변 생성 실패")
        sources_list = res.get("sources", [])
        
        # LLM이 마크다운 형식으로 생성하므로 변환
        answer_text = markdown_to_text(answer)  # 순수 텍스트
        answer_md = format_to_markdown(answer)  # 마크다운 (정리)
        answer_html = markdown_to_html(answer)  # HTML 변환
        
        # sources 객체 배열로 변환
        sources = format_sources(sources_list)

        response = QueryResponse(
            answer=answer_text,
            answer_md=answer_md,
            answer_html=answer_html,
            sources=sources,
        )

        elapsed_time = time.perf_counter() - start_time
        logger.info(f"응답 완료 ({len(sources)}개 출처, 소요 시간: {elapsed_time:.2f}초)")
        return response

    except ValueError as e:
        logger.warning(f"입력 오류: {e}")
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"서버 오류: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"서버 오류: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.d002.api_d002:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
