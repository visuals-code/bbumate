"""1. Load documents"""

import os
import json
import glob
from typing import List, Literal, Dict, Optional

from langchain_core.documents import Document


LoadMode = Literal["pdf", "html"]


def list_pdf_paths(root_dir: str) -> List[str]:
    """
    Return a sorted list of PDF file paths under any 'data' directory:
    'data' 디렉터리 하위의 모든 PDF 파일 경로를 재귀적으로 찾아 정렬하여 반환
    """
    pattern = os.path.join("**", "data", "**", "*.pdf")
    return sorted(glob.glob(pattern, recursive=True))


def _load_with_pypdf_loader(pdf_path: str) -> List[Document]:
    """
    Try to load a PDF using LangChain's PyPDFLoader:
    PyPDFLoader를 사용하여 PDF 파일을 로드해 Document 객체 리스트로 반환
    """
    try:
        from langchain_community.document_loaders import PyPDFLoader
    except ImportError:
        raise RuntimeError("PyPDFLoader is unavailable. Install 'pypdf'.")
    else:
        loader = PyPDFLoader(pdf_path)
        return loader.load()


def _extract_text_with_pdfplumber(pdf_path: str) -> str:
    """
    Extract raw text using pdfplumber as a lightweight fallback:
    pdfplumber로 PDF의 텍스트를 간단히 추출해 하나의 문자열로 반환
    """
    try:
        import pdfplumber
    except ImportError:
        raise RuntimeError(
            "pdfplumber is unavailable. Install 'pdfplumber' or enable PyPDFLoader."
        )
    else:
        with pdfplumber.open(pdf_path) as pdf:
            return "\n".join(page.extract_text() or "" for page in pdf.pages)


def _to_minimal_html(text: str) -> str:
    """
    Wrap plain text into minimal HTML paragraphs for downstream HTML-aware flows:
    HTML 태그를 최소화하여 텍스트를 간단한 HTML 문서로 변환
    """

    import html

    lines = [line.strip() for line in text.splitlines()]
    paragraphs: List[str] = []
    buffer: List[str] = []
    for line in lines:
        if not line:
            if buffer:
                paragraphs.append("<p>" + html.escape(" ".join(buffer)) + "</p>")
                buffer = []
            continue
        buffer.append(line)
    if buffer:
        paragraphs.append("<p>" + html.escape(" ".join(buffer)) + "</p>")
    return "\n".join(paragraphs) if paragraphs else "<p></p>"


def load_documents(data_dir: str, mode: LoadMode = "pdf") -> List[Document]:
    """
    Load PDF documents from a directory:

    - mode="pdf": Use PyPDFLoader (preferred). If unavailable, raise with guidance.
    - mode="html": Extract raw text (pdfplumber) and wrap into minimal HTML; store in Document.page_content.
    Returns a list of LangChain Documents with metadata including source path and loader mode.
    """
    pdf_paths = list_pdf_paths(data_dir)
    url_map = _load_url_map(data_dir)
    all_docs: List[Document] = []

    for path in pdf_paths:
        if mode == "pdf":
            docs = _load_with_pypdf_loader(path)
            for d in docs:
                enriched = {**(d.metadata or {}), "source": path, "loader": "pypdf"}
                mapped_url = _lookup_url(url_map, data_dir, path)
                if mapped_url:
                    enriched["url"] = mapped_url
                d.metadata = enriched
            all_docs.extend(docs)
        else:
            raw_text = _extract_text_with_pdfplumber(path)
            html_text = _to_minimal_html(raw_text)
            meta: Dict[str, str] = {"source": path, "loader": "pdfplumber_html"}
            mapped_url = _lookup_url(url_map, data_dir, path)
            if mapped_url:
                meta["url"] = mapped_url
            all_docs.append(Document(page_content=html_text, metadata=meta))

    return all_docs


def _load_url_map(data_dir: str) -> Dict[str, str]:
    """Load an optional URL mapping from JSON.

    Priority:
    1) URL_MAP_JSON env var path
    2) <data_dir>/url_map.json
    3) ./data/url_map.json (repo default)

    The JSON should be an object mapping keys to URLs. Keys can be:
    - full path (as stored in metadata["source"])
    - path relative to data_dir
    - base filename (with or without extension)
    """
    candidates: List[str] = []
    env_path = os.getenv("URL_MAP_JSON")
    if env_path:
        candidates.append(env_path)
    candidates.append(os.path.join(data_dir, "url_map.json"))
    candidates.append(os.path.join("data", "url_map.json"))

    for p in candidates:
        try:
            if os.path.isfile(p):
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    # normalize keys to strings
                    return {str(k): str(v) for k, v in data.items()}
        except Exception:
            # best-effort; ignore malformed files
            pass
    return {}


def _lookup_url(url_map: Dict[str, str], data_dir: str, abs_path: str) -> Optional[str]:
    """Find URL for a given absolute path using several key strategies."""
    if not url_map:
        return None
    # candidates: absolute, relative to data_dir, basename, stem
    rel = os.path.relpath(abs_path, start=data_dir)
    base = os.path.basename(abs_path)
    stem, _ = os.path.splitext(base)
    for key in (abs_path, rel, base, stem):
        url = url_map.get(key)
        if url:
            return url
    return None
