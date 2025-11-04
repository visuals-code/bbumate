"""8. Run a test query against the RAG pipeline: 검색된 청크와 최종 답변을 출력하는 질의 테스트 스크립트"""

import argparse
import html
import time
import re
from typing import List

from langchain_core.documents import Document

from src.chains.d003.chain import answer_question
from src.retrieval.d003.retriever import retrieve_relevant_documents


# CLI 인자 정의/파싱을 해 주는 함수: argparse로 스크립트 실행 시 받는 옵션들을 등록하고, 실제 값들을 argparse.Namespace로 반환
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a test query against the RAG pipeline"
    )
    parser.add_argument("--question", type=str, required=True, help="User question")
    parser.add_argument("--k", type=int, default=3, help="Top-k chunks to retrieve")
    parser.add_argument(
        "--output-format",
        type=str,
        default="md",
        choices=["md", "html"],
        help="Format of final answer: 'md' or 'html'",
    )
    return parser.parse_args()


def print_sources(docs: List[Document], max_chars: int = 160):
    print("\n[Retrieved Chunks]")
    for i, d in enumerate(docs, 1):
        src = (d.metadata or {}).get("source", "unknown")
        snippet = (d.page_content or "")[:max_chars].replace("\n", " ")
        print(f"  {i}. source={src}\n     {snippet}...")


def format_markdown_answer(answer: str, docs: List[Document]) -> str:
    lines: List[str] = []
    prepared = insert_line_breaks_korean(answer.strip())
    lines.append(emphasize_price_terms_md(prepared))
    return "\n".join(lines)


def format_html_answer(answer: str, docs: List[Document]) -> str:
    parts: List[str] = []
    prepared = insert_line_breaks_korean(answer.strip())
    escaped = html.escape(prepared)
    emphasized = emphasize_price_terms_html_escaped(escaped)
    parts.append("<div>" + emphasized.replace("\n", "<br/>") + "</div>")
    return "\n".join(parts)


# 가격 용어 강조 함수: 가격 용어를 볼드처리하여 가독성 향상 (Markdown 형식)
def emphasize_price_terms_md(text: str) -> str:
    patterns = [
        r"(?:(?:매?월|연)\s*)?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?\s*원",
        r"\d+(?:\.\d+)?\s*%",
    ]

    def repl(m: re.Match) -> str:
        return f"**{m.group(0)}**"

    for p in patterns:
        text = re.sub(p, repl, text)
    return text


# 가격 용어 강조 함수: 가격 용어를 볼드처리하여 가독성 향상 (HTML 형식)
def emphasize_price_terms_html_escaped(escaped_text: str) -> str:
    patterns = [
        r"(?:(?:매?월|연)\s*)?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?\s*원",
        r"\d+(?:\.\d+)?\s*%",
    ]

    def repl(m: re.Match) -> str:
        return f"<strong>{m.group(0)}</strong>"

    for p in patterns:
        escaped_text = re.sub(p, repl, escaped_text)
    return escaped_text


# 줄바꿈 추가 함수: 한글 문장 끝에 줄바꿈을 추가하여 가독성 향상
def insert_line_breaks_korean(text: str) -> str:
    text = re.sub(r"(다|요|니다)\.(\s+)", r"\1.\n\n", text)
    text = re.sub(r"\s+(또한|그리고|한편|추가로|더불어|다만|참고로),", r"\n\n\1,", text)
    return text


def main() -> None:
    args = parse_args()

    docs = retrieve_relevant_documents(args.question, k=args.k)
    print_sources(docs)

    start_time = time.perf_counter()
    answer, used_docs = answer_question(args.question, k=args.k)

    if args.output_format == "html":
        formatted = format_html_answer(answer, used_docs)
    else:
        formatted = format_markdown_answer(answer, used_docs)

    elapsed_s = time.perf_counter() - start_time
    print(f"\n[TIME] Answer generated in {elapsed_s:.2f}s")

    print("\n[Answer]")
    print(formatted)


if __name__ == "__main__":
    main()
