from langchain.document_loaders import PyPDFLoader
from pathlib import Path
import argparse, logging
from html import escape


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def pdf_to_html(pdf_path: Path, html_path: Path) -> bool:
    """PDF를 HTML로 변환. 성공 시 True 반환.

    - PyPDFLoader로 페이지 텍스트를 추출하고, 개행을 보존하여 가독성을 유지한다.
    - HTML는 보안/표현을 위해 pre+escape 조합으로 감싼다.
    """
    try:
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()

        if not pages:
            logger.warning(f"PDF가 비어있음: {pdf_path.name}")
            return False

        text = "\n".join([doc.page_content for doc in pages])

        # HTML escape 처리를 위한 pre 태그 사용
        # - 원문에 포함된 태그/특수문자에 의한 레이아웃 깨짐과 XSS 위험을 방지
        html_content = f"""<!DOCTYPE html>
        <html lang="ko">
            <head>
                <meta charset="UTF-8">
                <title>{pdf_path.stem}</title>
            </head>
            <body>
                <pre>{escape(text)}</pre>
            </body>
        </html>"""

        html_path.write_text(html_content, encoding="utf-8")
        return True

    except Exception as e:
        logger.error(f"{pdf_path.name} 변환 실패: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="PDF를 HTML로 일괄 변환")
    parser.add_argument("--domain", required=True, help="도메인 이름")
    args = parser.parse_args()

    base_dir = Path("data") / args.domain
    pdf_dir = base_dir / "pdfs"
    html_dir = base_dir / "htmls"

    if not pdf_dir.exists():
        logger.error(f"PDF 디렉토리가 없음: {pdf_dir}")
        return

    html_dir.mkdir(parents=True, exist_ok=True)

    pdf_files = list(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        logger.warning(f"변환할 PDF가 없음: {pdf_dir}")
        return

    success_count = 0
    for pdf_file in pdf_files:
        logger.info(f"변환 중: {pdf_file.name}")
        html_file = html_dir / f"{pdf_file.stem}.html"

        if pdf_to_html(pdf_file, html_file):
            success_count += 1

    logger.info(f"변환 완료: {success_count}/{len(pdf_files)}개 성공")
    logger.info(f"출력 경로: {html_dir}")


if __name__ == "__main__":
    main()
