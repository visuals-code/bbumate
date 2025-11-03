from langchain_community.document_loaders import PDFMinerPDFasHTMLLoader
from bs4 import BeautifulSoup
import re
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tiktoken import get_encoding


def process_pdf_to_semantic_chunks(file_path):

    # 1. HTML 로드
    loader = PDFMinerPDFasHTMLLoader(file_path)
    docs = loader.load()
    if not docs:
        return []

    # 2. HTML 파싱 및 글꼴 스니펫 추출
    soup = BeautifulSoup(docs[0].page_content, "html.parser")
    content = soup.find_all("div")

    cur_fs = None
    cur_text = ""
    snippets = []

    for c in content:
        sp = c.find("span")
        if not sp:
            continue
        st = sp.get("style")
        if not st:
            continue
        fs_match = re.findall("font-size:(\\d+)px", st)
        if not fs_match:
            continue

        fs = int(fs_match[0])

        if not cur_fs:
            cur_fs = fs

        if fs == cur_fs:
            cur_text += c.text
        else:
            snippets.append((cur_text, cur_fs))
            cur_fs = fs
            cur_text = c.text

    if cur_text:
        snippets.append((cur_text, cur_fs))

    # 3. Semantic Chunking
    cur_idx = -1
    semantic_snippets = []

    for s in snippets:
        # 파일 경로를 메타데이터에 추가
        metadata_base = docs[0].metadata.copy()
        metadata_base["source_file"] = file_path

        if (
            not semantic_snippets
            or s[1] > semantic_snippets[cur_idx].metadata["heading_font"]
        ):
            metadata = {"heading": s[0], "content_font": 0, "heading_font": s[1]}
            metadata.update(metadata_base)
            semantic_snippets.append(Document(page_content="", metadata=metadata))
            cur_idx += 1
            continue

        if (
            not semantic_snippets[cur_idx].metadata["content_font"]
            or s[1] <= semantic_snippets[cur_idx].metadata["content_font"]
        ):
            semantic_snippets[cur_idx].page_content += s[0]
            semantic_snippets[cur_idx].metadata["content_font"] = max(
                s[1], semantic_snippets[cur_idx].metadata["content_font"]
            )
            continue

        metadata = {"heading": s[0], "content_font": 0, "heading_font": s[1]}
        metadata.update(metadata_base)
        semantic_snippets.append(Document(page_content="", metadata=metadata))
        cur_idx += 1

    # 청크 사이즈 제한으로 인한
    CHUNK_SIZE_LIMIT = 1500
    CHUNK_OVERLAP = 0

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=CHUNK_SIZE_LIMIT,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )

    resharded_snippets = []
    total_new_chunks = 0

    for doc in semantic_snippets:
        # metadata를 유지하면서 청크 재분할
        new_chunks = text_splitter.split_documents([doc])
        resharded_snippets.extend(new_chunks)
        total_new_chunks += len(new_chunks)

    # 재분할된 리스트를 다음 단계(4단계)의 입력으로 사용하도록 변경
    semantic_snippets = resharded_snippets
    print(f"    - 길이 기반 재분할 후 총 {len(semantic_snippets)}개의 청크 생성")

    # 4. 빈 문서 필터링
    valid_snippets = []
    filtered_count = 0

    for doc in semantic_snippets:
        if doc.page_content and doc.page_content.strip():
            valid_snippets.append(doc)
        else:
            filtered_count += 1

    if filtered_count > 0:
        print(f"    - 빈 문서 {filtered_count}개 제거 ")

    return valid_snippets
