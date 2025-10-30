import os
import json
import argparse
from langchain.text_splitter import RecursiveCharacterTextSplitter


def chunk_html_files(input_dir, output_path):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

    all_chunks = []
    total_length = 0

    for file in os.listdir(input_dir):
        if file.endswith(".html"):
            with open(os.path.join(input_dir, file), "r", encoding="utf-8") as f:
                text = f.read()

            chunks = splitter.split_text(text)
            for i, chunk in enumerate(chunks):
                all_chunks.append({"source": file, "chunk_id": i, "text": chunk})
                total_length += len(chunk)

            print(f"📄 {file} → {len(chunks)}개 청크로 분할 완료")

    avg_length = total_length / len(all_chunks) if all_chunks else 0

    # JSON 파일로 저장
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 전체 {len(all_chunks)}개 청크 저장 완료: {output_path}")
    print(f"📊 평균 청크 길이: {avg_length:.1f}자")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", required=True, help="도메인 이름 예: d002")
    args = parser.parse_args()

    base_dir = f"data/{args.domain}"
    input_dir = os.path.join(base_dir, "htmls")
    output_path = os.path.join(base_dir, "chunks", "chunks.json")

    chunk_html_files(input_dir, output_path)
