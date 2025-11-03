"""ingestion pipeline
load documents -> split -> embed -> persist to chroma
"""

import argparse
import os
import sys
from typing import Literal

from dotenv import load_dotenv

from .loader import load_documents
from .splitter import split_documents
from .embedder import get_upstage_embeddings
from .vectorstore import persist_to_chroma


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest documents into Chroma database"
    )
    parser.add_argument(
        "--data-dir", type=str, default="data", help="PDF root directory"
    )
    parser.add_argument(
        "--persist-dir", type=str, default=None, help="Chroma persist directory"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="pdf",
        choices=["pdf", "html"],
        help="Loader mode: 'pdf' (PyPDFLoader) or 'html' (pdfplumber to minimal HTML)",
    )
    parser.add_argument("--chunk-size", type=int, default=1000, help="Chunk size")
    parser.add_argument("--chunk-overlap", type=int, default=200, help="Chunk overlap")
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()

    if not os.path.isdir(args.data_dir):
        print(f"[ERROR] Data directory not found: {args.data_dir}")
        sys.exit(1)

    print(f"[1/4] Loading documents from {args.data_dir} (mode={args.mode}) ...")
    docs = load_documents(args.data_dir, mode=cast_mode(args.mode))
    if not docs:
        print("[WARN] No documents found. Exiting.")
        return

    print(f"[2/4] Splitting {len(docs)} documents ...")
    chunks = split_documents(
        docs,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        strip_html=(args.mode == "html"),
    )
    print(f"[INFO] Created {len(chunks)} chunks.")

    print("[3/4] Initializing embeddings ...")
    embeddings = get_upstage_embeddings()

    print("[4/4] Persisting to Chroma ...")
    persist_to_chroma(
        chunks, persist_directory=args.persist_dir, embedding_function=embeddings
    )

    print("[DONE] Ingestion completed.")


def cast_mode(mode_str: str) -> Literal["pdf", "html"]:
    return "pdf" if mode_str == "pdf" else "html"


if __name__ == "__main__":
    main()
