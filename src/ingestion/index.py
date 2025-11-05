"""Ingestion index: bridge to d003 ingestion pipeline pieces.

Console-friendly `ingest` that mirrors src/ingestion/d003/pipeline.py
logging and behavior (load → split → embed → persist with timing).
Switch domain by editing imports here.
"""

import os
import time
from typing import Optional
from dotenv import load_dotenv

from src.ingestion.d003.loader import load_documents
from src.ingestion.d003.splitter import split_documents
from src.ingestion.d003.embedder import get_upstage_embeddings
from src.ingestion.d003.vectorstore import persist_to_chroma


def ingest(
    data_dir: str = "data",
    mode: str = "pdf",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    persist_dir: Optional[str] = None,
) -> None:
    load_dotenv()

    if not os.path.isdir(data_dir):
        print(f"[ERROR] Data directory not found: {data_dir}")
        return

    print(f"[1/4] Loading documents from {data_dir} (mode={mode}) ...")
    docs = load_documents(data_dir, mode=("pdf" if mode == "pdf" else "html"))
    if not docs:
        print("[WARN] No documents found. Exiting.")
        return

    print(f"[2/4] Splitting {len(docs)} documents ...")
    chunks = split_documents(
        docs,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        strip_html=(mode == "html"),
    )
    print(f"[INFO] Created {len(chunks)} chunks.")

    print("[3/4] Initializing embeddings ...")
    embeddings = get_upstage_embeddings()

    print("[4/4] Persisting to Chroma ...")
    start_time = time.perf_counter()
    persist_to_chroma(chunks, embedding_function=embeddings)
    elapsed_s = time.perf_counter() - start_time
    print(f"[TIME] Embedded and persisted {len(chunks)} chunks in {elapsed_s:.2f}s")

    print("[DONE] Ingestion completed.")


__all__ = ["ingest"]
