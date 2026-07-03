# src/ingest.py
# ─────────────────────────────────────────────────────────────
# PURPOSE: Load contract files (PDF / DOCX) from the data/contracts/
#          folder, clean the text, and split it into overlapping
#          chunks ready for embedding.
# ─────────────────────────────────────────────────────────────

import os
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


# ── Config ────────────────────────────────────────────────────
CONTRACTS_DIR = Path("C:\\Users\\Prerna\\OneDrive\\Documents\\GENAICRAG\\data\\contracts")   # drop your files here

# Chunk size: ~500 tokens is a good sweet spot for legal text.
# Overlap of 100 ensures a clause split across two chunks isn't lost.
CHUNK_SIZE    = 500
CHUNK_OVERLAP = 100


# ── Helpers ───────────────────────────────────────────────────

def _clean_text(text: str) -> str:
    """
    Remove common PDF artifacts:
      - excessive whitespace / newlines
      - form-feed characters (\x0c)
    """
    import re
    text = text.replace("\x0c", " ")         # form-feed → space
    text = re.sub(r"\n{3,}", "\n\n", text)   # 3+ newlines → 2
    text = re.sub(r"[ \t]{2,}", " ", text)   # multiple spaces → one
    return text.strip()


def _load_single_file(file_path: Path) -> list[Document]:
    """
    Load one contract file.
    Supports: .pdf, .docx
    Returns a list of LangChain Document objects.
    Each Document has:
      - page_content : the raw text of that page / section
      - metadata     : source filename, page number (PDFs)
    """
    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        loader = PyPDFLoader(str(file_path))
    elif suffix == ".docx":
        loader = Docx2txtLoader(str(file_path))
    else:
        print(f"[ingest] Skipping unsupported file type: {file_path.name}")
        return []

    docs = loader.load()

    # Clean text and tag with the source filename
    for doc in docs:
        doc.page_content = _clean_text(doc.page_content)
        doc.metadata["source"] = file_path.name   # e.g. "nda_2024.pdf"

    return docs


# ── Main public function ───────────────────────────────────────

def load_and_chunk_contracts(contracts_dir: Path = CONTRACTS_DIR) -> list[Document]:
    """
    1. Walks contracts_dir and loads every PDF / DOCX file.
    2. Cleans the extracted text.
    3. Splits documents into overlapping chunks.
    4. Returns the final list of chunk Documents.

    Each returned Document has metadata:
      {
        "source": "filename.pdf",
        "page"  : 0,            # (PDFs only, 0-indexed)
        "chunk" : 3             # chunk index within that file
      }
    """

    if not contracts_dir.exists():
        raise FileNotFoundError(
            f"Contracts folder not found: {contracts_dir}\n"
            "Create the folder and add your PDF / DOCX files."
        )

    # ── Step 1: Load all files ─────────────────────────────────
    all_docs: list[Document] = []
    files = list(contracts_dir.glob("**/*"))   # recursive search

    for file_path in files:
        if file_path.is_file():
            print(f"[ingest] Loading: {file_path.name}")
            docs = _load_single_file(file_path)
            all_docs.extend(docs)

    if not all_docs:
        raise ValueError(
            "No supported files found in contracts/. "
            "Add .pdf or .docx files and try again."
        )

    print(f"[ingest] Loaded {len(all_docs)} page(s) from "
          f"{len(set(d.metadata['source'] for d in all_docs))} file(s).")

    # ── Step 2: Split into chunks ──────────────────────────────
    # RecursiveCharacterTextSplitter tries to split on:
    #   "\n\n"  (paragraphs)  →  "\n"  (lines)  →  " "  (words)
    # It never cuts a chunk in the middle of a word.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(all_docs)

    # ── Step 3: Tag each chunk with its index ──────────────────
    source_counter: dict[str, int] = {}
    for chunk in chunks:
        src = chunk.metadata["source"]
        idx = source_counter.get(src, 0)
        chunk.metadata["chunk"] = idx
        source_counter[src] = idx + 1

    print(f"[ingest] Split into {len(chunks)} chunk(s). "
          f"chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}")

    return chunks


# ── Quick test (run this file directly to verify) ─────────────
if __name__ == "__main__":
    chunks = load_and_chunk_contracts()

    print("\n── Sample chunk ──────────────────────────────")
    sample = chunks[0]
    print(f"Source : {sample.metadata['source']}")
    print(f"Chunk  : {sample.metadata['chunk']}")
    print(f"Length : {len(sample.page_content)} chars")
    print(f"Preview: {sample.page_content[:300]}")